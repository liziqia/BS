import json
from time import time
import argparse
import logging
import os
from pathlib import Path
import math
import matplotlib.pyplot as plt

import numpy as np
from PIL import Image
from copy import deepcopy

import torch
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms

from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration, set_seed
from diffusers.optimization import get_scheduler
from accelerate.utils import DistributedType
from peft import LoraConfig, set_peft_model_state_dict, PeftModel, get_peft_model
from peft.utils import get_peft_model_state_dict
from huggingface_hub import snapshot_download
from safetensors.torch import save_file

from diffusers.models import AutoencoderKL

from Ship import OmniGen, OmniGenProcessor
from Ship.train_helper import DatasetFromJson, TrainDataCollator
from Ship.train_helper import training_losses
from Ship.utils import (
    create_logger,
    update_ema,
    requires_grad,
    center_crop_arr,
    crop_arr,
    vae_encode,
    vae_encode_list
)


def main(args):
    # Setup accelerator:
    from accelerate import DistributedDataParallelKwargs as DDPK
    kwargs = DDPK(find_unused_parameters=False)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_dir=args.results_dir,
        kwargs_handlers=[kwargs],
        )
    device = accelerator.device
    accelerator.init_trackers("tensorboard_log", config=args.__dict__)

    # Setup an experiment folder:
    os.makedirs(args.results_dir, exist_ok=True)
    logger = create_logger(args.results_dir)
    checkpoint_dir = f"{args.results_dir}/checkpoints"  # Stores saved model checkpoints
    if accelerator.is_main_process:
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger.info(f"Experiment directory created at {args.results_dir}")
        json.dump(args.__dict__, open(os.path.join(args.results_dir, 'train_args.json'), 'w'))


    # Create model:    
    if not os.path.exists(args.model_name_or_path):
        cache_folder = os.getenv('HF_HUB_CACHE')
        args.model_name_or_path = snapshot_download(repo_id=args.model_name_or_path,
                                        cache_dir=cache_folder,
                                        ignore_patterns=['flax_model.msgpack', 'rust_model.ot', 'tf_model.h5'])
        logger.info(f"Downloaded model to {args.model_name_or_path}")
    model = OmniGen.from_pretrained(args.model_name_or_path)
    model.llm.config.use_cache = False
    model.llm.gradient_checkpointing_enable()
    model = model.to(device)

    # 初始化 loss 记录列表
    all_losses = []
    epoch_losses = []

    if args.vae_path is None:
        vae_path = os.path.join(args.model_name_or_path, "vae")
        if os.path.exists(vae_path):
            vae = AutoencoderKL.from_pretrained(vae_path).to(device)
        else:
            logger.info("No VAE found in model, downloading stabilityai/sdxl-vae from HF")
            logger.info("If you have VAE in local folder, please specify the path with --vae_path")
            vae = AutoencoderKL.from_pretrained("stabilityai/sdxl-vae").to(device)
    else:
        vae = AutoencoderKL.from_pretrained(args.vae_path).to(device)

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
    vae.to(dtype=torch.float32)
    model.to(weight_dtype)

    processor = OmniGenProcessor.from_pretrained(args.model_name_or_path)

    requires_grad(vae, False)
    if args.use_lora:
        if accelerator.distributed_type == DistributedType.FSDP:
            raise NotImplementedError("FSDP does not support LoRA")
        requires_grad(model, False)
        transformer_lora_config = LoraConfig(
            r=args.lora_rank,
            lora_alpha=args.lora_rank * 2,
            init_lora_weights="gaussian",
            target_modules=["qkv_proj", "o_proj"],
        )
        model.llm.enable_input_require_grads()
        model = get_peft_model(model, transformer_lora_config)
        model.to(weight_dtype)
        transformer_lora_parameters = list(filter(lambda p: p.requires_grad, model.parameters()))
        opt = torch.optim.AdamW(transformer_lora_parameters, lr=args.lr, weight_decay=args.adam_weight_decay)
    else:
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.adam_weight_decay)

    ema = None
    if args.use_ema:
        ema = deepcopy(model).to(device)  # Create an EMA of the model for use after training
        requires_grad(ema, False)
    
    # 断点续训：加载 checkpoint
    resume_step = 0
    resume_epoch = 0
    if args.resume_from is not None:
        logger.info(f"Resuming training from checkpoint: {args.resume_from}")
        
        if args.use_lora:
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, args.resume_from)
            model.to(weight_dtype)
            transformer_lora_parameters = list(filter(lambda p: p.requires_grad, model.parameters()))
            opt = torch.optim.AdamW(transformer_lora_parameters, lr=args.lr, weight_decay=args.adam_weight_decay)
            logger.info("Loaded LoRA weights from checkpoint")
        else:
            state_dict = torch.load(os.path.join(args.resume_from, "model.pt"), map_location=device)
            if hasattr(model, "module"):
                model.module.load_state_dict(state_dict)
            else:
                model.load_state_dict(state_dict)
            logger.info("Loaded model weights from checkpoint")
        
        # 加载 training state
        with open(os.path.join(args.resume_from, 'resume_state.json'), 'r') as f:
            resume_state = json.load(f)
        resume_step = resume_state['train_steps']
        resume_epoch = resume_state['epoch']
        logger.info(f"Resuming from step {resume_step}, epoch {resume_epoch}")
        
        # 加载 loss history
        loss_history_path = f'{args.results_dir}/loss_history.json'
        if os.path.exists(loss_history_path):
            with open(loss_history_path, 'r') as f:
                loss_data = json.load(f)
            all_losses = loss_data['all_step_losses'][:resume_step]
            epoch_losses = loss_data['epoch_losses'][:resume_epoch]
            logger.info(f"Restored loss history: {len(all_losses)} step losses, {len(epoch_losses)} epoch losses")
    

    # Setup data:
    crop_func = crop_arr
    if not args.keep_raw_resolution:
        crop_func = center_crop_arr
    image_transform = transforms.Compose([
        transforms.Lambda(lambda pil_image: crop_func(pil_image, args.max_image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
    ])
    
    dataset = DatasetFromJson(json_file=args.json_file, 
    image_path=args.image_path,
    processer=processor,
    image_transform=image_transform,
    max_input_length_limit=args.max_input_length_limit,
    condition_dropout_prob=args.condition_dropout_prob,
    keep_raw_resolution=args.keep_raw_resolution
    )
    collate_fn = TrainDataCollator(pad_token_id=processor.text_tokenizer.eos_token_id, hidden_size=model.llm.config.hidden_size, keep_raw_resolution=args.keep_raw_resolution)

    loader = DataLoader(
        dataset,
        collate_fn=collate_fn,
        batch_size=args.batch_size_per_device,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    
    if accelerator.is_main_process:
        logger.info(f"Dataset contains {len(dataset):,}")

    num_update_steps_per_epoch = math.ceil(len(loader) / args.gradient_accumulation_steps)
    max_train_steps = args.epochs * num_update_steps_per_epoch
    
    # 根据 epoch 数计算各种步数参数
    args.ckpt_every = args.ckpt_epoch * num_update_steps_per_epoch
    logger.info(f"Checkpoint every {args.ckpt_every} steps (every {args.ckpt_epoch} epochs)")
    
    args.lr_warmup_steps = args.lr_warmup_epoch * num_update_steps_per_epoch
    logger.info(f"LR warmup for {args.lr_warmup_steps} steps ({args.lr_warmup_epoch} epochs)")
    
    # log_every 也转换为步数
    args.log_every = args.log_every * args.gradient_accumulation_steps * num_update_steps_per_epoch
    logger.info(f"Log every {args.log_every} steps (every {args.log_every // (args.gradient_accumulation_steps * num_update_steps_per_epoch)} epochs)")
    
    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=opt,
        num_warmup_steps=args.lr_warmup_steps,
        num_training_steps=max_train_steps,
    )

    # Prepare models for training:
    model.train()  # important! This enables embedding dropout for classifier-free guidance
    
    if ema is not None:
        update_ema(ema, model, decay=0)  # Ensure EMA is initialized with synced weights
        ema.eval()  # EMA model should always be in eval mode
    

    if ema is not None:
        model, ema = accelerator.prepare(model, ema)
    else:
        model = accelerator.prepare(model)

    opt, loader, lr_scheduler = accelerator.prepare(opt, loader, lr_scheduler)
    
    
    # Variables for monitoring/logging purposes:
    train_steps = resume_step
    log_steps = 0
    running_loss = 0
    start_time = time()
    
    start_epoch = resume_epoch
    print(f"Training for {args.epochs} epochs (starting from epoch {start_epoch})...")
    
    for epoch in range(start_epoch, args.epochs):
        print(f"\n{'='*60}")
        print(f"Beginning epoch {epoch}...")
        print(f"Dataset size: {len(loader.dataset)} samples, {len(loader)} batches")
        print(f"{'='*60}\n")
        
        epoch_loss = 0.0
        num_batches = 0
        
        for batch_idx, data in enumerate(loader):
            if batch_idx % 10 == 0:
                print(f"\n[TRAIN] Processing batch {batch_idx}/{len(loader)}")
            
            # 记录 batch 开始时间
            batch_start_time = time()
            step_times = {}
            step_start = time()
            
            with accelerator.accumulate(model):
                with torch.no_grad():
                    output_images = data['output_images']
                    
                    step_times['data_load'] = time() - step_start
                    step_start = time()
                    
                    input_pixel_values = data['input_pixel_values']
                    if isinstance(output_images, list):
                        output_images = vae_encode_list(vae, output_images, weight_dtype)
                        if input_pixel_values is not None:
                            input_pixel_values = vae_encode_list(vae, input_pixel_values, weight_dtype)
                    else:
                        output_images = vae_encode(vae, output_images, weight_dtype)
                        if input_pixel_values is not None:
                            input_pixel_values = vae_encode(vae, input_pixel_values, weight_dtype)
                    
                    step_times['vae_encode'] = time() - step_start
                    step_start = time()
                
                model_kwargs = dict(
                    input_ids=data['input_ids'], 
                    input_img_latents=input_pixel_values, 
                    input_image_sizes=data['input_image_sizes'], 
                    attention_mask=data['attention_mask'], 
                    position_ids=data['position_ids'], 
                    padding_latent=data['padding_images'], 
                    past_key_values=None, 
                    return_past_key_values=False
                )

                loss_dict = training_losses(model, output_images, model_kwargs=model_kwargs)
                
                step_times['loss_computation'] = time() - step_start
                step_start = time()
                
                loss = loss_dict["loss"].mean()

                # 记录 loss
                all_losses.append(loss.item())
                
                # 累加 epoch loss
                epoch_loss += loss.item()
                num_batches += 1

                running_loss += loss.item()
                accelerator.backward(loss)
                
                if accelerator.sync_gradients:
                    if args.max_grad_norm is not None:
                        accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    opt.step()
                    lr_scheduler.step()
                    opt.zero_grad()

                step_times['backward'] = time() - step_start

                # 每个 batch 完成后输出日志
                batch_time = time() - batch_start_time
                current_epoch_progress = batch_idx / len(loader) * 100
                
                # 输出详细的时间分解（每 10 个 batch 输出一次）
                if batch_idx % 10 == 0:
                    print(f"\n[Time Breakdown] Batch {batch_idx}:")
                    print(f"  - Data loading: {step_times.get('data_load', 0):.2f}s")
                    print(f"  - VAE encode: {step_times.get('vae_encode', 0):.2f}s")
                    print(f"  - Loss computation (model forward): {step_times.get('loss_computation', 0):.2f}s")
                    print(f"  - Backward pass: {step_times.get('backward', 0):.2f}s")
                    print(f"  - Total: {batch_time:.2f}s")
                
                print(f"\n[Epoch {epoch}] Batch {batch_idx}/{len(loader)} ({current_epoch_progress:.1f}%) - "
                      f"Time: {batch_time:.2f}s, Loss: {loss.item():.4f}")

                log_steps += 1
                train_steps += 1

                accelerator.log({"training_loss": loss.item()}, step=train_steps)
                if train_steps % args.gradient_accumulation_steps == 0:
                    if accelerator.sync_gradients and ema is not None: 
                        update_ema(ema, model)
                    
                if train_steps % args.log_every == 0 and train_steps > 0:
                    torch.cuda.synchronize()
                    end_time = time()
                    steps_per_sec = log_steps / args.gradient_accumulation_steps / (end_time - start_time)
                    # Reduce loss history over all processes:
                    avg_loss = torch.tensor(running_loss / log_steps, device=device)
                    if dist.is_available() and dist.is_initialized():
                        dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)                        
                    avg_loss = avg_loss.item() / accelerator.num_processes 
                        
                    if accelerator.is_main_process:
                        cur_lr = opt.param_groups[0]["lr"]
                        logger.info(f"(step={int(train_steps/args.gradient_accumulation_steps):07d}) Train Loss: {avg_loss:.4f}, Train Steps/Sec: {steps_per_sec:.2f}, Epoch: {train_steps/len(loader)}, LR: {cur_lr}")

                    # Reset monitoring variables:
                    running_loss = 0
                    log_steps = 0
                    start_time = time()


            if train_steps % args.ckpt_every == 0 and train_steps > 0:
                if accelerator.distributed_type == DistributedType.FSDP:
                    state_dict = accelerator.get_state_dict(model)
                    ema_state_dict = accelerator.get_state_dict(ema) if ema is not None else None
                else:
                    if not args.use_lora:
                        if hasattr(model, "module"):
                            state_dict = model.module.state_dict()
                        else:
                            state_dict = model.state_dict()
                        ema_state_dict = accelerator.get_state_dict(ema) if ema is not None else None

                if accelerator.is_main_process:
                    if args.use_lora:
                        checkpoint_path = f"{checkpoint_dir}/{int(train_steps/args.gradient_accumulation_steps):07d}/"
                        os.makedirs(checkpoint_path, exist_ok=True)
                        
                        if hasattr(model, "module"):
                            model.module.save_pretrained(checkpoint_path)
                        else:
                            model.save_pretrained(checkpoint_path)
                    else:
                        checkpoint_path = f"{checkpoint_dir}/{int(train_steps/args.gradient_accumulation_steps):07d}/"
                        os.makedirs(checkpoint_path, exist_ok=True)
                        torch.save(state_dict, os.path.join(checkpoint_path, "model.pt"))
                        processor.text_tokenizer.save_pretrained(checkpoint_path)
                        model.llm.config.save_pretrained(checkpoint_path)
                        if ema_state_dict is not None:
                            checkpoint_path = f"{checkpoint_dir}/{int(train_steps/args.gradient_accumulation_steps):07d}_ema"
                            os.makedirs(checkpoint_path, exist_ok=True)
                            torch.save(ema_state_dict, os.path.join(checkpoint_path, "model.pt"))
                            processor.text_tokenizer.save_pretrained(checkpoint_path)
                            model.llm.config.save_pretrained(checkpoint_path)
                    # 保存恢复状态
                    resume_state = {'train_steps': train_steps, 'epoch': epoch}
                    with open(os.path.join(checkpoint_path, 'resume_state.json'), 'w') as f:
                        json.dump(resume_state, f)
                    logger.info(f"Saved checkpoint to {checkpoint_path}")
                    
            if dist.is_available() and dist.is_initialized():
                dist.barrier()
        
        # Epoch 结束：保存 epoch loss 并绘图
        if accelerator.is_main_process:
            avg_epoch_loss = epoch_loss / max(num_batches, 1)
            epoch_losses.append(avg_epoch_loss)
            logger.info(f"Epoch {epoch} completed - Average Loss: {avg_epoch_loss:.4f}")
            
            # 绘制 loss-epoch 图
            if args.plot_loss:
                plt.figure(figsize=(10, 6))
                plt.plot(range(len(epoch_losses)), epoch_losses, 'b-', linewidth=2, marker='o', markersize=4)
                plt.xlabel('Epoch', fontsize=12)
                plt.ylabel('Average Loss', fontsize=12)
                plt.title('Training Loss Curve', fontsize=14)
                plt.grid(True, alpha=0.3)
                plt.savefig(f'{args.results_dir}/loss_epoch_curve.png', dpi=150, bbox_inches='tight')
                plt.close()
                
                # 保存 loss 数据到 JSON
                with open(f'{args.results_dir}/loss_history.json', 'w') as f:
                    json.dump({
                        'epoch_losses': epoch_losses,
                        'all_step_losses': all_losses
                    }, f, indent=2)
                
                logger.info(f"Loss curve saved to {args.results_dir}/loss_epoch_curve.png")
    accelerator.end_training()
    model.eval()  
    
    if accelerator.is_main_process:
        logger.info("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, default="/root/autodl-tmp/Fusion1/OmniGen/Ship/results/LoRA/reference_to_target", help="results directory")
    parser.add_argument("--model_name_or_path", type=str, default="/root/autodl-tmp/Fusion1/OmniGen/models/OmniGen-v1-weights", help="model name or path (HuggingFace model ID or local path)")
    parser.add_argument("--json_file", type=str, default="/root/autodl-tmp/Fusion1/OmniGen/Ship/data/annotations/train.json", help="training JSON file")
    parser.add_argument("--image_path", type=str, default="/root/autodl-tmp/Fusion1/OmniGen/Ship/data/images/FGSC", help="image path")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch_size_per_device", type=int, default=4)
    parser.add_argument("--vae_path", type=str, default=None) 
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--log_every", type=int, default=1, help="Log every N epochs (will be converted to steps)")
    parser.add_argument("--ckpt_epoch", type=int, default=25, help="Save checkpoint every N epochs")
    parser.add_argument("--lr_warmup_epoch", type=int, default=5, help="LR warmup for N epochs")
    parser.add_argument("--resume_from", type=str, default=None, help="Resume from checkpoint path")
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--max_input_length_limit", type=int, default=20000, help="max input length (default=20000 for multi-image input)")
    parser.add_argument("--condition_dropout_prob", type=float, default=0.1)
    parser.add_argument("--adam_weight_decay", type=float, default=0.0)
    parser.add_argument(
            "--keep_raw_resolution",
            action="store_true",
            help="multiple_resolutions",
    )
    parser.add_argument("--max_image_size", type=int, default=512, help="max image size")

    parser.add_argument(
            "--use_lora",
            action="store_true",
            default=True,
            help="use LoRA for parameter-efficient fine-tuning",
    )
    parser.add_argument("--lora_rank", type=int, default=128)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2)
    parser.add_argument("--mixed_precision", type=str, default="bf16", choices=["no", "fp16", "bf16"])
    parser.add_argument("--report_to", type=str, default="tensorboard")
    parser.add_argument("--lr_scheduler", type=str, default="linear")
    parser.add_argument(
            "--use_ema",
            action="store_true",
            help="use EMA model",
    )
    parser.add_argument(
            "--plot_loss",
            action="store_true",
            default=True,
            help="plot loss curve",
    )
    
    args = parser.parse_args()
    main(args)
