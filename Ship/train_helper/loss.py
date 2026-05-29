import torch
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('[%(levelname)s] %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)


def sample_x0(x1):
    """Sampling x0 & t based on shape of x1 (if needed)
    Args:
      x1 - data point; [batch, *dim]
    """
    if isinstance(x1, (list, tuple)):
        x0 = [torch.randn_like(img_start) for img_start in x1]
    else:
        x0 = torch.randn_like(x1)

    return x0


def sample_timestep(x1):
    u = torch.normal(mean=0.0, std=1.0, size=(len(x1),))
    t = 1 / (1 + torch.exp(-u))
    t = t.to(x1[0])
    return t


def training_losses(model, x1, model_kwargs=None, snr_type='uniform', patch_weight=None):
    """Loss for training the score model with standard flow matching (linear path)
    Args:
    - model: backbone model
    - x1: target datapoint (Vis image)
    - model_kwargs: additional arguments for torch model
    
    标准流匹配（线性路径）: x(t) = t*x1 + (1-t)*x0
    目标速度：ut = x1 - x0
    """
    if model_kwargs == None:
        model_kwargs = {}

    B = len(x1)
    x0 = sample_x0(x1)
    t = sample_timestep(x1)
    
    if isinstance(x1, (list, tuple)):
        dims = [1] * (len(x1[0].size()) - 1)
        t_ = [t[i].view(*dims) for i in range(B)]
        xt = [t_[i] * x1[i] + (1 - t_[i]) * x0[i] for i in range(B)]
        ut = [x1[i] - x0[i] for i in range(B)]
    else:
        dims = [1] * (len(x1.size()) - 1)
        t_ = t.view(t.size(0), *dims)
        xt = t_ * x1 + (1 - t_) * x0
        ut = x1 - x0

    import time
    start_time = time.time()
    model_output = model(xt, t, **model_kwargs)
    elapsed = time.time() - start_time

    terms = {}

    if isinstance(x1, (list, tuple)):
        assert len(model_output) == len(ut) == len(x1)
        if patch_weight is not None:
            terms["loss"] = torch.stack(
            [((ut[i] - model_output[i]) ** 2 * patch_weight[i]).mean() for i in range(B)],
            dim=0,
            )
        else:
            terms["loss"] = torch.stack(
            [((ut[i] - model_output[i]) ** 2).mean() for i in range(B)],
            dim=0,
            )
    else:
        if patch_weight is not None:
            loss = (model_output - ut) ** 2
            loss = loss * patch_weight
            terms["loss"] = mean_flat(loss)
        else:
            terms["loss"] = mean_flat(((model_output - ut) ** 2))

    return terms


def mean_flat(x):
    """
    Take the mean over all non-batch dimensions.
    """
    return torch.mean(x, dim=list(range(1, len(x.size()))))
