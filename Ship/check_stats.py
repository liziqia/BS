import json

with open('/root/autodl-tmp/Fusion1/OmniGen/Ship/data/annotations/train.json', 'r') as f:
    data = json.load(f)

print(f"总条目数：{len(data)}")

# 统计任务类型
rotation = 0
mask = 0
color = 0
i2i = 0

for item in data:
    instr = item['instruction']
    input_img = item['input_images'][0]
    
    if 'rotated counterclockwise' in instr:
        rotation += 1
    elif 'masked' in instr:
        mask += 1
    elif 'color' in instr or '_color.jpg' in input_img:
        color += 1
    else:
        i2i += 1

print(f"\n任务分布：")
print(f"  旋转任务：{rotation}")
print(f"  掩码任务：{mask}")
print(f"  颜色抖动：{color}")
print(f"  i2i 任务：{i2i}")

# 按类别统计
class_count = {}
for item in data:
    class_id = item['output_image'].split('/')[1]
    if class_id not in class_count:
        class_count[class_id] = 0
    class_count[class_id] += 1

print(f"\n类别分布（所有类别）：")
for cid in sorted(class_count.keys()):
    print(f"  类别 {cid}: {class_count[cid]} 个样本")

print(f"\n总类别数：{len(class_count)}")
