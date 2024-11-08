import os
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator

# 替换为你的 .tfevents 文件所在的日志目录
logdir = "log/run1"
event_file = None

# 找到最新的 .tfevents 文件
for file in os.listdir(logdir):
    if file.startswith("events"):
        event_file = os.path.join(logdir, file)
        break

# 确保找到文件
if event_file is None:
    raise FileNotFoundError("未找到 .tfevents 文件，请检查路径！")

# 使用 event_accumulator 加载事件文件
ea = event_accumulator.EventAccumulator(event_file)
ea.Reload()  # 加载所有事件

# 提取 train_loss 和 val_loss
train_loss = ea.Scalars("train_loss")
val_loss = ea.Scalars("val_loss")

# 提取步数和损失值
train_steps = [point.step for point in train_loss]
train_values = [point.value for point in train_loss]
val_steps = [point.step for point in val_loss]
val_values = [point.value for point in val_loss]

# 绘制曲线
plt.figure(figsize=(10, 5))
plt.plot(train_steps, train_values, label="Train Loss", color="blue")
plt.plot(val_steps, val_values, label="Validation Loss", color="orange")
plt.xlabel("Steps")
plt.ylabel("Loss")
plt.legend()
plt.title("Training and Validation Loss")
plt.show()
plt.savefig('train_val_loss.png')
