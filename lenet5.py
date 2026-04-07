"""
LeNet-5 手写数字识别
论文：Gradient-Based Learning Applied to Document Recognition (1998)
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import time

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'使用设备: {device}')

# ========== 1. 超参数设置 ==========
batch_size = 64
learning_rate = 0.001
num_epochs = 10

# ========== 2. 数据预处理 ==========
# LeNet-5 要求输入为 32x32，MNIST 是 28x28，所以需要 Resize
transform = transforms.Compose([
    transforms.Resize((32, 32)),              # 调整到 32x32
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = torchvision.datasets.MNIST(
    root='./data',
    train=True,
    transform=transform,
    download=True
)

test_dataset = torchvision.datasets.MNIST(
    root='./data',
    train=False,
    transform=transform,
    download=True
)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

print(f'训练集大小: {len(train_dataset)} 张图片')
print(f'测试集大小: {len(test_dataset)} 张图片')

# ========== 3. 定义 LeNet-5 模型 ==========
class LeNet5(nn.Module):
    """
    经典 LeNet-5 架构
    输入: 32x32x1 灰度图像
    输出: 10 个类别的 logits
    """
    def __init__(self, num_classes=10):
        super(LeNet5, self).__init__()
        
        # C1: 卷积层 (1 -> 6, 5x5 卷积核)
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=0)
        # S2: 池化层 (2x2)
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)
        
        # C3: 卷积层 (6 -> 16, 5x5 卷积核)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0)
        # S4: 池化层 (2x2)
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
        
        # C5: 卷积层 (16 -> 120, 5x5 卷积核)
        self.conv3 = nn.Conv2d(16, 120, kernel_size=5, stride=1, padding=0)
        
        # F6: 全连接层 (120 -> 84)
        self.fc1 = nn.Linear(120, 84)
        # Output: 全连接层 (84 -> 10)
        self.fc2 = nn.Linear(84, num_classes)
        
        # 激活函数
        self.tanh = nn.Tanh()  # 原论文使用 tanh
        
    def forward(self, x):
        # C1 -> S2
        x = self.pool1(self.tanh(self.conv1(x)))   # 32 -> 28 -> 14
        # C3 -> S4
        x = self.pool2(self.tanh(self.conv2(x)))   # 14 -> 10 -> 5
        # C5
        x = self.tanh(self.conv3(x))               # 5 -> 1
        # 展平
        x = x.view(x.size(0), -1)                  # 120
        # F6
        x = self.tanh(self.fc1(x))                 # 84
        # Output
        x = self.fc2(x)                            # 10
        return x

# 创建模型
model = LeNet5().to(device)

# 计算参数量
total_params = sum(p.numel() for p in model.parameters())
print(f'LeNet-5 总参数量: {total_params:,}')

# ========== 4. 损失函数和优化器 ==========
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# ========== 5. 训练模型 ==========
print('\n开始训练 LeNet-5...')
train_losses = []
start_time = time.time()

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)
        
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
        if (i+1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}')
    
    avg_loss = running_loss / len(train_loader)
    train_losses.append(avg_loss)
    print(f'Epoch [{epoch+1}/{num_epochs}] 平均损失: {avg_loss:.4f}')

training_time = time.time() - start_time
print(f'\n训练完成，耗时: {training_time:.2f} 秒')

# ========== 6. 测试模型 ==========
print('\n开始测试...')
model.eval()
correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f'LeNet-5 测试集准确率: {accuracy:.2f}%')

# ========== 7. 绘制训练损失曲线 ==========
plt.figure(figsize=(8, 5))
plt.plot(range(1, num_epochs+1), train_losses, marker='o', color='blue')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('LeNet-5 Training Loss Curve')
plt.grid(True)
plt.savefig('lenet5_loss.png')
plt.show()
print('训练损失曲线已保存为 lenet5_loss.png')