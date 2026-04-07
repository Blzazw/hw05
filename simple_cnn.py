"""
极简 CNN 手写数字识别
参考课程公众号文章《计算机视觉》第10章
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# 设置设备（有 GPU 就用 GPU，没有就用 CPU）
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'使用设备: {device}')

# ========== 1. 超参数设置 ==========
batch_size = 64          # 每次训练用64张图片
learning_rate = 0.001    # 学习率
num_epochs = 10          # 训练10轮

# ========== 2. 数据预处理 ==========
# MNIST 图片是 28x28，需要转换成张量并归一化到 [0,1] 区间
transform = transforms.Compose([
    transforms.ToTensor(),                    # 转换为张量
    transforms.Normalize((0.1307,), (0.3081,)) # 归一化（MNIST的均值和标准差）
])

# 下载训练集（第一次运行会自动下载）
train_dataset = torchvision.datasets.MNIST(
    root='./data',      # 保存路径
    train=True,         # 训练集
    transform=transform,
    download=True
)

# 下载测试集
test_dataset = torchvision.datasets.MNIST(
    root='./data',
    train=False,        # 测试集
    transform=transform,
    download=True
)

# 创建数据加载器（批量读取数据）
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

print(f'训练集大小: {len(train_dataset)} 张图片')
print(f'测试集大小: {len(test_dataset)} 张图片')

# ========== 3. 定义极简 CNN 模型 ==========
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        # 卷积层1：输入1通道（灰度图），输出32通道，卷积核3x3
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        # 卷积层2：输入32通道，输出64通道，卷积核3x3
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        # 池化层：2x2 最大池化
        self.pool = nn.MaxPool2d(2, 2)
        # 全连接层1：输入 64*7*7 = 3136，输出 128
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        # 全连接层2：输入 128，输出 10（10个数字类别）
        self.fc2 = nn.Linear(128, 10)
        # Dropout：防止过拟合
        self.dropout = nn.Dropout(0.25)
        # 激活函数
        self.relu = nn.ReLU()

    def forward(self, x):
        # 卷积 -> 激活 -> 池化
        x = self.pool(self.relu(self.conv1(x)))   # 28x28 -> 14x14
        x = self.pool(self.relu(self.conv2(x)))   # 14x14 -> 7x7
        # 展平：将 64*7*7 的 tensor 变成一维向量
        x = x.view(-1, 64 * 7 * 7)
        # 全连接层
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# 创建模型实例
model = SimpleCNN().to(device)

# 计算模型参数量
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f'模型总参数量: {total_params:,}')
print(f'可训练参数量: {trainable_params:,}')

# ========== 4. 定义损失函数和优化器 ==========
criterion = nn.CrossEntropyLoss()  # 交叉熵损失（适合分类问题）
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# ========== 5. 训练模型 ==========
print('\n开始训练...')
train_losses = []

for epoch in range(num_epochs):
    model.train()  # 切换到训练模式
    running_loss = 0.0
    
    for i, (images, labels) in enumerate(train_loader):
        # 将数据移到设备（GPU/CPU）
        images = images.to(device)
        labels = labels.to(device)
        
        # 前向传播：计算预测结果
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # 反向传播：更新权重
        optimizer.zero_grad()   # 清空梯度
        loss.backward()         # 计算梯度
        optimizer.step()        # 更新参数
        
        running_loss += loss.item()
        
        # 每100个batch打印一次
        if (i+1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}')
    
    avg_loss = running_loss / len(train_loader)
    train_losses.append(avg_loss)
    print(f'Epoch [{epoch+1}/{num_epochs}] 平均损失: {avg_loss:.4f}')

# ========== 6. 测试模型 ==========
print('\n开始测试...')
model.eval()  # 切换到评估模式
correct = 0
total = 0

with torch.no_grad():  # 不计算梯度（节省内存和计算）
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)  # 取概率最大的类别
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f'测试集准确率: {accuracy:.2f}%')

# ========== 7. 绘制训练损失曲线 ==========
plt.figure(figsize=(8, 5))
plt.plot(range(1, num_epochs+1), train_losses, marker='o')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss Curve')
plt.grid(True)
plt.savefig('training_loss.png')
plt.show()
print('训练损失曲线已保存为 training_loss.png')