import torch
import torch.nn as nn
import torch.nn.functional as F

class RSSICNN(nn.Module):
    def __init__(self, input_dim=620, dropout_rate=0.2):
        super().__init__()

        # Conv Block 1: 提取浅层局部特征
        # kernel=5, padding=2 可以保持卷积后序列长度不变
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(64)
        self.pool1 = nn.MaxPool1d(kernel_size=2)

        # Conv Block 2: 提取中层特征
        # kernel=3, padding=1 保持长度不变
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(128)
        self.pool2 = nn.MaxPool1d(kernel_size=2)

        # Conv Block 3: 提取高层特征
        self.conv3 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(256) # 补充了第三层的 BatchNorm 加快收敛

        # 真正实现多尺度特征融合：全局最大池化 (抓取最强信号) + 全局平均池化 (抓取整体信号分布)
        self.adaptive_max_pool = nn.AdaptiveMaxPool1d(1)
        self.adaptive_avg_pool = nn.AdaptiveAvgPool1d(1)

        # 启用 Dropout 防止过拟合
        self.dropout = nn.Dropout(p=dropout_rate)
        
        # 因为我们拼接了 Max (256维) 和 Avg (256维)，所以输入全连接层的维度是 512
        self.fc1 = nn.Linear(512, 64)
        self.fc2 = nn.Linear(64, 2)

    def forward(self, rssi):
        """
        rssi: (B, input_dim) 
        返回: (B, 2) 预测的 X, Y 坐标
        """
        # 增加通道维度: (B, 620) -> (B, 1, 620)
        x = rssi.unsqueeze(1) 

        # Block 1 -> (B, 64, 310)
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))  
        
        # Block 2 -> (B, 128, 155)
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))  
        
        # Block 3 -> (B, 256, 155)
        x = F.relu(self.bn3(self.conv3(x)))              

        # 特征融合：将序列维度压缩，提取核心信息
        x_max = self.adaptive_max_pool(x).squeeze(2)  # (B, 256)
        x_avg = self.adaptive_avg_pool(x).squeeze(2)  # (B, 256)
        
        # 将最大值和平均值拼接，维度变为 (B, 512)
        x_fused = torch.cat([x_max, x_avg], dim=1)    

        # 分类/回归头
        x_fused = self.dropout(x_fused)
        x_fused = F.relu(self.fc1(x_fused))
        out = self.fc2(x_fused)  # (B, 2)
        
        return out