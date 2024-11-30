import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix
import torch
import torch.nn as nn
import torch.optim as optim
import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleFullyConnectedNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleFullyConnectedNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        x = self.softmax(x)
        return x
    def feature_list(self, x):
        out_list = []

        # 第一层
        out = self.relu(self.fc1(x))
        out_list.append(out)  # 将第一层的特征添加到特征列表

        # 第二层
        out = self.fc2(out)
        out_list.append(out)  # 将第二层的特征添加到特征列表

        # 最后的 softmax 层
        out = self.softmax(out)
        out_list.append(out)  # 将 softmax 层的特征添加到特征列表

        return out, out_list  # 返回最终输出和特征列表
    def intermediate_forward(self, x, layer_index=None):
        # 提取中间层的输出
        out_list = []

        # 第一个全连接层的输出
        out = self.relu(self.fc1(x))
        out_list.append(out)  # 保存第一个全连接层的输出

        # 第二个全连接层的输出
        out = self.fc2(out)
        out_list.append(out)  # 保存第二个全连接层的输出

        # 根据 `layer_index` 提取指定层的输出
        if layer_index is not None:
            return out_list[layer_index]
        else:
            return out_list  # 如果没有指定层，返回所有层的输出
# 假设每个流的特征数是77（您可以根据您的数据进行调整）
input_size = 76  # 假设每个样本有77个特征
hidden_size = 128  # 隐藏层大小
output_size = 5  # 二分类问题：in-distribution vs. out-of-distribution
device = torch.device("cpu")  # 使用CPU