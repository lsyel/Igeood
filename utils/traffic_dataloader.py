import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

class NetworkTrafficDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        """
        Args:
            csv_file (string): CSV 文件路径，包含网络流量数据
            transform (callable, optional): 对每个样本应用的转换
        """
        # 读取CSV文件
        self.data = pd.read_csv(csv_file)
        
        # 特征列，选择你需要的特征
        feature_columns = [
            'Flow Duration', 'Total Fwd Packet', 'Total Bwd packets', 'Total Length of Fwd Packet',
            'Total Length of Bwd Packet', 'Fwd Packet Length Max', 'Fwd Packet Length Min', 
            'Fwd Packet Length Mean', 'Fwd Packet Length Std', 'Bwd Packet Length Max', 
            'Bwd Packet Length Min', 'Bwd Packet Length Mean', 'Bwd Packet Length Std', 
            'Flow Bytes/s', 'Flow Packets/s', 'Flow IAT Mean', 'Flow IAT Std', 'Flow IAT Max', 
            'Flow IAT Min', 'Fwd IAT Total', 'Fwd IAT Mean', 'Fwd IAT Std', 'Fwd IAT Max', 
            'Fwd IAT Min', 'Bwd IAT Total', 'Bwd IAT Mean', 'Bwd IAT Std', 'Bwd IAT Max', 
            'Bwd IAT Min', 'Fwd PSH Flags', 'Bwd PSH Flags', 'Fwd URG Flags', 'Bwd URG Flags', 
            'Fwd Header Length', 'Bwd Header Length', 'Fwd Packets/s', 'Bwd Packets/s', 'Packet Length Min', 
            'Packet Length Max', 'Packet Length Mean', 'Packet Length Std', 'Packet Length Variance', 
            'FIN Flag Count', 'SYN Flag Count', 'RST Flag Count', 'PSH Flag Count', 'ACK Flag Count', 
            'URG Flag Count', 'CWR Flag Count', 'ECE Flag Count', 'Down/Up Ratio', 'Average Packet Size', 
            'Fwd Segment Size Avg', 'Bwd Segment Size Avg', 'Fwd Bytes/Bulk Avg', 'Fwd Packet/Bulk Avg', 
            'Fwd Bulk Rate Avg', 'Bwd Bytes/Bulk Avg', 'Bwd Packet/Bulk Avg', 'Bwd Bulk Rate Avg', 
            'Subflow Fwd Packets', 'Subflow Fwd Bytes', 'Subflow Bwd Packets', 'Subflow Bwd Bytes', 
            'FWD Init Win Bytes', 'Bwd Init Win Bytes', 'Fwd Act Data Pkts', 'Fwd Seg Size Min', 
            'Active Mean', 'Active Std', 'Active Max', 'Active Min', 'Idle Mean', 'Idle Std', 'Idle Max', 
            'Idle Min'
        ]
        
        # 提取特征数据
        self.X = self.data[feature_columns].values
        self.y = self.data['Label'].values  # 目标标签
        
        # 数据标准化
        self.scaler = StandardScaler()
        self.X = self.scaler.fit_transform(self.X)

        # 如果有转换操作，应用它
        self.transform = transform

    def __len__(self):
        """返回数据集的大小"""
        return len(self.X)

    def __getitem__(self, idx):
        """返回索引对应的数据和标签"""
        sample = torch.tensor(self.X[idx], dtype=torch.float32)
        label = torch.tensor(self.y[idx], dtype=torch.long)
        
        # if self.transform:
        #     sample = self.transform(sample)
        
        return sample, label
