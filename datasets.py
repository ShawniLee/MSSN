import pandas as pd
from sklearn import preprocessing
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler


class MultiSignalDataset(Dataset):
    def __init__(self, sub_dir, signals_list, base_dir="/data/ECT_detect/dataset/", get_defect_lens=False):
        """
        初始化多信号数据集。
        Args:
            sub_dir (str): 数据的子目录名。
            signals_list (list): 需要加载的信号列表。
            base_dir (str): 数据的基本存储目录。
            get_defect_lens (bool): 是否计算缺陷长度。
        """
        self.base_dir = base_dir
        self.sub_dir = sub_dir
        self.data, self.labels = self.load_signals(signals_list)
        if get_defect_lens:
            self.instance_lens = [self.get_defect_lens(label.argmax(axis=0)) for label in self.labels]

    def load_signals(self, signals_list):
        """
        加载并合并信号数据。
        """
        data_list, label_list = [], []
        for signal in signals_list:
            data_path = os.path.join(self.base_dir, self.sub_dir, f"{signal}_x.npy")
            label_path = os.path.join(self.base_dir, self.sub_dir, f"{signal}_y.npy")
            if os.path.exists(data_path) and os.path.exists(label_path):
                data_list.append(np.load(data_path, allow_pickle=True))
                label_list.append(np.load(label_path, allow_pickle=True))

        return np.concatenate(data_list, axis=0), np.concatenate(label_list, axis=0)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        x = self.data[index]
        y = self.labels[index]
        if hasattr(self, 'instance_lens'):
            return x, y, self.instance_lens[index]
        return x, y

    def get_defect_lens(self, ts_label):
        """
        计算时间序列中缺陷的长度。
        """
        segments = []
        start_index = None
        for i, label in enumerate(ts_label):
            if label == 1 and start_index is None:
                start_index = i
            elif label == 0 and start_index is not None:
                segments.append((start_index, i))
                start_index = None
        if start_index is not None:
            segments.append((start_index, len(ts_label)))

        return [end - start for start, end in segments]


def create_data_loaders(dataset, batch_size, use_multi_gpus=False, shuffle=False):
    if use_multi_gpus:
        sampler = DistributedSampler(dataset)
        return DataLoader(dataset, batch_size=batch_size, sampler=sampler)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


class MackeyGlassDataset(Dataset):
    def __init__(self, csv_file, window_size=1000, step_size=100, tau_values=None, normalize=True):
        if tau_values is None:
            tau_values = [10, 17, 30, 100]
        self.dataframe = pd.read_csv(csv_file)
        self.window_size = window_size
        self.step_size = step_size
        self.tau_values = tau_values

        # 归一化处理
        if normalize:
            scaler = preprocessing.MinMaxScaler()
            self.dataframe[['x']] = scaler.fit_transform(self.dataframe[['x']])

        # Create a mapping for one-hot encoding of tau values
        self.tau_map = {tau: idx for idx, tau in enumerate(tau_values)}

        # Preprocessing to create windows
        self.windows = []
        for start in range(0, len(self.dataframe) - window_size, step_size):
            end = start + window_size
            self.windows.append((start, end))

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        start, end = self.windows[idx]
        x = self.dataframe.iloc[start:end]['x'].values
        tau = self.dataframe.iloc[start:end]['tau'].values

        # One-hot encode the tau values
        tau_one_hot = np.zeros((len(self.tau_values), len(tau)))
        for i, t in enumerate(tau):
            tau_idx = self.tau_map[t]
            tau_one_hot[tau_idx, i] = 1

        # Reshape x to have an additional dimension
        x_reshaped = x.reshape(1, -1)

        # Return as PyTorch tensors
        return torch.from_numpy(x_reshaped).float(), torch.from_numpy(tau_one_hot).float()
