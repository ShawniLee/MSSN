import einops
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
from torch.utils.data import Dataset


def get_memory_usage():
    import os
    import psutil
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 ** 2


class TimeSeriesDataset(Dataset):
    def __init__(self, data, labels, get_defect_lens=False):
        self.data = data
        self.labels = labels
        if get_defect_lens:
            self.instance_lens = torch.tensor(
                [self.get_defect_lens(ts_label.argmax(axis=0)) for ts_label in self.labels])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        x = self.data[index]
        y = self.labels[index]
        return x, y

    def get_defect_lens(self, ts_label):
        segments = []
        start_index = None
        for i in range(ts_label.shape[0]):
            if ts_label[i] == 1 and start_index is None:
                start_index = i
            elif ts_label[i] == 0 and start_index is not None:
                end_index = i
                segments.append((start_index, end_index))
                start_index = None
        if start_index is not None:
            end_index = ts_label.shape[0]
            segments.append((start_index, end_index))
        if len(segments) == 0:
            return 0
        elif len(segments) == 1:
            return segments[0][1] - segments[0][0]
        else:
            print("Warning: more than one defect in a time series")
            return [segment[1] - segment[0] for segment in segments]


def normalize_data(train_x, normalization_type):
    if normalization_type == "GlobalNormalization":  # 标准化
        standardScaler = StandardScaler()
        std_train_x = einops.rearrange(train_x, 'b f l -> (b l) f')
        standardScaler.fit(std_train_x)
        std_train_x = standardScaler.transform(std_train_x)
        train_x = einops.rearrange(std_train_x, '(b l) f -> b f l', b=train_x.shape[0])
    elif normalization_type == "TimeStampNormalization":
        standardScaler_array = []
        for i in range(train_x.shape[-1]):
            standardScaler = StandardScaler()
            standardScaler.fit(train_x[:, :, i])
            standardScaler_array.append(standardScaler)
    elif normalization_type == "None":
        pass
    else:
        raise Exception("Invalid normalization type!")

    return train_x


class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-5):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, predicts, targets):
        intersection = torch.sum(targets * predicts, dim=2)
        sum_targets = torch.sum(targets, dim=2)
        sum_predicts = torch.sum(predicts, dim=2)
        dice_loss = 1 - (2 * intersection + self.smooth) / (sum_targets + sum_predicts + self.smooth)
        return torch.mean(dice_loss)


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, reduction="mean"):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, input, target):
        BCE_loss = nn.functional.binary_cross_entropy_with_logits(input, target, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss

        if self.reduction == "mean":
            return F_loss.mean()
        elif self.reduction == "sum":
            return F_loss.sum()
        else:
            return F_loss


class ComboLoss(nn.Module):
    def __init__(self, alpha=0.7):
        super(ComboLoss, self).__init__()
        self.alpha = alpha
        self.dice_loss = DiceLoss()
        self.cross_entropy_loss = nn.CrossEntropyLoss()

    def forward(self, predicts, targets):
        dice_loss = self.dice_loss(predicts, targets)
        cross_entropy_loss = self.cross_entropy_loss(predicts, targets)
        return self.alpha * dice_loss + (1 - self.alpha) * cross_entropy_loss


def create_data_loaders(x, y, batch_size, use_multi_gpus=False, use_weighted_sampler=False, shuffle=False):
    if use_multi_gpus:
        dataset = TimeSeriesDataset(x, y)
        return DistributedSample(
            dataset, batch_size
        )
    elif use_weighted_sampler:
        dataset = TimeSeriesDataset(x, y, get_defect_lens=True)
        return WeightedSample(
            dataset, batch_size
        )
    else:
        dataset = TimeSeriesDataset(x, y)
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def DistributedSample(dataset, batch_size):
    sampler = torch.utils.data.distributed.DistributedSampler(dataset)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, sampler=sampler)
    return loader, sampler


def WeightedSample(dataset, batch_size, world_size=1):
    def get_weight():
        # 获取所有唯一的标签
        unique_labels = torch.unique(dataset.instance_lens, sorted=True)

        # 创建一个映射，将原始的标签映射到连续的索引上
        label_to_index = {label.item(): index for index, label in enumerate(unique_labels)}

        # 计算每个类别的样本数量
        class_sample_count = torch.tensor(
            [(dataset.instance_lens == label).sum() for label in unique_labels]
        )

        # 计算每个类别的权重
        weight = 1. / class_sample_count.float()

        # 计算每个样本的权重
        return torch.tensor([weight[label_to_index[label.item()]] for label in dataset.instance_lens])

    sampler = torch.utils.data.WeightedRandomSampler(get_weight(), len(dataset) // world_size, replacement=True)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, sampler=sampler)
    return loader, sampler


def model_inference(model, batch_x, batch_y, criterion):
    """
    Performs model inference on a single batch, calculates the loss, and returns predictions.
    """
    batch_x = preprocess_batch(batch_x, model)
    outputs = model(batch_x)
    loss = calculate_loss(outputs, model, batch_y, criterion)
    predicted = get_predictions(outputs, model)

    return loss, predicted


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0
    running_corrects = 0

    for batch_x, batch_y in loader:
        batch_x = batch_x.float().to(device)
        batch_y = batch_y.float().to(device)
        optimizer.zero_grad()
        loss, predicted = model_inference(model, batch_x, batch_y, criterion)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        running_corrects += (predicted == torch.max(batch_y, 1)[1]).sum().item()
        # running_corrects += (torch.round(outputs) == torch.max(batch_y, 1)[1]).sum().item()

    epoch_loss = running_loss / len(loader.dataset)
    epoch_accuracy = running_corrects / len(loader.dataset)

    return epoch_loss, epoch_accuracy


def eval_epoch(model, loader, criterion, device):
    model.eval()
    running_loss = 0
    running_corrects = 0

    with torch.no_grad():
        for batch_x, batch_y in loader:
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)
            loss, predicted = model_inference(model, batch_x, batch_y, criterion)

            running_loss += loss.item()
            running_corrects += (predicted == torch.max(batch_y, 1)[1]).sum().item()
            # running_corrects += (torch.round(outputs) == torch.max(batch_y, 1)[1]).sum().item()

    epoch_loss = running_loss / len(loader.dataset)
    epoch_accuracy = running_corrects / len(loader.dataset)

    return epoch_loss, epoch_accuracy


def preprocess_batch(batch_x, model):
    """
    Preprocesses the batch data based on the model requirements.
    """
    if "2D" in model.name:
        batch_x = batch_x.unsqueeze(1)
    return batch_x


def calculate_loss(outputs, model, batch_y, criterion):
    """
    Calculates loss based on the model type and outputs.
    """
    return criterion(outputs, batch_y)


def get_predictions(outputs, model):
    """
    Extracts predictions from the model outputs based on the model type.
    """
    if "MS" in model.name:
        _, predicted = torch.max(outputs[-1], 1)
    elif model.name == "Prectime":
        _, predicted = torch.max(outputs[0], 1)
    elif "Sleep" in model.name:
        _, predicted = torch.max(outputs[0], 1)
    else:
        _, predicted = torch.max(outputs, 1)
    return predicted


def test_epoch(test_loader, model, device):
    """
    Tests the model for an entire epoch and collects all predictions.
    """
    model.eval()
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.float().to(device)
            # batch_y = batch_y.float().to(device)
            batch_x = preprocess_batch(batch_x, model)
            outputs = model(batch_x)
            predicted = get_predictions(outputs, model)
            predicted = predicted.cpu().numpy()

            all_predictions.extend(predicted)
            all_labels.extend(batch_y.cpu().numpy())

    return all_predictions, all_labels
