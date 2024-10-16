import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn import preprocessing
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from datasets import MackeyGlassDataset
from loss import AsymmetricUnifiedFocalLoss, CeTmseLoss
from models.MSSN2D import MSSN2D
from models.FCN_inception_pytorch import Inception_Seg
from models.MS_TCN2 import MS_TCN2
from models.Prectime import Prectime
from models.TimesNet import TimesNet, Configs
from models.Transformer_lognorm_association_pytorch3 import Transformer_lognorm_association
from models.UTMAS import UTMAS
from models.Utimev2_pytorch import UTime
from utils import train_epoch, eval_epoch, DiceLoss, FocalLoss, ComboLoss


def transform_labels_to_one_hot(label):
    enc = preprocessing.OneHotEncoder(categories='auto')
    enc.fit(label.reshape(-1, 1))
    # 使用矩阵操作替代循环
    return enc.transform(label.reshape(-1, 1)).toarray().reshape(label.shape[0], label.shape[1], -1)


def my_train(train_loader, val_loader, model_name, loss="CrossEntropy", lr=0.001, batch_size=128,
             depth=6, kernel_size=80, epochs=5000, nb_filters=32, dilation=(1, 1), stride=(1, 1)):
    if model_name == 'MSSN2D':
        model = MSSN2D(input_shape=[1, 1, 1000], batch_size=batch_size, depth=depth,
                       kernel_size=kernel_size,
                       lr=lr, nb_filters=nb_filters, dilation=dilation, stride=stride, nb_classes=4)
    elif model_name == 'Inception_Seg':
        model = Inception_Seg(input_shape=[1, 1000], batch_size=batch_size, depth=depth, kernel_size=kernel_size,
                              nb_classes=4)
    elif model_name == 'UTime':
        model = UTime(input_shape=[1, 1000], batch_size=batch_size, depth=4, kernel_size=kernel_size)
    elif args.model == "UTMAS":
        model = UTMAS(input_shape=[1, 1000], nb_classes=4, N=3, q=50)
    elif model_name == 'Prectime':
        model = Prectime(1, 4)
    elif args.model == 'MS_TCN2':
        model = MS_TCN2(10, 9, 3, 64, 1, 4)
        depth = 10
    elif args.model == 'Sleep_Transformer':
        model = Transformer_lognorm_association(input_shape=[1, 1000], nb_classes=4, N=3, d_association=64,
                                                d_model=64,
                                                d_hidden=256, q=8, v=8, h=2, head_kernal_size=40, mode=1)
    elif args.model == "TimesNet":
        configs = Configs(
            task_name="segmentation", seq_len=1000, label_len=1000, pred_len=0,
            e_layers=2, d_model=32, dropout=0.1, enc_in=1, top_k=8,
            num_class=4, d_ff=256, num_kernels=3)
        model = TimesNet(configs)
    else:
        raise ValueError("Unknown model name")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    if not torch.cuda.is_available():
        raise NotImplementedError('error no gpu')

    if "MS" in model.name or model.name == "Prectime":
        criterion = CeTmseLoss()
    elif loss == "CrossEntropy":
        criterion = nn.CrossEntropyLoss()
    elif loss == "Dice":
        criterion = DiceLoss()
    elif loss == "Focal":
        criterion = FocalLoss()
    elif loss == "AsymmetricUnifiedFocalLoss":
        criterion = AsymmetricUnifiedFocalLoss()
    elif loss == "ComboLoss":
        criterion = ComboLoss()
    else:
        raise NotImplementedError(loss)
    # optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, min_lr=0.0001, verbose=True)

    best_val_acc = 0
    early_stopping_patience = 3
    early_stopping_counter = 0

    print('get writer')
    writer = SummaryWriter()

    for epoch in range(epochs):
        train_loss, train_accuracy = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_accuracy = eval_epoch(model, val_loader, criterion, device)
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/validation', val_loss, epoch)
        writer.add_scalar('Accuracy/train', train_accuracy, epoch)
        writer.add_scalar('Accuracy/validation', val_accuracy, epoch)

        print(
            f'Epoch {epoch + 1}/{epochs}, train_loss: {train_loss:.4f}, train_accuracy: {train_accuracy:.4f}, val_loss: {val_loss:.4f}, val_accuracy: {val_accuracy:.4f}'
        )
        scheduler.step(val_loss)

        if val_accuracy > best_val_acc:
            best_val_acc = val_accuracy
            early_stopping_counter = 0
            torch.save(model.state_dict(),
                       f"best_test/{dataset_name}_{model.name}_k{kernel_size}_nb{nb_filters}_d{depth}_{loss}.pth")
        else:
            early_stopping_counter += 1
            if early_stopping_counter >= early_stopping_patience:
                print(f"Early stopping at epoch {epoch}")
                break

    writer.close()


if __name__ == '__main__':
    # 加载数据并将其分为训练、验证和测试集
    shuffle_seeds = (0, 11, 34)
    use_multi_gpu = False
    parser = argparse.ArgumentParser(description='Model Training')
    parser.add_argument('--model', type=str, default="MSSN2D",
                        help='Model to train: MSSN2D, UTime, UTMAS, Prectime, Sleep_Transformer, Inception_Seg, TimesNet')

    parser.add_argument('--device', default='device')
    parser.add_argument('--data_shape', default=[1000, 1], type=int, nargs='+')
    parser.add_argument('--num_class', default=4, type=int)
    parser.add_argument('--dropout', default=0.2, type=float)
    parser.add_argument('--attn_heads', default=4, type=int)
    parser.add_argument('--eval_per_steps', default=16, type=int)
    parser.add_argument('--enable_res_parameter', default=1, type=int)
    parser.add_argument('--pooling_type', default='mean', type=str)
    parser.add_argument('--stages', default=3, type=int)
    parser.add_argument('--layer_per_stage', default=[1, 1, 1], type=int, nargs='+')
    parser.add_argument('--hidden_size_per_stage', default=[128, 128, 128], type=int, nargs='+')
    parser.add_argument('--slice_per_stage', default=[2, 2, 2], type=int, nargs='+')
    parser.add_argument('--stride_per_stage', default=[2, 2, 2], type=int, nargs='+')
    parser.add_argument('--tr', default=[2, 2, 1], type=int, nargs='+')
    parser.add_argument('--bottleneck_size', default=[128, 128, 128], type=int, nargs='+')
    parser.add_argument('--kernel_size', default=[64, 64, 64], type=int, nargs='+')
    parser.add_argument('--as_size', default=[64, 64, 64], type=int, nargs='+')
    parser.add_argument('--position_location', default='top', type=str)
    parser.add_argument('--position_type', default='cond', type=str)

    args = parser.parse_args()
    dataset_name = 'MG'
    batch_size = 128

    # 加载数据
    train_dataset = MackeyGlassDataset('mg_dataset/mackey_glass_train.csv')
    valid_dataset = MackeyGlassDataset('mg_dataset/mackey_glass_valid.csv')
    test_dataset = MackeyGlassDataset('mg_dataset/mackey_glass_test.csv')
    # Create DataLoaders for each Dataset
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # 拟合网络模型
    my_train(train_dataloader, valid_dataloader, model_name=args.model, batch_size=batch_size)
