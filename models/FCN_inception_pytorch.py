import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn.functional as F
from torch.utils.data import DataLoader
from utils import *


class InceptionModule(nn.Module):

    def __init__(self, in_channels, nb_filters, use_bottleneck=True, bottleneck_size=32, kernel_size=40):
        super(InceptionModule, self).__init__()
        self.use_bottleneck = False
        self.bottleneck_size = bottleneck_size

        if use_bottleneck and in_channels > self.bottleneck_size:
            self.use_bottleneck = use_bottleneck
            self.bottleneck = nn.Sequential(
                nn.Conv1d(in_channels, self.bottleneck_size, kernel_size=1, bias=False),
                nn.BatchNorm1d(self.bottleneck_size),
                nn.ReLU()
            )
            in_channels = self.bottleneck_size

        kernel_size_s = [kernel_size // (2 ** i) for i in range(3)]

        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(in_channels, nb_filters, kernel_size=k_size, padding='same', bias=False),
                nn.BatchNorm1d(nb_filters),
                nn.ReLU()
                #   Keras实现中没有添加激活函数
            )
            for k_size in kernel_size_s
        ])

        self.conv1x1 = nn.Sequential(
            nn.Conv1d(in_channels, nb_filters, kernel_size=1, bias=False),
            nn.BatchNorm1d(nb_filters),
            nn.ReLU()
        )

    def forward(self, x):
        if self.use_bottleneck:
            x = self.bottleneck(x)

        conv_outputs = [conv(x) for conv in self.convs]
        max_pool_output = self.conv1x1(F.max_pool1d(x, kernel_size=3, stride=1, padding=1))

        return torch.cat(conv_outputs + [max_pool_output], dim=1)


class ShortcutLayer(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(ShortcutLayer, self).__init__()

        self.conv1x1 = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=1, padding='same', bias=False),
            nn.BatchNorm1d(out_channels)
        )

    def forward(self, x1, x2):
        x1 = self.conv1x1(x1)
        return F.relu(x1 + x2)


class FCNHead(nn.Module):

    def __init__(self, in_channels, channels):
        super(FCNHead, self).__init__()

        self.layers = nn.Sequential(
            nn.Conv1d(in_channels, in_channels // 4, padding='same', kernel_size=1, bias=False),
            nn.BatchNorm1d(in_channels // 4),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv1d(in_channels // 4, channels, padding='same', kernel_size=1, bias=False)
        )

    def forward(self, x):
        return F.softmax(self.layers(x), dim=1)


class Inception_Seg(nn.Module):

    def __init__(self, input_shape, nb_classes=2, batch_size=64, lr=0.001, nb_filters=32, use_residual=True,
                 use_bottleneck=True, depth=6, kernel_size=41, nb_epochs=1500):

        super(Inception_Seg, self).__init__()

        self.nb_filters = nb_filters
        self.use_residual = use_residual
        self.use_bottleneck = use_bottleneck
        self.depth = depth
        self.kernel_size = kernel_size - 1
        self.nb_epochs = nb_epochs

        self.batch_size = batch_size
        self.bottleneck_size = 32
        self.nb_epochs = nb_epochs
        self.lr = lr
        self.instance_norm = nn.InstanceNorm1d(1, affine=False)
        self.name = "Inception_Seg"


        layers = []
        input_raw = input_shape
        for d in range(self.depth):
            layers.append(InceptionModule(input_shape[0], self.nb_filters, use_bottleneck=self.use_bottleneck))
            if self.use_residual and d % 3 == 2:
                layers.append(ShortcutLayer(input_raw[0], self.nb_filters * 4))
                input_raw = (self.nb_filters * 4, input_shape[1])

            input_shape = (self.nb_filters * 4, input_shape[1])

        self.layers = nn.Sequential(*layers)

        self.head = FCNHead(input_shape[0], nb_classes)

    def forward(self, x):
        x = self.instance_norm(x)

        input_res = x
        for layer in self.layers:
            if isinstance(layer, ShortcutLayer):
                x = layer(input_res, x)
                input_res = x
            else:
                x = layer(x)
        x = self.head(x)
        return x.squeeze(-1)

    def fit(self, x_train, y_train, x_val, y_val):
        def create_data_loaders(x_train, y_train, x_val, y_val, batch_size):
            x_train_tensor = torch.from_numpy(x_train).float().to(device)
            y_train_tensor = torch.from_numpy(y_train).float().to(device)
            x_val_tensor = torch.from_numpy(x_val).float().to(device)
            y_val_tensor = torch.from_numpy(y_val).float().to(device)

            train_dataset = TimeSeriesDataset(x_train_tensor, y_train_tensor)
            val_dataset = TimeSeriesDataset(x_val_tensor, y_val_tensor)

            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

            return train_loader, val_loader

        def train_epoch(model, loader, criterion, optimizer):
            model.train()
            running_loss = 0
            running_corrects = 0

            for batch_x, batch_y in loader:
                optimizer.zero_grad()
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                running_corrects += (predicted == torch.max(batch_y, 1)[1]).sum().item()

            epoch_loss = running_loss / len(loader.dataset)
            epoch_accuracy = running_corrects / len(loader.dataset)

            return epoch_loss, epoch_accuracy

        def eval_epoch(model, loader, criterion):
            model.eval()
            running_loss = 0
            running_corrects = 0

            with torch.no_grad():
                for batch_x, batch_y in loader:
                    outputs = model(batch_x)
                    loss = criterion(outputs, batch_y)

                    running_loss += loss.item()
                    _, predicted = torch.max(outputs, 1)
                    running_corrects += (predicted == torch.max(batch_y, 1)[1]).sum().item()

            epoch_loss = running_loss / len(loader.dataset)
            epoch_accuracy = running_corrects / len(loader.dataset)

            return epoch_loss, epoch_accuracy

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device)

        if not torch.cuda.is_available():
            print('error no gpu')
            exit()

        criterion = nn.CrossEntropyLoss()
        # criterion = DiceLoss()
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, min_lr=0.0001, verbose=True)

        best_val_loss = float('inf')
        early_stopping_patience = 11
        early_stopping_counter = 0

        writer = SummaryWriter()

        train_loader, val_loader = create_data_loaders(x_train, y_train, x_val, y_val, self.batch_size)

        for epoch in range(self.nb_epochs):
            train_loss, train_accuracy = train_epoch(self, train_loader, criterion, optimizer)
            val_loss, val_accuracy = eval_epoch(self, val_loader, criterion)

            writer.add_scalar('Loss/train', train_loss, epoch)
            writer.add_scalar('Loss/validation', val_loss, epoch)
            writer.add_scalar('Accuracy/train', train_accuracy, epoch)
            writer.add_scalar('Accuracy/validation', val_accuracy, epoch)

            print(
                f'Epoch {epoch + 1}/{self.nb_epochs}, train_loss: {train_loss:.4f}, train_accuracy: {train_accuracy:.4f}, '
                f'val_loss: {val_loss:.4f}, val_accuracy: {val_accuracy:.4f}')

            scheduler.step(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                early_stopping_counter = 0
                torch.save(self.state_dict(), "../best_model.pth")
            else:
                early_stopping_counter += 1
                if early_stopping_counter >= early_stopping_patience:
                    print(f"Early stopping at epoch {epoch}")
                    break

        writer.close()
        self.load_state_dict(torch.load("best_model.pth"))
