import os

import pandas as pd
import plotly.graph_objs as go

import numpy as np
from matplotlib import pyplot as plt, patches

from TSMetric import TSMetric


class Counter:
    def __init__(self, name, label, save_pic=True, model_name="Unknown"):
        """
        初始化计数器。
        :param label: 当前计数器对应的标签。
        """
        self.label = label
        self.name = name
        self.weight = 1
        self.metrics = {
            name:
                TSMetric(metric_option=option, alpha_r=alpha_r, cardinality="reciprocal", bias_p="flat",
                         bias_r=bias_r)
            for name, option, alpha_r, bias_r in [
                ("classic", "classic", 0.0, "flat"),
                ("normal", "time-series", 0.0, "flat"),
                ("middle", "time-series", 0.0, "middle"),
                ("a1", "time-series", 1.0, "flat")
            ]
        }

        self.eval_res = {key: {"precision": 0, "recall": 0, "f1": 0, "total_real_pre": 0, "total_real_rec": 0}
                         for key in self.metrics.keys()}
        self.save_pic = save_pic
        self.model_name = model_name

    def update(self, label, predicted, actual_data, file_name, weight=1):
        """
        更新计数器的指标，并在特定条件下绘制图表。
        :param label: 真实标签。
        :param predicted: 预测标签。
        :param actual_data: 实际数据。
        :param file_name: 文件名，用于保存图表。
        :param weight: 权重。
        """
        for key, metric in self.metrics.items():
            real_label = np.where(label == self.label, 1, 0)
            pre_label = np.where(predicted == self.label, 1, 0)

            if not self._is_binary(real_label) and not self._is_binary(pre_label):
                continue

            if not self._is_binary(real_label):
                # 没有实际缺陷，因此无法计算召回率
                precision, recall, f1 = 0, 0, 0
                self.eval_res[key]["total_real_rec"] += weight

            elif not self._is_binary(pre_label):
                # 没有预测缺陷，因此无法计算精度
                precision, recall, f1 = 0, 0, 0
                self.eval_res[key]["total_real_pre"] += weight

            else:
                precision, recall, f1 = metric.score(real_label, pre_label)
                self.eval_res[key]["total_real_pre"] += weight
                self.eval_res[key]["total_real_rec"] += weight

            self.eval_res[key]["precision"] += precision * weight
            self.eval_res[key]["recall"] += recall * weight
            self.eval_res[key]["f1"] += f1 * weight

            if (0.1 < recall < 0.98 or 0.1 < precision < 0.98) and self.save_pic and key == "normal":
                # self.plot_graph_matplotlib_v3(label, predicted, actual_data, precision, recall, key)
                self._plot_graph(label, predicted, actual_data, precision, recall, file_name, key)

    def save_metrics_to_csv(self, model_name, csv_folder='metrics_csv'):
        """
        将非零的计数器值保存到CSV文件。
        :param csv_folder: 保存CSV文件的文件夹名称。
        """
        # 确保目标文件夹存在
        if not os.path.exists(csv_folder):
            os.makedirs(csv_folder)

        # 准备数据
        data = {}
        for key, metrics in self.eval_res.items():
            adjusted_metrics = self._adjust_metrics(metrics)
            for metric_name, value in adjusted_metrics.items():
                if value != 0:
                    if metric_name not in data:
                        data[metric_name] = {}
                    data[metric_name][key] = value

        # 转换为DataFrame
        df = pd.DataFrame(data)

        # 保存为CSV
        csv_file_path = os.path.join(csv_folder, f"{model_name}_{self.name}.csv")
        df.to_csv(csv_file_path)

    def _save_to_csv(self, label, predicted, channel_data, title):
        """
        将数据保存为CSV文件。
        """
        data = {
            'True Label': label,
            'Predicted Label': predicted,
            'Actual Data0': channel_data
        }
        df = pd.DataFrame(data)

        csv_folder = 'CSV_Data'
        if not os.path.exists(csv_folder):
            os.makedirs(csv_folder)

        csv_file_path = os.path.join(csv_folder, f"{title}.csv")
        df.to_csv(csv_file_path, index=False)

    def _plot_graph(self, label, predicted, actual_data, precision, recall, file_name, key):
        """
        绘制图表。
        :param label: 真实标签。
        :param predicted: 预测标签。
        :param actual_data: 实际数据。
        :param precision: 准确率。
        :param recall: 召回率。
        :param file_name: 文件名。
        :param key: 计数器键。
        """
        channel_data = actual_data[:, 8]  # 选择第8个通道的数据

        fig = go.Figure()
        fig.add_trace(go.Scatter(y=label, mode='lines', name='True Label'))
        fig.add_trace(go.Scatter(y=predicted, mode='lines', name='Predicted Label'))
        for i in range(actual_data.shape[1]):
            fig.add_trace(go.Scatter(y=actual_data[:, i], mode='lines', name=f'Actual Data{i}'))

        title = f"{key}_p{round(precision * 100, 1)}_r{round(recall * 100, 1)}_{file_name[:-4]}"
        fig.update_layout(title=title, xaxis_title="Time", yaxis_title="Value", legend_title="Legend")

        fig_folder = f'{self.model_name}_fig'
        if not os.path.exists(fig_folder):
            os.makedirs(fig_folder)

        fig_file_path = os.path.join(fig_folder, f"{title}.html")
        fig.write_html(fig_file_path)
        # 调用保存CSV的函数
        self._save_to_csv(label, predicted, channel_data, title)

    def _is_binary(self, array):
        """
        检查数组是否为二元（0和1）。
        :param array: 待检查的数组。
        :return: 如果是二元数组，则返回True，否则返回False。
        """
        return np.allclose(np.unique(array), np.array([0, 1]))

    def print_non_zero_values(self):
        """
        打印非零的计数器值，并对precision和recall进行调整。
        """
        res = {}
        print(f"Data for label {self.name}:")
        for key, metrics in self.eval_res.items():
            adjusted_metrics = self._adjust_metrics(metrics)
            if any(value != 0 for value in adjusted_metrics.values()):
                print(f" {key}: {adjusted_metrics}")
                res[key] = adjusted_metrics
        print("\n")
        return res

    def _adjust_metrics(self, metrics):
        """
        调整计数器中的metrics，对precision和recall进行计算。
        :param metrics: 原始的计数器metrics。
        :return: 调整后的metrics。
        """
        adjusted = {}
        for k, v in metrics.items():
            if k == "precision" and metrics["total_real_pre"] != 0:
                adjusted[k] = v / metrics["total_real_pre"]
            elif k == "recall" and metrics["total_real_rec"] != 0:
                adjusted[k] = v / metrics["total_real_rec"]
            else:
                adjusted[k] = v

        if adjusted["precision"] + adjusted["recall"] != 0:
            adjusted["f1"] = 2 * adjusted["precision"] * adjusted["recall"] / (
                    adjusted["precision"] + adjusted["recall"])

        return adjusted


class ModelEvaluator:
    def __init__(self, labelset_keys, label_values, save_pic, model_name):
        """
        初始化模型评估器。
        :param label_kind: 标签种类的数量，默认为12种。
        """
        self.labelset_keys = labelset_keys
        self.label_values = label_values
        self.counters = [Counter(self.labelset_keys[index], self.label_values[index], save_pic=save_pic, model_name=model_name) for index in
                         range(len(labelset_keys))]

    def evaluate(self, all_predictions, all_labels, all_preprocessed_data):
        """
        评估模型。
        :param all_predictions: 模型的所有预测结果。
        :param all_labels: 真实标签。
        :param all_preprocessed_data: 预处理后的所有数据。
        """
        real_MBM_count = 0
        for file_name, label in all_labels.items():
            predicted = all_predictions[file_name]
            actual_data = self._get_actual_data(all_preprocessed_data, file_name)
            if 1 in label:
                real_MBM_count += 1

            for counter in self.counters:
                counter.update(label, predicted, actual_data, file_name)
        print(f"MBM count: {real_MBM_count}")

    def evaluate_npy(self, all_predictions, all_labels, all_data):
        """
        评估npy格式的数据。
        :param all_predictions: 模型的所有预测结果。
        :param all_labels: 真实标签。
        :param metrics: 用于计算指标的度量方法。
        :param all_data: 预处理后的所有数据，格式为[(data, label), ...]。
        """
        npy_index = 0
        for actual_data, predicted, label in zip(all_data, all_predictions, all_labels):
            npy_index += 1
            for counter in self.counters:
                assert len(label) == len(predicted)
                counter.update(label, predicted, actual_data.T, f"{npy_index}_.csv")

    def print_non_zero_values(self, model_name):
        """
        对于每个Counter实例，打印非零的计数器值。
        """
        print(model_name)
        res = []
        for counter in self.counters:
            single_metric = counter.print_non_zero_values()
            res.append([counter.label, single_metric])

    def _get_actual_data(self, all_preprocessed_data, file_name):
        """
        根据输入数据类型获取实际数据。
        :param all_preprocessed_data: 所有预处理后的数据。
        :param file_name: 文件名或标识符。
        :return: 对应的实际数据。
        """
        if isinstance(all_preprocessed_data, list) and all(isinstance(item, tuple) for item in all_preprocessed_data):
            # 基于文件名的数据处理
            return [item for item in all_preprocessed_data if item[0] == file_name][0][1]
        elif isinstance(all_preprocessed_data, np.ndarray):
            # 直接处理npy数据
            return all_preprocessed_data
        else:
            raise ValueError("Unsupported data format for all_preprocessed_data")

    def save_all_metrics_to_csv(self, model_name, csv_folder='metrics_csv'):
        """
        为所有Counter实例保存度量到CSV文件。
        :param csv_folder: 保存CSV文件的文件夹名称。
        """
        for counter in self.counters:
            counter.save_metrics_to_csv(model_name, csv_folder)
