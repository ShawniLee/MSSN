import pandas as pd
import ModelEvaluator
from TSMetric import TSMetric
import numpy as np
from train_MG import *
from utils import TimeSeriesDataset, test_epoch
from torch.utils.data import DataLoader
from models.Transformer_lognorm_association_pytorch3 import Transformer_lognorm_association
import einops

data_splits = ('train', 'valid', 'test')


def load_csv_data(sub_dir="test"):
    data = pd.read_csv(f'mg_dataset/mackey_glass_{sub_dir}.csv')
    x = np.array(data.x)
    y = np.array(data.tau)
    replacement_dict = {10: 0, 17: 1, 30: 2, 100: 3}
    # 通过循环实现替换
    for old_value, new_value in replacement_dict.items():
        y[y == old_value] = new_value

    x = np.split(x, x.shape[0] / 1000)
    y = np.split(y, y.shape[0] / 1000)

    x = np.array(x)
    y = np.array(y)

    x = x.reshape(-1, 1000, 1)
    y = y.reshape(-1, 1000, 1)

    return x, y


parser = argparse.ArgumentParser(description='模型选择')
parser.add_argument('--model', type=str, default='MSSN2D',
                    help='Model to Test: MSSN2D, UTime, UTMAS, Prectime, Sleep_Transformer, Inception_Seg, TimesNet')
parser.add_argument('--save_pic', type=bool, default=False,
                    help='选择是否保存图片')
# FormerTime
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

# 加载数据
test_x, test_y = load_csv_data(sub_dir="test")

# 根据输入的参数选择模型
if args.model == 'UTime':
    model = UTime(input_shape=[1, 1000], batch_size=128, depth=4, nb_classes=4, kernel_size=80)
elif args.model == 'Inception_Seg':
    model = Inception_Seg(input_shape=[1, 1000], batch_size=128, depth=4, kernel_size=80, nb_classes=4)
elif args.model == 'MSSN2D':
    model = MSSN2D(input_shape=[1, 1, 1000], batch_size=256, depth=6, kernel_size=80,
                   nb_classes=4)
elif args.model == 'Prectime':
    model = Prectime(1, 4)
elif args.model == 'MS_TCN2':
    model = MS_TCN2(11, 10, 3, 64, 1, 4)
elif args.model == 'Sleep_Transformer':
    model = Transformer_lognorm_association(input_shape=[1, 1000], nb_classes=4, N=1, d_association=32,
                                            d_model=32, d_hidden=64, q=8, v=8, h=2, head_kernal_size=10, mode=0)
elif args.model == "UTMAS":
    model = UTMAS(input_shape=[1, 1000], nb_classes=4, N=3, q=50)

else:
    raise ValueError('未知的模型类型')

if args.model == 'UTime':
    model.load_state_dict(torch.load(
        f"best_test/MG_UTime_k80_nb32_d6_CrossEntropy.pth"))
elif args.model == 'Inception_Seg':
    model.load_state_dict(torch.load(
        f"best_test/MG_FCNInception1D_k60_nb32_d4_CrossEntropy.pth"))
elif args.model == 'MSSN2D':
    model.load_state_dict(torch.load(
        f"best_test/MG_MSSN2D_k80_nb32_d6_CrossEntropy.pth"))
elif args.model == 'MS_TCN2':
    model.load_state_dict(torch.load(
        f"best_test/MG_MS_TCN2_11_3.pth"))
elif args.model == 'Prectime':
    model.load_state_dict(torch.load(
        f"best_test/MG_Prectime_k40_nb32_d4_CrossEntropy.pth"))
elif args.model == 'Sleep_Transformer':
    model.load_state_dict(torch.load(
        "best_test/MG_Sleep_Transformer_preconv2_depth1_d_model32_d_association32_head_kernal_size_10_CrossEntropy_mode0.pth"))
elif args.model == "UTMAS":
    model.load_state_dict(torch.load(
        'best_test/MG_UTMAS_k80_nb32_d6_CrossEntropy.pth'))


model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

# # 重排数据以匹配网络的输入要求
test_x = torch.tensor(np.array(test_x), dtype=torch.float32)
test_x = einops.rearrange(test_x, 'b l c -> b c l')

test_y = einops.rearrange(test_y, 'b l 1 -> b l')
batch_size = 128
test_dataset = TimeSeriesDataset(test_x, test_y)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

predictions, labels = test_epoch(test_loader, model, torch.device("cuda" if torch.cuda.is_available() else "cpu"))
metrics = [
    TSMetric(metric_option=option, alpha_r=alpha_r, cardinality="reciprocal", bias_p="flat", bias_r=bias_r)
    for option, alpha_r, bias_r in [
        ("classic", 0.0, "flat"),
        ("time-series", 0.0, "flat"),
        ("time-series", 0.0, "middle"),
        ("time-series", 1.0, "flat")
    ]
]

evaluates = ModelEvaluator.ModelEvaluator(['10', '17', '30', '100'], [0, 1, 2, 3], args.save_pic,
                                          model_name=args.model)
print("actual label set: ", np.unique([item for sublist in labels for item in sublist]))

evaluates.evaluate_npy(predictions, labels, test_x)
evaluates.print_non_zero_values(model.name)
evaluates.save_all_metrics_to_csv(model.name, csv_folder='mg_metrics_csv')
