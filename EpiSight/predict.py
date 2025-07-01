import json
import torch
import os
import numpy as np
from pathlib import Path
from utils import *
from model.SCNet import SCC
import torch.nn.functional as F
from model.dataloader import process_eeg
from torch.utils.data import DataLoader, TensorDataset

current_dir = Path(os.path.abspath(__file__)).parent

def analyze_seizure_tensor(arr):
    # Ensure 1D numpy array
    arr = np.asarray(arr).flatten()
    n = len(arr)

    # Total durations
    total_seconds = n
    normal_seconds = np.sum(arr == 0)
    seizure_seconds = np.sum(arr == 1)

    # Individual seizure duration stats
    in_seizure = False
    seizure_durations = []
    current_duration = 0

    for v in arr:
        if v == 1:
            if not in_seizure:
                in_seizure = True
                current_duration = 1
            else:
                current_duration += 1
        else:
            if in_seizure:
                seizure_durations.append(current_duration)
                in_seizure = False
                # Handle ending with 1
    if in_seizure:
        seizure_durations.append(current_duration)

    if seizure_durations:
        avg_seizure = sum(seizure_durations) / len(seizure_durations)
        max_seizure = max(seizure_durations)
        min_seizure = min(seizure_durations)
    else:
        avg_seizure = max_seizure = min_seizure = 0

    ratio = 100 * seizure_seconds / total_seconds if total_seconds > 0 else 0

    result = {
        'total_duration': int(total_seconds),
        'normal_duration': int(normal_seconds),
        'seizure_duration': int(seizure_seconds),
        'average_single_seizure_duration': int(avg_seizure),
        'min_single_seizure_duration': int(min_seizure),
        'max_single_seizure_duration': int(max_seizure),
        'seizure_ratio': ratio
    }
    return result

def predictor(dataloader):
    # 2. 初始化模型
    model_A = SCC(1, 18, 128, 2)
    model_B = SCC(1, 18, 128, 2)
    model_C = SCC(1, 18, 128, 2)

    # 3. 加载模型权重
    state_dict_A = torch.load(current_dir / "model/expert_A.pt", weights_only=True, map_location=torch.device('cpu'))
    state_dict_B = torch.load(current_dir / "model/expert_B.pt", weights_only=True,map_location=torch.device('cpu'))
    state_dict_C = torch.load(current_dir / "model/expert_C.pt", weights_only=True,map_location=torch.device('cpu'))

    model_A.load_state_dict(state_dict_A)
    model_B.load_state_dict(state_dict_B)
    model_C.load_state_dict(state_dict_C)

    # 设置为评估模式（避免 Dropout 和 BatchNorm 的影响）
    model_A.eval()
    model_B.eval()
    model_C.eval()

    # 如果有 CUDA，使用 GPU
    device = torch.device("cpu")
    model_A.to(device)
    model_B.to(device)
    model_C.to(device)

    # 4. 开始预测
    all_predictions = []  # 存储所有预测结果
    # 初始化变量，存放所有样本中类别最可能为 1 的数据和分类值
    max_likelihood_sample = None  # 保存数据的变量
    max_likelihood_value = -float("inf")  # 保存最大可能性的值

    with torch.no_grad():  # 禁用梯度计算
        for batch_data in dataloader:
            # 将数据送到 GPU（如果可用）
            inputs = batch_data[0].to(device)

            # 单个模型预测（输出经过 sigmoid）
            pred_A = model_A(inputs)
            pred_B = model_B(inputs)
            pred_C = model_C(inputs)

            # 软投票：求三个模型输出的平均值（按元素平均）
            soft_votes = (pred_A + pred_B + pred_C) / 3.0

            # 使用阈值 0.5，生成最终的类别预测结果
            batch_predictions = (soft_votes > 0.5).long()

            # 找到当前批次中类别为 1 的最大可能值和对应样本
            batch_max_likelihood_value, batch_max_likelihood_idx = torch.max(soft_votes[:, 0], dim=0)  # 假设是二分类问题
            if batch_max_likelihood_value > max_likelihood_value:
                max_likelihood_value = batch_max_likelihood_value.item()  # 更新最大值
                max_likelihood_sample = inputs[batch_max_likelihood_idx].cpu()  # 保存该样本数据

            # 将当前批次的预测保存
            all_predictions.append(batch_predictions.cpu())  # 转回 CPU

    # 保存最终结果到变量
    most_likely_sample_value = max_likelihood_value  # 样本的最大可能性值
    most_likely_sample = max_likelihood_sample  # 数据对应的样本

    # 拼接所有批次的预测
    all_predictions = torch.cat(all_predictions, dim=0)

    return all_predictions, most_likely_sample, most_likely_sample_value


def predict(edf_path, channels, original_sampling_rate, target_sampling_rate, bipolar):
    channel_labels = get_channel_labels(channels, bipolar)
    data = process_eeg(edf_path, channel_labels, original_sampling_rate, target_sampling_rate, bipolar)
    # Convert to PyTorch Tensor
    data_tensor = torch.tensor(data, dtype=torch.float32)  # shape: (c, d)

    # Get the number of data points in 1 second based on the target sampling rate
    points_per_second = int(target_sampling_rate)

    # Calculate the number of complete 1-second samples
    total_points = data_tensor.shape[1]
    num_complete_segments = total_points // points_per_second

    # Discard the incomplete segment
    truncated_length = num_complete_segments * points_per_second
    data_tensor = data_tensor[:, :truncated_length]  # shape: (c, truncated_length)

    # Split the data into 1-second segments
    data_tensor = data_tensor.view(data_tensor.shape[0], num_complete_segments,
                                   points_per_second)  # shape: (c, n, d1)
    data_tensor = data_tensor.permute(1, 2, 0)  # shape: (n, d1, c)
    data_tensor = F.normalize(data_tensor, dim=1)

    # unsqueeze to add a new dimension
    data_tensor = data_tensor.unsqueeze(1)  # shape: (n, 1, d1, c)

    # 创建 TensorDataset
    dataset = TensorDataset(data_tensor)

    # 创建 DataLoader
    dataloader = DataLoader(dataset, batch_size=64, shuffle=False)

    all_predictions, most_likely_sample, most_likely_sample_value = predictor(dataloader)

    most_likely_sample = most_likely_sample[0].T.numpy()


    features = extract_eeg_features(most_likely_sample, target_sampling_rate)
    features["overview"] = analyze_seizure_tensor(all_predictions)
    features.update(get_eeg_visualization_data(most_likely_sample, target_sampling_rate))
    
    return features

if __name__ == '__main__':
    file_path = "upload\eeg1.edf"
    fs = 256
    bipolar = False
    channels = {"Fp1":1, "Fp2":2, "F3":3, "F4":4, "F7":5, "F8":6, "Fz":7, "C3":8, "C4":9, "Cz":10, "T3":11, "T5":12, "T4":13, "T6":14, "P3":15, "P4":16, "Pz":17, "O1":18, "O2":19}
    with open("rst.json", "w") as f:
        json.dump(predict(file_path, channels, fs, 64, bipolar), f)
    # predict(file_path, channels, fs, 64, bipolar)