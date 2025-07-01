import numpy as np
import pyedflib
from scipy.signal import resample, butter, filtfilt, iirnotch
import numpy as np
import matplotlib.pyplot as plt


def plot_eeg_waveform(processed_signals, start_time, end_time, sampling_rate=64):
    """
    绘制指定时间范围内 EEG 信号的波形图，18 个通道分别绘制。

    Parameters:
        processed_signals (numpy.ndarray): EEG 数据，形状为 (18, n_samples)。
        start_time (int): 绘制信号的起始时间（单位：秒）。
        end_time (int): 绘制信号的结束时间（单位：秒）。
        sampling_rate (int): 采样率（默认为 64 Hz）。
    """
    # Step 1: 计算切片范围（时间点到样本点的映射）
    start_sample = int(start_time * sampling_rate)
    end_sample = int(end_time * sampling_rate)

    # Step 2: 提取对应时间段信号
    signal_segment = processed_signals[:, start_sample:end_sample]

    # Step 3: 创建时间轴（以秒为单位）
    times = np.linspace(start_time, end_time, signal_segment.shape[1])

    # Step 4: 绘制波形
    plt.figure(figsize=(12, 8))
    offset = 20  # 设置通道之间的间隔，避免信号重叠
    for i in range(signal_segment.shape[0]):  # 遍历每个通道
        plt.plot(times, signal_segment[i] + i * offset, label=f"Channel {i + 1}")

        # 设置图示
    plt.title(f"EEG Signals from {start_time} s to {end_time} s", fontsize=16)
    plt.xlabel("Time (s)", fontsize=12)
    plt.ylabel("Amplitude (with Offset)", fontsize=12)
    plt.yticks(
        ticks=[i * offset for i in range(signal_segment.shape[0])],
        labels=[f"Channel {i + 1}" for i in range(signal_segment.shape[0])],
        fontsize=10,
    )
    plt.grid(linestyle="--", alpha=0.6)
    plt.legend(loc="upper right", fontsize=10)
    plt.tight_layout()
    plt.show()


def read_edf(file_path):
    """
    读取 EDF 文件并提取原始信号和通道标签。

    Parameters:
        file_path (str): EDF 文件路径。

    Returns:
        signals (numpy.ndarray): 信号数据数组，形状为 (n_samples, n_channels)。
        channel_labels (list): 通道标签。
        sampling_rate (int): 信号原始采样率。
    """
    with pyedflib.EdfReader(file_path) as reader:
        n_channels = reader.signals_in_file
        signals = np.array([reader.readSignal(i) for i in range(n_channels)]).T
    return signals


def generate_bipolar_montage(signals, channel_labels):
    """
    生成双极蒙太奇（bipolar montage），计算两通道之间的差值。

    Parameters:
        signals (numpy.ndarray): 信号数据，形状为 (n_samples, n_channels)。
        channel_labels (list): 包含信号对应的通道标签，对应信号列顺序。

    Returns:
        montage_signals (numpy.ndarray): 生成的双极蒙太奇信号，形状为 (n_samples, 18)。
    """
    # 使用新提供的双极蒙太奇定义
    bipolar_pairs = [
        ('Fp2', 'F4'), ('F4', 'C4'), ('C4', 'P4'),
        ('P4', 'O2'), ('Fp1', 'F3'), ('F3', 'C3'),
        ('C3', 'P3'), ('P3', 'O1'), ('Fp2', 'F8'),
        ('F8', 'T4'), ('T4', 'T6'), ('T6', 'O2'),
        ('Fp1', 'F7'), ('F7', 'T3'), ('T3', 'T5'),
        ('T5', 'O1'), ('Fz', 'Cz'), ('Cz', 'Pz')
    ]

    montage_signals = []
    for pair in bipolar_pairs:
        try:
            # 获取每个通道的索引
            idx1 = channel_labels.index(pair[0])
            idx2 = channel_labels.index(pair[1])

            # 计算两通道信号差值
            montage_signals.append(signals[:, idx1] - signals[:, idx2])
        except ValueError as e:
            raise ValueError(f"Channel {pair} does not exist, please check the channel name.") from e

    return np.array(montage_signals).T  # 转置为 (n_samples, 18)


def get_bipolar_montage(signals, channel_labels):
    bipolar_pairs = [
        'Fp2-F4', 'F4-C4', 'C4-P4',
        'P4-O2', 'Fp1-F3', 'F3-C3',
        'C3-P3', 'P3-O1', 'Fp2-F8',
        'F8-T4', 'T4-T6', 'T6-O2',
        'Fp1-F7', 'F7-T3', 'T3-T5',
        'T5-O1', 'Fz-Cz', 'Cz-Pz'
    ]

    montage_signals = []
    for pair in bipolar_pairs:
        try:
            # 获取每个通道的索引
            idx = channel_labels.index(pair)

            # 计算两通道信号差值
            montage_signals.append(signals[:, idx])
        except ValueError as e:
            raise ValueError(f"Channel {pair} does not exist, please check the channel name.") from e

    return np.array(montage_signals).T  # 转置为 (n_samples, 18)


def apply_filters(signals, sampling_rate):
    """
    对信号应用 50Hz 陷波滤波和带通滤波（0.3Hz ~ 30Hz）。

    Parameters:
        signals (numpy.ndarray): 原始信号数据，(n_samples, n_channels)。
        sampling_rate (int): 原始采样率。

    Returns:
        filtered_signals (numpy.ndarray): 经滤波的信号数据。
    """
    # 50Hz 陷波滤波器
    notch_freq = 50
    quality_factor = 30
    nyquist = 0.5 * sampling_rate
    w0 = notch_freq / nyquist
    b_notch, a_notch = iirnotch(w0, quality_factor)

    # 带通滤波器（Butterworth）
    lowcut = 0.3
    highcut = 30
    b_band, a_band = butter(4, [lowcut / nyquist, highcut / nyquist], btype="band")

    # 初始化滤波后的信号矩阵
    filtered_signals = np.zeros_like(signals)

    # 对每个通道应用滤波器
    for i in range(signals.shape[1]):
        # 1. 陷波滤波
        notch_filtered_signal = filtfilt(b_notch, a_notch, signals[:, i])
        # 2. 带通滤波
        filtered_signal = filtfilt(b_band, a_band, notch_filtered_signal)
        filtered_signals[:, i] = filtered_signal

    return filtered_signals


def downsample(signals, original_sampling_rate, target_sampling_rate):
    """
    使用 scipy.signal.resample 函数对信号进行降采样。

    Parameters:
        signals (numpy.ndarray): 输入信号数据，形状为 (n_samples, n_channels)。
        original_sampling_rate (int): 原始采样率。
        target_sampling_rate (int): 目标采样率。

    Returns:
        downsampled_signals (numpy.ndarray): 降采样后的信号数据，形状为 (n_downsampled_samples, n_channels)。
    """
    n_samples, n_channels = signals.shape
    target_n_samples = int(n_samples * target_sampling_rate / original_sampling_rate)  # 计算目标采样点数
    downsampled_signals = np.zeros((target_n_samples, n_channels))  # 初始化降采样结果

    # 对每个通道的信号分别执行重采样
    for ch in range(n_channels):
        downsampled_signals[:, ch] = resample(signals[:, ch], target_n_samples)

    return downsampled_signals


def process_eeg(file_path, channel_labels, original_sampling_rate, target_sampling_rate, bipolar=False):
    """
    主函数：读取 EDF 文件、生成双极蒙太奇、滤波、降采样。

    Parameters:
        file_path (str): EDF 文件路径。
        target_sampling_rate (int): 最终目标采样率（默认 64Hz）。

    Returns:
        processed_signals (numpy.ndarray): 经处理的信号数据，形状为 (18, n)。
    """
    # Step 1: 读取 EDF 文件
    signals = read_edf(file_path)

    # Step 2: 生成双极蒙太奇
    if bipolar:
        montage_signals = get_bipolar_montage(signals, channel_labels)
    else:
        montage_signals = generate_bipolar_montage(signals, channel_labels)

    # Step 3: 对蒙太奇信号进行滤波
    filtered_signals = apply_filters(montage_signals, original_sampling_rate)

    # Step 4: 将滤波后的信号降采样
    downsampled_signals = downsample(filtered_signals, original_sampling_rate, target_sampling_rate)

    # plot_eeg_waveform(downsampled_signals.T, 103, 120, 64)

    return downsampled_signals.T  # 最终返回 (18, n)


# 示例使用
# if __name__ == "__main__":
#     file_path = "../edf/eeg2.edf"  # 替换为实际的 EDF 文件路径
#     channel_labels = ["Fp1", "Fp2", "F3", "F4", "F7", "F8", "Fz", "C3", "C4", "Cz", "T3", "T5", "T4", "T6", "P3", "P4", "Pz", "O1", "O2"]
#     processed_signals = process_eeg(file_path, channel_labels, original_sampling_rate=256, target_sampling_rate=64)
#     # np.save("bipolar_eeg_data.npy", processed_signals)
#
#     print(f"Processed signals shape: {processed_signals.shape}")  # 输出形状为 (18, n)
