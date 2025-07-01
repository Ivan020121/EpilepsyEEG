import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch


def get_channel_labels(channels, bipolar):
    """
    获取通道列表。

    Parameters:
        channels (dict): 包含通道名称（键）及对应位置（值）的字典，值从1开始。
                         - bipolar=False: 如 {'Fp1': 1, 'Fp2': 2, ...}
                         - bipolar=True: 如 {'Fp2-F4': 1, 'F4-C4': 2, ...}
        bipolar (bool): 是否为双极通道配置。
                        - True: 使用双极通道键。
                        - False: 使用单极通道键。

    Returns:
        list: 根据 `channels` 值（从1开始）排序的通道名列表，缺失位置用 None 填充。
    """
    if bipolar:
        # 双极通道的所有可能名称
        all_possible_channels = [
            'Fp2-F4', 'F4-C4', 'C4-P4', 'P4-O2',
            'Fp1-F3', 'F3-C3', 'C3-P3', 'P3-O1',
            'Fp2-F8', 'F8-T4', 'T4-T6', 'T6-O2',
            'Fp1-F7', 'F7-T3', 'T3-T5', 'T5-O1',
            'Fz-Cz', 'Cz-Pz'
        ]
    else:
        # 单极通道的所有可能名称
        all_possible_channels = [
            "Fp1", "Fp2", "F3", "F4", "F7", "F8", "Fz",
            "C3", "C4", "Cz", "T3", "T5", "T4", "T6",
            "P3", "P4", "Pz", "O1", "O2"
        ]

    # 获取需要的列表长度（最大值 + 1）
    max_index = max(channels.values()) if channels else 0
    channel_labels = [None] * max_index  # 初始化长度为最大索引值的列表（从1开始）

    # 根据 `channels` 填充结果，为了对齐 Python 索引，需将索引减1
    for key, idx in channels.items():
        if 1 <= idx <= max_index:  # 确保索引合法（从1开始）
            channel_labels[idx - 1] = key  # 将索引减1，以对齐 Python 的0索引

    return channel_labels


def plot_eeg_figures(eeg_data, fs, svg_prefix="eeg"):
    """
    绘制EEG分析图像，并全部保存为SVG文件。
    """
    bipolar_pairs = [
        'Fp2-F4', 'F4-C4', 'C4-P4',
        'P4-O2', 'Fp1-F3', 'F3-C3',
        'C3-P3', 'P3-O1', 'Fp2-F8',
        'F8-T4', 'T4-T6', 'T6-O2',
        'Fp1-F7', 'F7-T3', 'T3-T5',
        'T5-O1', 'Fz-Cz', 'Cz-Pz'
    ]
    n_channels, n_samples = eeg_data.shape
    time = np.linspace(0, n_samples / fs, n_samples)

    # 1. 原始信号时域叠加图
    plt.figure(figsize=(12, 10))
    offset = 1
    cmap = plt.get_cmap('tab20c')
    for i in range(n_channels):
        plt.plot(time, eeg_data[i] + i * offset, color=cmap(i % 20))
    plt.xlabel("Time (s)", fontsize=20, color='black')
    plt.ylabel("Channel", fontsize=20, color='black')
    plt.title("Raw EEG Signal (Time Domain Overlay)", fontsize=25, color='black', fontweight='bold')
    plt.yticks([i * offset for i in range(n_channels)], bipolar_pairs, fontsize=12, color='black')
    plt.xticks(fontsize=12, color='black')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(f"{svg_prefix}_time_domain.svg", format="svg")
    plt.close()

    # 2. PSD 功率谱密度图
    plt.figure(figsize=(12, 6))
    n_per_seg = int(2 * fs)
    for i in range(n_channels):
        f, psd = welch(eeg_data[i], fs=fs, nperseg=n_per_seg)
        plt.semilogy(f, psd, label=bipolar_pairs[i], color=cmap(i % 20))
    plt.xlabel("Frequency (Hz)", fontsize=20)
    plt.ylabel("PSD (μV²/Hz)", fontsize=20)
    plt.title("Power Spectral Density (PSD)", fontsize=25, fontweight='bold')
    plt.legend(loc="upper right", bbox_to_anchor=(1.15, 1), fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    plt.savefig(f"{svg_prefix}_psd.svg", format="svg")
    plt.close()

    # 3. 频带堆叠条形图
    bands = {
        "Delta (0.5-4 Hz)": (0.5, 4),
        "Theta (4-8 Hz)": (4, 8),
        "Alpha (8-12 Hz)": (8, 12),
        "Beta (12-30 Hz)": (12, 30),
        "Gamma (30-50 Hz)": (30, 50),
    }
    band_power = {band: [] for band in bands}
    for i in range(n_channels):
        f, psd = welch(eeg_data[i], fs=fs, nperseg=n_per_seg)
        for band, (fmin, fmax) in bands.items():
            idx = (f >= fmin) & (f <= fmax)
            band_power[band].append(np.sum(psd[idx]))
    x = np.arange(n_channels) + 1
    bottom = np.zeros(n_channels)
    colors = ['#4682B4', '#6495ED', '#008B8B', '#00BFFF', '#7B68EE']
    plt.figure(figsize=(12, 6))
    for i, (band_name, power) in enumerate(band_power.items()):
        plt.bar(x, power, bottom=bottom, label=band_name, color=colors[i])
        bottom += power
    plt.xlabel("Channel", fontsize=20)
    plt.ylabel("Power (μV²)", fontsize=20)
    plt.title("Frequency Band Power per Channel", fontsize=25, fontweight='bold')
    plt.legend(fontsize=20, loc="center left", bbox_to_anchor=(1, 0.7))
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(f"{svg_prefix}_band_power.svg", format="svg")
    plt.close()

    # 4. RMS 振幅曲线
    rms = np.sqrt(np.mean(eeg_data ** 2, axis=0))
    plt.figure(figsize=(12, 6))
    plt.plot(time, rms, label="RMS Amplitude", color='#1f77b4')
    plt.xlabel("Time (s)", fontsize=20)
    plt.ylabel("RMS (μV)", fontsize=20)
    plt.title("RMS Amplitude Across Time", fontsize=25, fontweight='bold')
    plt.legend(fontsize=14)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(f"{svg_prefix}_rms.svg", format="svg")
    plt.close()

    # 5. 协方差矩阵热图
    covariance = np.cov(eeg_data)
    plt.figure(figsize=(12, 10))
    im = plt.imshow(covariance, cmap="plasma", aspect="auto")
    cbar = plt.colorbar(im, label="Covariance")
    cbar.ax.tick_params(labelsize=14)
    cbar.set_label(label="Covariance", size=18)
    plt.title("Channel Covariance Heatmap", fontsize=25, fontweight='bold')
    plt.xlabel("Channels", fontsize=20)
    plt.ylabel("Channels", fontsize=20)
    plt.xticks(range(n_channels), [f"{bipolar_pairs[i]}" for i in range(n_channels)], rotation=45, ha="right",
               fontsize=10)
    plt.yticks(range(n_channels), [f"{bipolar_pairs[i]}" for i in range(n_channels)], fontsize=10)
    plt.tight_layout()
    plt.savefig(f"{svg_prefix}_covariance.svg", format="svg")
    plt.close()


def get_eeg_visualization_data(eeg_data, fs):
    """
    提取EEG可视化所需的数据，用于前端绘制图表。
    
    参数:
        eeg_data (np.ndarray): EEG信号数据，形状为(n_channels, n_samples)
        fs (float): 采样率(Hz)
        
    返回:
        dict: 包含时域信号、PSD、频带功率和协方差矩阵等数据的字典
    """
    bipolar_pairs = [
        'Fp2-F4', 'F4-C4', 'C4-P4',
        'P4-O2', 'Fp1-F3', 'F3-C3',
        'C3-P3', 'P3-O1', 'Fp2-F8',
        'F8-T4', 'T4-T6', 'T6-O2',
        'Fp1-F7', 'F7-T3', 'T3-T5',
        'T5-O1', 'Fz-Cz', 'Cz-Pz'
    ]
    n_channels, n_samples = eeg_data.shape
    time = np.linspace(0, n_samples / fs, n_samples)
    
    # 1. 时域信号数据
    time_domain_data = {
        'time': time.tolist(),
        'signals': [(eeg_data[i]+i+1).tolist() for i in range(n_channels)],
    }
    
    # 2. PSD数据
    n_per_seg = int(2 * fs)
    psd_data = {'frequencies': [], 'psd_values': []}
    for i in range(n_channels):
        f, psd = welch(eeg_data[i], fs=fs, nperseg=n_per_seg)
        psd_data['frequencies'] = f.tolist()
        psd_data['psd_values'].append(psd.tolist())
    
    # 3. 频带功率数据
    bands = {
        "Delta (0.5-4 Hz)": (0.5, 4),
        "Theta (4-8 Hz)": (4, 8),
        "Alpha (8-12 Hz)": (8, 12),
        "Beta (12-30 Hz)": (12, 30),
        "Gamma (30-50 Hz)": (30, 50),
    }
    band_power = {band: [] for band in bands}
    for i in range(n_channels):
        f, psd = welch(eeg_data[i], fs=fs, nperseg=n_per_seg)
        for band, (fmin, fmax) in bands.items():
            idx = (f >= fmin) & (f <= fmax)
            band_power[band].append(float(np.sum(psd[idx])))
    # 4. RMS 振幅曲线

    rms = np.sqrt(np.mean(eeg_data ** 2, axis=0))
    rms_data = {
        'time': time.tolist(),
        'rms_values': rms.tolist()
    }
    
    return {
        'time_domain': time_domain_data,
        'psd': psd_data,
        'band_power': band_power,
        'rms_amplitude': rms_data,
        'channel_labels': bipolar_pairs
    }

def extract_eeg_features(eeg_data, fs):
    """
    提取 EEG 的各种数值特征，全部字典化返回。
    """
    bipolar_pairs = [
        'Fp2-F4', 'F4-C4', 'C4-P4',
        'P4-O2', 'Fp1-F3', 'F3-C3',
        'C3-P3', 'P3-O1', 'Fp2-F8',
        'F8-T4', 'T4-T6', 'T6-O2',
        'Fp1-F7', 'F7-T3', 'T3-T5',
        'T5-O1', 'Fz-Cz', 'Cz-Pz'
    ]
    n_channels, n_samples = eeg_data.shape

    # 1. 时域统计特征
    time_domain_stats = {}
    for i in range(n_channels):
        signal = eeg_data[i]
        time_domain_stats[bipolar_pairs[i]] = [
            float(np.mean(signal)),
            float(np.std(signal)),
            float(np.max(signal)),
            float(np.min(signal)),
            float(np.median(signal)),
            float(np.max(signal) - np.min(signal))
        ]

    # 2. 频域统计特征和频带功率占比
    frequency_stats = {}
    n_per_seg = int(2 * fs)
    bands = {
        "Delta (0.5-4 Hz)": (0.5, 4),
        "Theta (4-8 Hz)": (4, 8),
        "Alpha (8-12 Hz)": (8, 12),
        "Beta (12-30 Hz)": (12, 30),
        "Gamma (30-50 Hz)": (30, 50),
    }
    for i in range(n_channels):
        f, psd = welch(eeg_data[i], fs=fs, nperseg=n_per_seg)
        total_power = np.sum(psd)
        band_ratio = [total_power.astype(float)]
        for fmin, fmax in bands.values():
            idx_band = (f >= fmin) & (f <= fmax)
            band_power_value = np.sum(psd[idx_band]) if np.any(idx_band) else 0.
            band_ratio.append(float(band_power_value / total_power * 100 if total_power > 0 else 0))
        frequency_stats[bipolar_pairs[i]] = band_ratio

    # 3. RMS
    rms_per_channel = np.sqrt(np.mean(eeg_data ** 2, axis=1))

    rms_stats = [
        [float(rms_per_channel[i]) for i in range(n_channels)],
        float(np.mean(rms_per_channel)),
        float(np.std(rms_per_channel))
    ]

    # 4. 协方差矩阵
    covariance = np.cov(eeg_data)
    mean_cov_per_channel = np.mean(covariance, axis=1)
    cov_stats = [
        float(np.mean(covariance)),
        float(np.max(covariance)),
        float(np.min(covariance)),
        # [float(mean_cov_per_channel[i]) for i in range(n_channels)],
        np.cov(eeg_data).tolist()
    ]

    # 组合所有特征
    features = {
        "TimeDomainStats": time_domain_stats,
        "FrequencyStats": frequency_stats,
        "RMS": rms_stats,
        "Covariance": cov_stats
    }
    return features

# 生成测试数据 (18 个通道，每秒 256 个采样点，持续 10 秒)
np.random.seed(42)
n_channels, fs, duration = 18, 256, 10
n_samples = fs * duration
eeg_data = np.random.randn(n_channels, n_samples) * 50  # 模拟数据，单位 μV


def visualize_eeg(eeg_data, fs):
    """
    可视化 EEG 信号的各类分析结果，包括时域分析、频域分析和统计分析。

    参数：
        eeg_data (np.ndarray): EEG 信号数据，形状为 (n_channels, n_samples)。
        fs (float or int): 采样率（单位 Hz）。

    返回：
        None，直接绘制信号分析图并打印数值描述。
    """
    # 获取信号的基本信息
    bipolar_pairs = [
        'Fp2-F4', 'F4-C4', 'C4-P4',
        'P4-O2', 'Fp1-F3', 'F3-C3',
        'C3-P3', 'P3-O1', 'Fp2-F8',
        'F8-T4', 'T4-T6', 'T6-O2',
        'Fp1-F7', 'F7-T3', 'T3-T5',
        'T5-O1', 'Fz-Cz', 'Cz-Pz'
    ]
    n_channels, n_samples = eeg_data.shape
    time = np.linspace(0, n_samples / fs, n_samples)

    # --------- 时域分析 ---------
    print("===== Time Domain Analysis =====")
    # 1. 时域可视化：原始信号叠加图
    plt.figure(figsize=(12, 12))
    offset = 0.5  # 每个通道的偏移量
    for i in range(n_channels):
        plt.plot(time, eeg_data[i] + i * offset, label=bipolar_pairs[i])
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude (μV)")
    plt.title("Raw EEG Signal (Time Domain Overlay)")
    plt.legend(loc="upper right", bbox_to_anchor=(1.15, 1))
    plt.show()

    # 2. 数值描述：每个通道的基本统计量
    time_domain_stats = {}
    for i in range(n_channels):
        signal = eeg_data[i]
        mean_val = np.mean(signal)  # 均值
        std_val = np.std(signal)  # 标准差
        max_val = np.max(signal)  # 最大值
        min_val = np.min(signal)  # 最小值
        median_val = np.median(signal)  # 中位数
        signal_range = max_val - min_val  # 幅值范围
        time_domain_stats[bipolar_pairs[i]] = {
            "Mean": mean_val,
            "Std": std_val,
            "Max": max_val,
            "Min": min_val,
            "Median": median_val,
            "Range": signal_range,
        }
        print(f"{bipolar_pairs[i]}: Mean={mean_val:.2f}, Std={std_val:.2f}, "
              f"Max={max_val:.2f}, Min={min_val:.2f}, Median={median_val:.2f}, "
              f"Range={signal_range:.2f}")

    # --------- 频域分析 ---------
    print("\n===== Frequency Domain Analysis =====")
    # 3. 功率谱密度 (PSD) 图
    plt.figure(figsize=(12, 6))
    n_per_seg = int(2 * fs)  # 使用 2 秒的窗口大小
    for i in range(n_channels):
        f, psd = welch(eeg_data[i], fs=fs, nperseg=n_per_seg)
        plt.semilogy(f, psd, label=bipolar_pairs[i])
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("PSD (μV²/Hz)")
    plt.title("Power Spectral Density (PSD)")
    plt.legend(loc="upper right", bbox_to_anchor=(1.15, 1))
    plt.show()

    # 4. 数值描述：频段功率计算
    bands = {
        "Delta (0.5-4 Hz)": (0.5, 4),
        "Theta (4-8 Hz)": (4, 8),
        "Alpha (8-12 Hz)": (8, 12),
        "Beta (12-30 Hz)": (12, 30),
        "Gamma (30-50 Hz)": (30, 50),
    }

    # 初始化存储变量
    frequency_stats = {}  # 存储每个通道的频率统计信息
    band_power = {band_name: [] for band_name in bands.keys()}  # 存储每个频段功率

    # 遍历每个通道并计算频段功率统计
    for i in range(n_channels):
        f, psd = welch(eeg_data[i], fs=fs, nperseg=n_per_seg)  # 计算功率谱密度
        total_power = np.sum(psd)  # 总功率
        band_ratio = {}

        for band_name, (fmin, fmax) in bands.items():
            idx_band = (f >= fmin) & (f <= fmax)  # 筛选指定频段
            band_power_value = np.sum(psd[idx_band])  # 计算频段功率
            band_power[band_name].append(band_power_value)  # 存储到对应频段
            band_ratio[band_name] = band_power_value / total_power * 100  # 计算频段功率占比

        # 存储当前通道的频率统计信息
        frequency_stats[bipolar_pairs[i]] = {"Total Power": total_power, **band_ratio}
        print(f"{bipolar_pairs[i]}: Total Power={total_power:.2f}, Band Ratios={band_ratio}")

        # 绘制堆叠条形图
    x = np.arange(n_channels) + 1  # 每个通道的索引
    bottom = np.zeros(n_channels)  # 初始化堆叠“基线”
    plt.figure(figsize=(12, 6))
    for band_name, power in band_power.items():
        plt.bar(x, power, bottom=bottom, label=band_name)
        bottom += power  # 更新堆叠基线
    plt.xlabel("Channel")
    plt.ylabel("Power (μV²)")
    plt.title("Frequency Band Power per Channel")
    plt.legend()
    plt.show()

    # 计算所有通道的频段平均功率占比
    avg_band_ratios = {band_name: np.mean(
        [frequency_stats[bipolar_pairs[i]][band_name] for i in range(n_channels)]
    ) for band_name in bands.keys()}
    print("\nAverage Band Ratios Across Channels:")
    for band_name, avg_ratio in avg_band_ratios.items():
        print(f"{band_name}: {avg_ratio:.2f}%")

    # --------- 信号强度分析 ---------
    print("\n===== Signal Intensity Analysis =====")
    # 5. RMS 振幅分析
    rms = np.sqrt(np.mean(eeg_data ** 2, axis=0))  # 瞬时 RMS
    plt.figure(figsize=(12, 4))
    plt.plot(time, rms, label="RMS Amplitude")
    plt.xlabel("Time (s)")
    plt.ylabel("RMS (μV)")
    plt.title("RMS Amplitude Across Time")
    plt.legend()
    plt.show()

    # 6. 数值描述：每通道 RMS 均值
    rms_per_channel = np.sqrt(np.mean(eeg_data ** 2, axis=1))  # 每行 RMS
    for i, rms_avg in enumerate(rms_per_channel):
        print(f"{bipolar_pairs[i]}: RMS Average={rms_avg:.2f} μV")

    print(f"Overall RMS Mean Across Channels: {np.mean(rms_per_channel):.2f} μV")
    print(f"Overall RMS Std Across Channels: {np.std(rms_per_channel):.2f} μV")

    # --------- 统计分析 ---------
    print("\n===== Statistical Analysis =====")
    # 7. 协方差矩阵热图
    covariance = np.cov(eeg_data)
    plt.figure(figsize=(12, 10))
    plt.imshow(covariance, cmap="viridis", aspect="auto")
    plt.colorbar(label="Covariance")
    plt.title("Channel Covariance Heatmap")
    plt.xlabel("Channels")
    plt.ylabel("Channels")
    # plt.xticks(range(n_channels), [f"{bipolar_pairs[i]}" for i in range(n_channels)])
    plt.yticks(range(n_channels), [f"{bipolar_pairs[i]}" for i in range(n_channels)])
    plt.show()

    # 8. 数值描述：协方差整体特性
    mean_cov = np.mean(covariance)
    max_cov = np.max(covariance)
    min_cov = np.min(covariance)

    print(f"Mean Covariance: {mean_cov:.2f}")
    print(f"Max Covariance: {max_cov:.2f}")
    print(f"Min Covariance: {min_cov:.2f}")

    # 每个通道的平均协方差
    mean_cov_per_channel = np.mean(covariance, axis=1)
    for i, mean_cov_ch in enumerate(mean_cov_per_channel):
        print(f"{bipolar_pairs[i]}: Mean Covariance={mean_cov_ch:.2f}")

    return time_domain_stats, frequency_stats, rms_per_channel, np.mean(rms_per_channel), np.std(
        rms_per_channel), mean_cov_per_channel, mean_cov, max_cov, min_cov

# # 生成测试数据 (18 个通道，每秒 256 个采样点，持续 10 秒)
# np.random.seed(42)
# n_channels, fs, duration = 18, 256, 10
# n_samples = fs * duration
# eeg_data = np.random.randn(n_channels, n_samples) * 50  # 模拟数据，单位 μV
#
# # 调用可视化函数
# visualize_eeg(eeg_data, fs)
