import os
import h5py
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

WRK_DIR = '../working/'
INPUT_DIR = '../working/inputs/'
# Number of EEG signals in the HDF5 file
NO_OF_EEG_CHANELS = 18
# Column number in the HDF5 file where class indicator is written
SEIZURE_INDICATOR = 20


class CustomDataset(Dataset):
    def __init__(self, x, y, device=None):
        self.x = x
        self.y = y
        self.device = device

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx].to(self.device), self.y[idx].to(self.device)

def get_data_loaders(x_train, y_train, x_test, y_test, batch_size=16, device=None):

    # 创建训练集和测试集的自定义数据集对象
    train_dataset = CustomDataset(x_train, y_train, device)
    test_dataset = CustomDataset(x_test, y_test, device)

    # 创建训练集和测试集的数据加载器
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


def read_data(freq, which_expert, window, chunks, device, print_to_file=False):
    # freq - base frequency in .hdf5 file
    # which_expert - "A", "B" or "C"
    # window - sliding-window size
    # chunks - number of contiguous chunks

    freq = freq
    which_expert = which_expert
    window = window
    chunks = chunks

    data_name = "expert_{}_{}sec_{}chunk_{}Hz.hdf5".format(which_expert, window, chunks, freq)
    data_name_no_ext = os.path.splitext(data_name)[0]
    results_str = WRK_DIR + "results/results_" + data_name_no_ext + ".txt"

    if print_to_file:
        f = open(results_str, "a")
        print(data_name)
        print("########################################################", file=f)
        print("#", data_name, file=f)
        print("########################################################", file=f)

    ds = 1  # down sampling factor  (must be divisible by "freq", that is, 1,2,4,8 etc. In practice no more than 4, 1 - no down-sampling)
    # While preparing our hdf5 files in R, they have already gone through the down sampling procedure (256Hz --> 64Hz).
    # Therefore, we set ds = 1 here.
    size = int(freq * window / ds)

    file_h5 = h5py.File(INPUT_DIR + data_name, "r")
    list(file_h5.keys())
    temp = file_h5["FINAL.mtx"]
    temp = np.array(temp)
    # Transposition is needed here. In hdf5 file rows and columns are swapped
    dd = np.transpose(temp)
    print(f"dd.shape: {dd.shape}")

    ddss = dd[::ds]
    print(f"ddss.shape: {ddss.shape}")
    dr = ddss.reshape(int(ddss.shape[0] / size), size, dd.shape[1])
    print(f"dr.shape: {dr.shape}")

    # See 'Global variables' block
    x = torch.tensor(dr[:, :, 0:NO_OF_EEG_CHANELS])
    print(f"x.shape: {x.shape}")
    # '-1' as in Python indexing starts from 0
    y = torch.tensor(dr[:, 0, SEIZURE_INDICATOR - 1])
    print(f"y.shape: {y.shape}")

    # Normalization
    x = F.normalize(x, dim=1)

    if print_to_file:
        # print("dd:", dd.shape, file = f)
        print("Input matrix dims:", ddss.shape, file=f)
        print("x:", x.shape, file=f)
        print("y:", y.shape, file=f)
        f.close()

    return x, y, data_name

def preprocess_data(x, y, folds, fold_number, batch_size, device=None):
    # Add a dimension for Conv2D
    x_train_cv = x[folds[fold_number][0]]
    y_train_cv = y[folds[fold_number][0]]
    x_test_cv = x[folds[fold_number][1]]
    y_test_cv = y[folds[fold_number][1]]

    my_shape = np.expand_dims(x, axis=3).shape

    #####################################################################
    # add dimmension for conv2d
    #####################################################################
    x_train_3d = x_train_cv
    x_train_3d = x_train_3d.unsqueeze(dim=-1)
    x_train_3d = x_train_3d.permute(0, 3, 1, 2)

    x_test_3d = x_test_cv
    x_test_3d = x_test_3d.unsqueeze(dim=-1)
    x_test_3d = x_test_3d.permute(0, 3, 1, 2)

    train_loader, test_loader = get_data_loaders(x_train_3d, y_train_cv, x_test_3d, y_test_cv, batch_size, device)

    return train_loader, test_loader
