import os
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import random_split
import pandas as pd

# 截取的数据长度
CLIP_SIZE = 60
# 子载波数
SUBCARRIES = 64
LABELS = ['empty', 'fall', "walk"]
CUT_LEN = True
MIX = True # 是否混合多日数据
PRINTSHORT = False # 是否打印数据长度小于clip_size的样本
SHORTDELETE = False # 是否删除数据长度小于clip_size的样本

class CSIDataset(Dataset):
    def __init__(self, data_path):
        self.data_path = data_path
        self.data_files, self.labels = self.read_folder(data_path)

        csi_data = self.read_files(self.data_files)

        if CUT_LEN == True:
            csi_data = self.cut_data_len(csi_data)

        self.amplitudes = self.compute_amplitude(csi_data)

        # 归一化
        self.amplitudes = [(amplitude - np.mean(amplitude)) /
                           np.std(amplitude) for amplitude in self.amplitudes]

    def read_folder(self, dir_path):
        """
        返回文件夹中所有.csv文件的路径
        :param dir_path: 文件夹路径
        :return: csv_file_paths 一个列表,列表中的每个元素是一个数据文件的路径
        """
        data_files = []
        files_labels = []
        if MIX:
            subfiles = os.listdir(dir_path)
            for subfile in subfiles:
                if subfile.startswith('.'):
                    continue
                subfile_path = os.path.join(dir_path, subfile)
                for label in LABELS:
                    label_number = LABELS.index(label)
                    class_data_path = os.path.join(subfile_path, label)
                    # print(class_data_path)
                    files = os.listdir(class_data_path)
                    for file in files:
                        if file.endswith('.csv') or file.endswith('.txt'):
                            data_files.append(os.path.join(class_data_path, file))
                            files_labels.append(label_number)
            return data_files, files_labels
        else:
            for label in LABELS:
                label_number = LABELS.index(label)
                class_data_path = os.path.join(dir_path, label)
                files = os.listdir(class_data_path)
                for file in files:
                    if file.endswith('.csv'):
                        data_files.append(os.path.join(class_data_path, file))
                        files_labels.append(label_number)
            return data_files, files_labels

    def read_files(self, data_paths):
        """
        读取所有.csv文件中的CSI数据，由于存在文件第一行数据不完整，将不完整文件处理
        :param data_paths: 一个列表,列表中的每个元素是一个数据文件的路径
        :return: csi_data 一个列表,列表中的每个元素是一个数据文件的CSI数据
        """
        # cnt = 0
        csi_data = []
        for data_path in data_paths:
            # print(data_path)
            data = pd.read_csv(data_path, header=None)
            # cnt += 1
            # if cnt%10 == 1:
            #     print(data)
            first_row = data.iloc[0]
            if first_row[0] == "CSI_DATA":
                csidata = np.array([np.array(eval(csi))
                                   for csi in data.iloc[:, -1].values])
            else:
                data = pd.read_csv(data_path, header=None, skiprows=1)
                csidata = np.array([np.array(eval(csi))
                                   for csi in data.iloc[:, -1].values])
            if len(csidata) < CLIP_SIZE and PRINTSHORT:
                print(data_path)
                if SHORTDELETE:
                    os.remove(data_path)
            csi_data.append(csidata)
        return csi_data

    def cut_data_len(self, data_list, clip_size=CLIP_SIZE):
        """
        截取每一个数据的长度为clip_size，从数据的中间截取，如果数据长度小于clip_size，则丢弃
        :param data_list: 一个列表,列表中的每个元素是一个数据文件的CSI数据
        :param clip_size: 截取的长度
        """
        new_data_list = []
        for data in data_list:
            if len(data) >= clip_size:
                start = len(data) // 2 - clip_size // 2
                new_data_list.append(data[start: start + clip_size])
        return new_data_list

    def sum_data(self, data_list):
        """
        将数据列表中的数据合并
        :param data_list: 一个列表,列表中的每个元素是一个数据文件的CSI数据，shape: (clip_size, tx, rx, subcarries)
        :return: 合并后的数据, shape: (clip_size, subcarries)，原先的数据可以分出 tx * rx 个子数据，将子数据数值相加

        data_list = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
        print(np.sum(data_list, axis=(1, 2)))  # [21 57]
        """
        new_data_list = []
        for data in data_list:
            new_data_list.append(np.sum(data, axis=(1, 2)))
        return new_data_list

    def to_complex(self, arr):
        return np.vectorize(complex)(arr[:, 0::2], arr[:, 1::2])

    def compute_amplitude(self, csis_data):
        """
        使用NumPy的向量化操作计算CSI数据的幅度
        :param csis_data: 一个列表,列表中的每个元素是一个数据文件的CSI数据, 包含实数和虚数 shape: (clip_size, 2*subcarries)
        :return: amplitude_data 一个列表，列表中的一个元素是对应的振幅数据, shape: (clip_size, subcarries) (100, 64)
        """
        csi_data = []
        for csi in csis_data:
            csi_data.append(np.vstack(self.to_complex(csi)))

        amplitude_data = [np.abs(csi) for csi in csi_data]  # 计算CSI数据的幅度，即取绝对值
        amplitude_data = np.array(amplitude_data)
        print("shape", amplitude_data.shape)
        return amplitude_data

    def __len__(self):
        return len(self.amplitudes)

    def __getitem__(self, idx):
        return self.amplitudes[idx], self.labels[idx]


def process_single_csv_file(csv_path):
    """
    处理单个.csv文件，截取数据的中间部分，计算振幅数据，并进行标准化
    用于模型训练和验证
    :param csv_path: .csv文件路径
    :return: amplitude_data 振幅数据，shape: (clip_size, subcarries)
    """
    data = pd.read_csv(csv_path, header=None)
    first_row = data.iloc[0]
    if first_row[0] == "CSI_DATA":
        csidata = np.array([np.array(eval(csi))
                           for csi in data.iloc[:, -1].values])
    else:
        data = pd.read_csv(csv_path, header=None, skiprows=1)
        csidata = np.array([np.array(eval(csi))
                           for csi in data.iloc[:, -1].values])

    if len(csidata) < CLIP_SIZE:
        raise ValueError('The length of data is less than CLIP_SIZE')
    start = len(csidata) // 2 - CLIP_SIZE // 2
    csi_np = csidata[start: start + CLIP_SIZE]
    csi_np = np.vectorize(complex)(csi_np[:, 0::2], csi_np[:, 1::2])
    amplitude_data = np.abs(csi_np)
    amplitude_data = (amplitude_data - np.mean(amplitude_data)
                       ) / np.std(amplitude_data)
    return amplitude_data


if __name__ == '__main__':
    BATCH_SIZE = 2
    csi_dataset = CSIDataset('./data/train')
    total_size = len(csi_dataset)
    train_size = int(total_size * 0.8)
    val_size = total_size - train_size
    train_dataset, val_dataset = random_split(
        csi_dataset, [train_size, val_size])  # 分割数据集
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # 统计训练集和验证集的每种类别的数量
    class_num = [0, 0, 0]
    for data, label in train_loader:
        for l in label:
            class_num[l] += 1

    print('Train dataset:')
    print('Empty:', class_num[0])
    print('Fall:', class_num[1])
    print('Walk:', class_num[2])

    class_num = [0, 0, 0]
    for data, label in val_loader:
        for l in label:
            class_num[l] += 1

    print('Validation dataset:')
    print('Empty:', class_num[0])
    print('Fall:', class_num[1])
    print('Walk:', class_num[2])
