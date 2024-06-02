import os
import numpy as np
import pandas as pd
import glob
import re
import torch

## 新加入
import matplotlib.pyplot as plt
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
# from sktime.utils import load_data
import warnings

warnings.filterwarnings('ignore')

class PSMSegLoader(Dataset):
    def __init__(self, root_path, win_size, step=1, picture=True, flag="train"):
        self.picture = picture
        self.flag = flag
        self.step = win_size
        self.win_size = win_size
        self.scaler = StandardScaler()
        data = pd.read_csv(os.path.join(root_path, 'train.csv'))
        data = data.values[:, 1:]
        data = np.nan_to_num(data)
        self.scaler.fit(data)
        data = self.scaler.transform(data)
        test_data = pd.read_csv(os.path.join(root_path, 'test.csv'))
        test_data = test_data.values[:, 1:]
        test_data = np.nan_to_num(test_data)
        self.test = self.scaler.transform(test_data)
        self.train = data
        self.val = self.test
        self.test_labels = pd.read_csv(os.path.join(root_path, 'test_label.csv')).values[:, 1:]
        print("test:", self.test.shape)
        print("train:", self.train.shape)

    def __len__(self):
        if self.flag == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.flag == "train":
            sample_raw = np.float32(self.train[index:index + self.win_size])
            ## 画图
            total_path = r''
            if not os.path.exists(total_path):
                os.makedirs(total_path)

            picture_path = os.path.join(total_path, f"{index}")
            if not os.path.exists(picture_path):
                os.makedirs(picture_path)

            # 创建图像(只能在完整训练一个epoch才可以省略，因为index是随机的，注意batch丢失问题）
            if self.picture:
                for i in range(sample_raw.shape[1]):
                    time = [j for j in range(sample_raw.shape[0])]
                    line = sample_raw[:, i]
                    plt.figure(figsize=(2.24, 2.24))
                    plt.plot(time, line)
                    filename_png = f"sample_index_{index}_dimension_{i}.png"
                    plt.savefig(os.path.join(picture_path, filename_png))
                    plt.close()

                # 加载图像，并将其拼接为，[特征，通道，长，宽]
            sample_pucture = []
            for i in range(sample_raw.shape[1]):
                filename_png = f"sample_index_{index}_dimension_{i}.png"
                img_path = os.path.join(picture_path, filename_png)
                img = Image.open(img_path).convert('RGB')
                #                img = img.resize((224, 224))
                picture = np.array(img).transpose(2, 0, 1)
                img.close()
                sample_pucture.append(picture)
            sample_pucture = np.stack(sample_pucture, axis=0)
            return np.float32(self.train[index:index + self.win_size]), \
                   np.float32(self.test_labels[0:self.win_size]), sample_pucture

        elif (self.flag == 'val'):
            sample_raw = np.float32(self.val[index:index + self.win_size])
            ## 画图
            total_path = r''
            if not os.path.exists(total_path):
                os.makedirs(total_path)

            picture_path = os.path.join(total_path, f"{index}")
            if not os.path.exists(picture_path):
                os.makedirs(picture_path)

            # 创建图像(只能在完整训练一个epoch才可以省略，因为index是随机的，注意batch丢失问题）
            if self.picture:
                for i in range(sample_raw.shape[1]):
                    time = [j for j in range(sample_raw.shape[0])]
                    line = sample_raw[:, i]
                    plt.figure(figsize=(2.24, 2.24))
                    plt.plot(time, line)
                    filename_png = f"sample_index_{index}_dimension_{i}.png"
                    plt.savefig(os.path.join(picture_path, filename_png))
                    plt.close()

                # 加载图像，并将其拼接为，[特征，通道，长，宽]
            sample_pucture = []
            for i in range(sample_raw.shape[1]):
                filename_png = f"sample_index_{index}_dimension_{i}.png"
                img_path = os.path.join(picture_path, filename_png)
                img = Image.open(img_path).convert('RGB')
                #                img = img.resize((224, 224))
                picture = np.array(img).transpose(2, 0, 1)
                img.close()
                sample_pucture.append(picture)
            sample_pucture = np.stack(sample_pucture, axis=0)
            return np.float32(self.val[index:index + self.win_size]), \
                   np.float32(self.test_labels[0:self.win_size]), sample_pucture

        elif (self.flag == 'test'):
            sample_raw = np.float32(self.test[index:index + self.win_size])
            # mean_values = np.mean(sample_raw, axis=0)
            # std_values = np.std(sample_raw, axis=0)
            # sample_raw = (sample_raw-mean_values)/(std_values + 1e-5)
            # sample_raw = (sample_raw - mean_values)

            ## 画图
            total_path = r''
            if not os.path.exists(total_path):
                os.makedirs(total_path)

            picture_path = os.path.join(total_path, f"{index}")
            if not os.path.exists(picture_path):
                os.makedirs(picture_path)

            # 创建图像(只能在完整训练一个epoch才可以省略，因为index是随机的，注意batch丢失问题）
            if self.picture:
                for i in range(sample_raw.shape[1]):
                    time = [j for j in range(sample_raw.shape[0])]
                    line = sample_raw[:, i]
                    plt.figure(figsize=(2.24, 2.24))
                    plt.plot(time, line)
                    filename_png = f"sample_index_{index}_dimension_{i}.png"
                    plt.savefig(os.path.join(picture_path, filename_png))
                    plt.close()

                # 加载图像，并将其拼接为，[特征，通道，长，宽]
            sample_pucture = []
            for i in range(sample_raw.shape[1]):
                filename_png = f"sample_index_{index}_dimension_{i}.png"
                img_path = os.path.join(picture_path, filename_png)
                img = Image.open(img_path).convert('RGB')
                #                img = img.resize((224, 224))
                picture = np.array(img).transpose(2, 0, 1)
                img.close()
                sample_pucture.append(picture)
            sample_pucture = np.stack(sample_pucture, axis=0)
            return np.float32(self.test[index:index + self.win_size]), \
                   np.float32(self.test_labels[index:index + self.win_size]), sample_pucture

        else:
            return np.float32(self.test[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])


class MSLSegLoader(Dataset):
    def __init__(self, root_path, win_size, step=1, picture=True, flag="train"):
        self.picture = picture
        self.flag = flag
        self.step = win_size
        self.win_size = win_size
        self.scaler = StandardScaler()
        data = np.load(os.path.join(root_path, "MSL_train.npy"))
        self.scaler.fit(data)
        data = self.scaler.transform(data)
        test_data = np.load(os.path.join(root_path, "MSL_test.npy"))
        self.test = self.scaler.transform(test_data)
        self.train = data
        self.val = self.test
        self.test_labels = np.load(os.path.join(root_path, "MSL_test_label.npy"))
        print("test:", self.test.shape)
        print("train:", self.train.shape)


    def __len__(self):
        if self.flag == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.flag == "train":
            sample_raw = np.float32(self.train[index:index + self.win_size])
            ## 画图
            total_path = r''
            if not os.path.exists(total_path):
                os.makedirs(total_path)

            picture_path = os.path.join(total_path, f"{index}")
            if not os.path.exists(picture_path):
                os.makedirs(picture_path)

            # 创建图像(只能在完整训练一个epoch才可以省略，因为index是随机的，注意batch丢失问题）
            if self.picture:
                for i in range(sample_raw.shape[1]):
                    time = [j for j in range(sample_raw.shape[0])]
                    line = sample_raw[:, i]
                    plt.figure(figsize=(2.24, 2.24))
                    plt.plot(time, line)
                    filename_png = f"sample_index_{index}_dimension_{i}.png"
                    plt.savefig(os.path.join(picture_path, filename_png))
                    plt.close()

                # 加载图像，并将其拼接为，[特征，通道，长，宽]
            sample_pucture = []
            for i in range(sample_raw.shape[1]):
                filename_png = f"sample_index_{index}_dimension_{i}.png"
                img_path = os.path.join(picture_path, filename_png)
                img = Image.open(img_path).convert('RGB')
                #                img = img.resize((224, 224))
                picture = np.array(img).transpose(2, 0, 1)
                img.close()
                sample_pucture.append(picture)
            sample_pucture = np.stack(sample_pucture, axis=0)
            return np.float32(self.train[index:index + self.win_size]), \
                   np.float32(self.test_labels[0:self.win_size]),sample_pucture

        elif (self.flag == 'val'):
            sample_raw = np.float32(self.val[index:index + self.win_size])
            ## 画图
            total_path = r''
            if not os.path.exists(total_path):
                os.makedirs(total_path)

            picture_path = os.path.join(total_path, f"{index}")
            if not os.path.exists(picture_path):
                os.makedirs(picture_path)

            # 创建图像(只能在完整训练一个epoch才可以省略，因为index是随机的，注意batch丢失问题）
            if self.picture:
                for i in range(sample_raw.shape[1]):
                    time = [j for j in range(sample_raw.shape[0])]
                    line = sample_raw[:, i]
                    plt.figure(figsize=(2.24, 2.24))
                    plt.plot(time, line)
                    filename_png = f"sample_index_{index}_dimension_{i}.png"
                    plt.savefig(os.path.join(picture_path, filename_png))
                    plt.close()

                # 加载图像，并将其拼接为，[特征，通道，长，宽]
            sample_pucture = []
            for i in range(sample_raw.shape[1]):
                filename_png = f"sample_index_{index}_dimension_{i}.png"
                img_path = os.path.join(picture_path, filename_png)
                img = Image.open(img_path).convert('RGB')
                #                img = img.resize((224, 224))
                picture = np.array(img).transpose(2, 0, 1)
                img.close()
                sample_pucture.append(picture)
            sample_pucture = np.stack(sample_pucture, axis=0)
            return np.float32(self.val[index:index + self.win_size]), \
                   np.float32(self.test_labels[0:self.win_size]),sample_pucture

        elif (self.flag == 'test'):
            sample_raw = np.float32(self.test[index:index + self.win_size])
            ## 画图
            total_path = r''
            if not os.path.exists(total_path):
                os.makedirs(total_path)

            picture_path = os.path.join(total_path, f"{index}")
            if not os.path.exists(picture_path):
                os.makedirs(picture_path)

            # 创建图像(只能在完整训练一个epoch才可以省略，因为index是随机的，注意batch丢失问题）
            if self.picture:
                for i in range(sample_raw.shape[1]):
                    time = [j for j in range(sample_raw.shape[0])]
                    line = sample_raw[:, i]
                    plt.figure(figsize=(2.24, 2.24))
                    plt.plot(time, line)
                    filename_png = f"sample_index_{index}_dimension_{i}.png"
                    plt.savefig(os.path.join(picture_path, filename_png))
                    plt.close()

                # 加载图像，并将其拼接为，[特征，通道，长，宽]
            sample_pucture = []
            for i in range(sample_raw.shape[1]):
                filename_png = f"sample_index_{index}_dimension_{i}.png"
                img_path = os.path.join(picture_path, filename_png)
                img = Image.open(img_path).convert('RGB')
                #                img = img.resize((224, 224))
                picture = np.array(img).transpose(2, 0, 1)
                img.close()
                sample_pucture.append(picture)

            sample_pucture = np.stack(sample_pucture, axis=0)
            return np.float32(self.test[index:index + self.win_size]), \
                   np.float32(self.test_labels[index:index + self.win_size]), sample_pucture

        else:
            return np.float32(self.test[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])


class SMAPSegLoader(Dataset):
    def __init__(self, root_path, win_size, step=1, picture=True, flag="train"):
        self.picture = picture
        self.flag = flag
        self.step = win_size
        self.win_size = win_size
        self.scaler = StandardScaler()
        data = np.load(os.path.join(root_path, "SMAP_train.npy"))
        self.scaler.fit(data)
        data = self.scaler.transform(data)
        test_data = np.load(os.path.join(root_path, "SMAP_test.npy"))
        self.test = self.scaler.transform(test_data)
        self.train = data
        self.val = self.test
        self.test_labels = np.load(os.path.join(root_path, "SMAP_test_label.npy"))
        print("test:", self.test.shape)
        print("train:", self.train.shape)

    def __len__(self):

        if self.flag == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.flag == "train":
            sample_raw = np.float32(self.train[index:index + self.win_size])
            ## 画图
            total_path = r''
            if not os.path.exists(total_path):
                os.makedirs(total_path)

            picture_path = os.path.join(total_path, f"{index}")
            if not os.path.exists(picture_path):
                os.makedirs(picture_path)

            # 创建图像(只能在完整训练一个epoch才可以省略，因为index是随机的，注意batch丢失问题）
            if self.picture:
                for i in range(sample_raw.shape[1]):
                    time = [j for j in range(sample_raw.shape[0])]
                    line = sample_raw[:, i]
                    plt.figure(figsize=(2.24, 2.24))
                    plt.plot(time, line)
                    filename_png = f"sample_index_{index}_dimension_{i}.png"
                    plt.savefig(os.path.join(picture_path, filename_png))
                    plt.close()

                # 加载图像，并将其拼接为，[特征，通道，长，宽]
            sample_pucture = []
            for i in range(sample_raw.shape[1]):
                filename_png = f"sample_index_{index}_dimension_{i}.png"
                img_path = os.path.join(picture_path, filename_png)
                img = Image.open(img_path).convert('RGB')
                #                img = img.resize((224, 224))
                picture = np.array(img).transpose(2, 0, 1)
                img.close()
                sample_pucture.append(picture)
            sample_pucture = np.stack(sample_pucture, axis=0)
            return np.float32(self.train[index:index + self.win_size]), \
                   np.float32(self.test_labels[0:self.win_size]), sample_pucture

        elif (self.flag == 'val'):
            sample_raw = np.float32(self.val[index:index + self.win_size])
            ## 画图
            total_path = r''
            if not os.path.exists(total_path):
                os.makedirs(total_path)

            picture_path = os.path.join(total_path, f"{index}")
            if not os.path.exists(picture_path):
                os.makedirs(picture_path)

            # 创建图像(只能在完整训练一个epoch才可以省略，因为index是随机的，注意batch丢失问题）
            if self.picture:
                for i in range(sample_raw.shape[1]):
                    time = [j for j in range(sample_raw.shape[0])]
                    line = sample_raw[:, i]
                    plt.figure(figsize=(2.24, 2.24))
                    plt.plot(time, line)
                    filename_png = f"sample_index_{index}_dimension_{i}.png"
                    plt.savefig(os.path.join(picture_path, filename_png))
                    plt.close()

                # 加载图像，并将其拼接为，[特征，通道，长，宽]
            sample_pucture = []
            for i in range(sample_raw.shape[1]):
                filename_png = f"sample_index_{index}_dimension_{i}.png"
                img_path = os.path.join(picture_path, filename_png)
                img = Image.open(img_path).convert('RGB')
                #                img = img.resize((224, 224))
                picture = np.array(img).transpose(2, 0, 1)
                img.close()
                sample_pucture.append(picture)
            sample_pucture = np.stack(sample_pucture, axis=0)
            return np.float32(self.val[index:index + self.win_size]), \
                   np.float32(self.test_labels[0:self.win_size]), sample_pucture

        elif (self.flag == 'test'):
            sample_raw = np.float32(self.test[index:index + self.win_size])
            ## 画图
            total_path = r''
            if not os.path.exists(total_path):
                os.makedirs(total_path)

            picture_path = os.path.join(total_path, f"{index}")
            if not os.path.exists(picture_path):
                os.makedirs(picture_path)

            # 创建图像(只能在完整训练一个epoch才可以省略，因为index是随机的，注意batch丢失问题）
            if self.picture:
                for i in range(sample_raw.shape[1]):
                    time = [j for j in range(sample_raw.shape[0])]
                    line = sample_raw[:, i]
                    plt.figure(figsize=(2.24, 2.24))
                    plt.plot(time, line)
                    filename_png = f"sample_index_{index}_dimension_{i}.png"
                    plt.savefig(os.path.join(picture_path, filename_png))
                    plt.close()

                # 加载图像，并将其拼接为，[特征，通道，长，宽]
            sample_pucture = []
            for i in range(sample_raw.shape[1]):
                filename_png = f"sample_index_{index}_dimension_{i}.png"
                img_path = os.path.join(picture_path, filename_png)
                img = Image.open(img_path).convert('RGB')
                #                img = img.resize((224, 224))
                picture = np.array(img).transpose(2, 0, 1)
                img.close()
                sample_pucture.append(picture)
            sample_pucture = np.stack(sample_pucture, axis=0)
            return np.float32(self.test[index:index + self.win_size]), \
                   np.float32(self.test_labels[index:index + self.win_size]), sample_pucture

        else:
            return np.float32(self.test[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[
                index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])


class SMDSegLoader(Dataset):
    def __init__(self, root_path, win_size, step=1, picture=True, flag="train"):
        self.picture = picture
        self.flag = flag
        self.step = win_size
        self.win_size = win_size
        self.scaler = StandardScaler()
        data = np.load(os.path.join(root_path, "SMD_train.npy"))
        self.scaler.fit(data)
        data = self.scaler.transform(data)
        test_data = np.load(os.path.join(root_path, "SMD_test.npy"))
        self.test = self.scaler.transform(test_data)
        self.train = data
        data_len = len(self.train)
        self.val = self.train[(int)(data_len * 0.8):]
        self.test_labels = np.load(os.path.join(root_path, "SMD_test_label.npy"))

    def __len__(self):
        if self.flag == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.flag == "train":
            sample_raw = np.float32(self.train[index:index + self.win_size])
            ## 画图
            total_path = r''
            if not os.path.exists(total_path):
                os.makedirs(total_path)

            picture_path = os.path.join(total_path, f"{index}")
            if not os.path.exists(picture_path):
                os.makedirs(picture_path)

            # 创建图像(只能在完整训练一个epoch才可以省略，因为index是随机的，注意batch丢失问题）
            if self.picture:
                for i in range(sample_raw.shape[1]):
                    time = [j for j in range(sample_raw.shape[0])]
                    line = sample_raw[:, i]
                    plt.figure(figsize=(2.24, 2.24))
                    plt.plot(time, line)
                    filename_png = f"sample_index_{index}_dimension_{i}.png"
                    plt.savefig(os.path.join(picture_path, filename_png))
                    plt.close()

                # 加载图像，并将其拼接为，[特征，通道，长，宽]
            sample_pucture = []
            for i in range(sample_raw.shape[1]):
                filename_png = f"sample_index_{index}_dimension_{i}.png"
                img_path = os.path.join(picture_path, filename_png)
                img = Image.open(img_path).convert('RGB')
                #                img = img.resize((224, 224))
                picture = np.array(img).transpose(2, 0, 1)
                img.close()
                sample_pucture.append(picture)
            sample_pucture = np.stack(sample_pucture, axis=0)
            return np.float32(self.train[index:index + self.win_size]), \
                   np.float32(self.test_labels[0:self.win_size]), sample_pucture

        elif (self.flag == 'val'):
            sample_raw = np.float32(self.val[index:index + self.win_size])
            ## 画图
            total_path = r''
            if not os.path.exists(total_path):
                os.makedirs(total_path)

            picture_path = os.path.join(total_path, f"{index}")
            if not os.path.exists(picture_path):
                os.makedirs(picture_path)

            # 创建图像(只能在完整训练一个epoch才可以省略，因为index是随机的，注意batch丢失问题）
            if self.picture:
                for i in range(sample_raw.shape[1]):
                    time = [j for j in range(sample_raw.shape[0])]
                    line = sample_raw[:, i]
                    plt.figure(figsize=(2.24, 2.24))
                    plt.plot(time, line)
                    filename_png = f"sample_index_{index}_dimension_{i}.png"
                    plt.savefig(os.path.join(picture_path, filename_png))
                    plt.close()

                # 加载图像，并将其拼接为，[特征，通道，长，宽]
            sample_pucture = []
            for i in range(sample_raw.shape[1]):
                filename_png = f"sample_index_{index}_dimension_{i}.png"
                img_path = os.path.join(picture_path, filename_png)
                img = Image.open(img_path).convert('RGB')
                #                img = img.resize((224, 224))
                picture = np.array(img).transpose(2, 0, 1)
                img.close()
                sample_pucture.append(picture)
            sample_pucture = np.stack(sample_pucture, axis=0)
            return np.float32(self.val[index:index + self.win_size]), \
                   np.float32(self.test_labels[0:self.win_size]), sample_pucture

        elif (self.flag == 'test'):
            sample_raw = np.float32(self.test[index:index + self.win_size])
            ## 画图
            total_path = r''
            if not os.path.exists(total_path):
                os.makedirs(total_path)

            picture_path = os.path.join(total_path, f"{index}")
            if not os.path.exists(picture_path):
                os.makedirs(picture_path)

            # 创建图像(只能在完整训练一个epoch才可以省略，因为index是随机的，注意batch丢失问题）
            if self.picture:
                for i in range(sample_raw.shape[1]):
                    time = [j for j in range(sample_raw.shape[0])]
                    line = sample_raw[:, i]
                    plt.figure(figsize=(2.24, 2.24))
                    plt.plot(time, line)
                    filename_png = f"sample_index_{index}_dimension_{i}.png"
                    plt.savefig(os.path.join(picture_path, filename_png))
                    plt.close()

                # 加载图像，并将其拼接为，[特征，通道，长，宽]
            sample_pucture = []
            for i in range(sample_raw.shape[1]):
                filename_png = f"sample_index_{index}_dimension_{i}.png"
                img_path = os.path.join(picture_path, filename_png)
                img = Image.open(img_path).convert('RGB')
                #                img = img.resize((224, 224))
                picture = np.array(img).transpose(2, 0, 1)
                img.close()
                sample_pucture.append(picture)
            sample_pucture = np.stack(sample_pucture, axis=0)
            return np.float32(self.test[index:index + self.win_size]), \
                   np.float32(self.test_labels[index:index + self.win_size]), sample_pucture

        else:
            return np.float32(self.test[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[
                index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])


class SWATSegLoader(Dataset):
    def __init__(self, root_path, win_size, step=1, picture=True, flag="train"):
        self.picture = picture
        self.flag = flag
        self.step = win_size
        self.win_size = win_size
        self.scaler = StandardScaler()
        data = pd.read_csv(os.path.join(root_path, 'train.csv'))
        data = data.values[:, 1:]
        data = np.nan_to_num(data)
        self.scaler.fit(data)
        data = self.scaler.transform(data)
        test_data = pd.read_csv(os.path.join(root_path, 'test.csv'))
        test_data = test_data.values[:, 1:]
        test_data = np.nan_to_num(test_data)
        self.test = self.scaler.transform(test_data)
        self.train = data
        self.val = self.test
        self.test_labels = pd.read_csv(os.path.join(root_path, 'test_label.csv')).values[:, 1:]
        print("test:", self.test.shape)
        print("train:", self.train.shape)

    def __len__(self):
        """
        Number of images in the object dataset.
        """
        if self.flag == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.flag == "train":
            sample_raw = np.float32(self.train[index:index + self.win_size])
            ## 画图
            total_path = r''
            if not os.path.exists(total_path):
                os.makedirs(total_path)

            picture_path = os.path.join(total_path, f"{index}")
            if not os.path.exists(picture_path):
                os.makedirs(picture_path)

            # 创建图像(只能在完整训练一个epoch才可以省略，因为index是随机的，注意batch丢失问题）
            if self.picture:
                for i in range(sample_raw.shape[1]):
                    time = [j for j in range(sample_raw.shape[0])]
                    line = sample_raw[:, i]
                    plt.figure(figsize=(2.24, 2.24))
                    plt.plot(time, line)
                    filename_png = f"sample_index_{index}_dimension_{i}.png"
                    plt.savefig(os.path.join(picture_path, filename_png))
                    plt.close()

                # 加载图像，并将其拼接为，[特征，通道，长，宽]
            sample_pucture = []
            for i in range(sample_raw.shape[1]):
                filename_png = f"sample_index_{index}_dimension_{i}.png"
                img_path = os.path.join(picture_path, filename_png)
                img = Image.open(img_path).convert('RGB')
                #                img = img.resize((224, 224))
                picture = np.array(img).transpose(2, 0, 1)
                img.close()
                sample_pucture.append(picture)
            sample_pucture = np.stack(sample_pucture, axis=0)
            return np.float32(self.train[index:index + self.win_size]), \
                   np.float32(self.test_labels[0:self.win_size]), sample_pucture

        elif (self.flag == 'val'):
            sample_raw = np.float32(self.val[index:index + self.win_size])
            ## 画图
            total_path = r''
            if not os.path.exists(total_path):
                os.makedirs(total_path)

            picture_path = os.path.join(total_path, f"{index}")
            if not os.path.exists(picture_path):
                os.makedirs(picture_path)

            # 创建图像(只能在完整训练一个epoch才可以省略，因为index是随机的，注意batch丢失问题）
            if self.picture:
                for i in range(sample_raw.shape[1]):
                    time = [j for j in range(sample_raw.shape[0])]
                    line = sample_raw[:, i]
                    plt.figure(figsize=(2.24, 2.24))
                    plt.plot(time, line)
                    filename_png = f"sample_index_{index}_dimension_{i}.png"
                    plt.savefig(os.path.join(picture_path, filename_png))
                    plt.close()

                # 加载图像，并将其拼接为，[特征，通道，长，宽]
            sample_pucture = []
            for i in range(sample_raw.shape[1]):
                filename_png = f"sample_index_{index}_dimension_{i}.png"
                img_path = os.path.join(picture_path, filename_png)
                img = Image.open(img_path).convert('RGB')
                #                img = img.resize((224, 224))
                picture = np.array(img).transpose(2, 0, 1)
                img.close()
                sample_pucture.append(picture)
            sample_pucture = np.stack(sample_pucture, axis=0)
            return np.float32(self.val[index:index + self.win_size]), \
                   np.float32(self.test_labels[0:self.win_size]), sample_pucture

        elif (self.flag == 'test'):
            sample_raw = np.float32(self.test[index:index + self.win_size])
            ## 画图
            total_path = r''
            if not os.path.exists(total_path):
                os.makedirs(total_path)

            picture_path = os.path.join(total_path, f"{index}")
            if not os.path.exists(picture_path):
                os.makedirs(picture_path)

            # 创建图像(只能在完整训练一个epoch才可以省略，因为index是随机的，注意batch丢失问题）
            if self.picture:
                for i in range(sample_raw.shape[1]):
                    time = [j for j in range(sample_raw.shape[0])]
                    line = sample_raw[:, i]
                    plt.figure(figsize=(2.24, 2.24))
                    plt.plot(time, line)
                    filename_png = f"sample_index_{index}_dimension_{i}.png"
                    plt.savefig(os.path.join(picture_path, filename_png))
                    plt.close()

                # 加载图像，并将其拼接为，[特征，通道，长，宽]
            sample_pucture = []
            for i in range(sample_raw.shape[1]):
                filename_png = f"sample_index_{index}_dimension_{i}.png"
                img_path = os.path.join(picture_path, filename_png)
                img = Image.open(img_path).convert('RGB')
                #                img = img.resize((224, 224))
                picture = np.array(img).transpose(2, 0, 1)
                img.close()
                sample_pucture.append(picture)
            sample_pucture = np.stack(sample_pucture, axis=0)
            return np.float32(self.test[index:index + self.win_size]), \
                   np.float32(self.test_labels[index:index + self.win_size]), sample_pucture

        else:
            return np.float32(self.test[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[
                index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])

class NIPS_TS_SwanSegLoader(object):
    def __init__(self, root_path, win_size, step=1, picture=True, flag="train"):
        self.flag = flag
        self.picture = picture
        data_path = root_path
        self.step = win_size
        self.win_size = win_size
        self.scaler = StandardScaler()
        data = np.load(data_path + "/NIPS_TS_Swan_train.npy")
        self.scaler.fit(data)
        data = self.scaler.transform(data)
        test_data = np.load(data_path + "/NIPS_TS_Swan_test.npy")
        self.test = self.scaler.transform(test_data)

        self.train = data
        self.val = self.test
        self.test_labels = np.load(data_path + "/NIPS_TS_Swan_test_label.npy")
        print("test:", self.test.shape)
        print("train:", self.train.shape)

    def __len__(self):
        """
        Number of images in the object dataset.
        """
        if self.flag == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.flag == "train":
            sample_raw = np.float32(self.train[index:index + self.win_size])
            ## 画图
            total_path = r''
            if not os.path.exists(total_path):
                os.makedirs(total_path)

            picture_path = os.path.join(total_path, f"{index}")
            if not os.path.exists(picture_path):
                os.makedirs(picture_path)

            # 创建图像(只能在完整训练一个epoch才可以省略，因为index是随机的，注意batch丢失问题）
            if self.picture:
                for i in range(sample_raw.shape[1]):
                    time = [j for j in range(sample_raw.shape[0])]
                    line = sample_raw[:, i]
                    plt.figure(figsize=(2.24, 2.24))
                    plt.plot(time, line)
                    filename_png = f"sample_index_{index}_dimension_{i}.png"
                    plt.savefig(os.path.join(picture_path, filename_png))
                    plt.close()

                # 加载图像，并将其拼接为，[特征，通道，长，宽]
            sample_pucture = []
            for i in range(sample_raw.shape[1]):
                filename_png = f"sample_index_{index}_dimension_{i}.png"
                img_path = os.path.join(picture_path, filename_png)
                img = Image.open(img_path).convert('RGB')
                #                img = img.resize((224, 224))
                picture = np.array(img).transpose(2, 0, 1)
                img.close()
                sample_pucture.append(picture)
            sample_pucture = np.stack(sample_pucture, axis=0)
            return np.float32(self.train[index:index + self.win_size]), \
                   np.float32(self.test_labels[0:self.win_size]), sample_pucture

        elif (self.flag == 'val'):
            sample_raw = np.float32(self.val[index:index + self.win_size])
            ## 画图
            total_path = r''
            if not os.path.exists(total_path):
                os.makedirs(total_path)

            picture_path = os.path.join(total_path, f"{index}")
            if not os.path.exists(picture_path):
                os.makedirs(picture_path)

            # 创建图像(只能在完整训练一个epoch才可以省略，因为index是随机的，注意batch丢失问题）
            if self.picture:
                for i in range(sample_raw.shape[1]):
                    time = [j for j in range(sample_raw.shape[0])]
                    line = sample_raw[:, i]
                    plt.figure(figsize=(2.24, 2.24))
                    plt.plot(time, line)
                    filename_png = f"sample_index_{index}_dimension_{i}.png"
                    plt.savefig(os.path.join(picture_path, filename_png))
                    plt.close()

                # 加载图像，并将其拼接为，[特征，通道，长，宽]
            sample_pucture = []
            for i in range(sample_raw.shape[1]):
                filename_png = f"sample_index_{index}_dimension_{i}.png"
                img_path = os.path.join(picture_path, filename_png)
                img = Image.open(img_path).convert('RGB')
                #                img = img.resize((224, 224))
                picture = np.array(img).transpose(2, 0, 1)
                img.close()
                sample_pucture.append(picture)
            sample_pucture = np.stack(sample_pucture, axis=0)
            return np.float32(self.val[index:index + self.win_size]), \
                   np.float32(self.test_labels[0:self.win_size]), sample_pucture

        elif (self.flag == 'test'):
            sample_raw = np.float32(self.test[index:index + self.win_size])
            ## 画图
            total_path = r''
            if not os.path.exists(total_path):
                os.makedirs(total_path)

            picture_path = os.path.join(total_path, f"{index}")
            if not os.path.exists(picture_path):
                os.makedirs(picture_path)

            # 创建图像(只能在完整训练一个epoch才可以省略，因为index是随机的，注意batch丢失问题）
            if self.picture:
                for i in range(sample_raw.shape[1]):
                    time = [j for j in range(sample_raw.shape[0])]
                    line = sample_raw[:, i]
                    plt.figure(figsize=(2.24, 2.24))
                    plt.plot(time, line)
                    filename_png = f"sample_index_{index}_dimension_{i}.png"
                    plt.savefig(os.path.join(picture_path, filename_png))
                    plt.close()

                # 加载图像，并将其拼接为，[特征，通道，长，宽]
            sample_pucture = []
            for i in range(sample_raw.shape[1]):
                filename_png = f"sample_index_{index}_dimension_{i}.png"
                img_path = os.path.join(picture_path, filename_png)
                img = Image.open(img_path).convert('RGB')
                #                img = img.resize((224, 224))
                picture = np.array(img).transpose(2, 0, 1)
                img.close()
                sample_pucture.append(picture)
            sample_pucture = np.stack(sample_pucture, axis=0)
            return np.float32(self.test[index:index + self.win_size]), \
                   np.float32(self.test_labels[index:index + self.win_size]), sample_pucture

        else:
            return np.float32(self.test[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[
                index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])

class NIPS_TS_WaterSegLoader(object):
    def __init__(self, root_path, win_size, step=1, picture=True, flag="train"):
        self.flag = flag
        self.picture = picture
        data_path = root_path
        self.step = win_size
        self.win_size = win_size
        self.scaler = StandardScaler()
        data = np.load(data_path + "/NIPS_TS_Water_train.npy")
        self.scaler.fit(data)
        data = self.scaler.transform(data)
        test_data = np.load(data_path + "/NIPS_TS_Water_test.npy")
        self.test = self.scaler.transform(test_data)

        self.train = data
        self.val = self.test
        self.test_labels = np.load(data_path + "/NIPS_TS_Water_test_label.npy")
        print("test:", self.test.shape)
        print("train:", self.train.shape)

    def __len__(self):
        """
        Number of images in the object dataset.
        """
        if self.flag == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.flag == "train":
            sample_raw = np.float32(self.train[index:index + self.win_size])
            ## 画图
            total_path = r''
            if not os.path.exists(total_path):
                os.makedirs(total_path)

            picture_path = os.path.join(total_path, f"{index}")
            if not os.path.exists(picture_path):
                os.makedirs(picture_path)

            # 创建图像(只能在完整训练一个epoch才可以省略，因为index是随机的，注意batch丢失问题）
            if self.picture:
                for i in range(sample_raw.shape[1]):
                    time = [j for j in range(sample_raw.shape[0])]
                    line = sample_raw[:, i]
                    plt.figure(figsize=(2.24, 2.24))
                    plt.plot(time, line)
                    filename_png = f"sample_index_{index}_dimension_{i}.png"
                    plt.savefig(os.path.join(picture_path, filename_png))
                    plt.close()

                # 加载图像，并将其拼接为，[特征，通道，长，宽]
            sample_pucture = []
            for i in range(sample_raw.shape[1]):
                filename_png = f"sample_index_{index}_dimension_{i}.png"
                img_path = os.path.join(picture_path, filename_png)
                img = Image.open(img_path).convert('RGB')
                #                img = img.resize((224, 224))
                picture = np.array(img).transpose(2, 0, 1)
                img.close()
                sample_pucture.append(picture)
            sample_pucture = np.stack(sample_pucture, axis=0)
            return np.float32(self.train[index:index + self.win_size]), \
                   np.float32(self.test_labels[0:self.win_size]), sample_pucture

        elif (self.flag == 'val'):
            sample_raw = np.float32(self.val[index:index + self.win_size])
            ## 画图
            total_path = r''
            if not os.path.exists(total_path):
                os.makedirs(total_path)

            picture_path = os.path.join(total_path, f"{index}")
            if not os.path.exists(picture_path):
                os.makedirs(picture_path)

            # 创建图像(只能在完整训练一个epoch才可以省略，因为index是随机的，注意batch丢失问题）
            if self.picture:
                for i in range(sample_raw.shape[1]):
                    time = [j for j in range(sample_raw.shape[0])]
                    line = sample_raw[:, i]
                    plt.figure(figsize=(2.24, 2.24))
                    plt.plot(time, line)
                    filename_png = f"sample_index_{index}_dimension_{i}.png"
                    plt.savefig(os.path.join(picture_path, filename_png))
                    plt.close()

                # 加载图像，并将其拼接为，[特征，通道，长，宽]
            sample_pucture = []
            for i in range(sample_raw.shape[1]):
                filename_png = f"sample_index_{index}_dimension_{i}.png"
                img_path = os.path.join(picture_path, filename_png)
                img = Image.open(img_path).convert('RGB')
                #                img = img.resize((224, 224))
                picture = np.array(img).transpose(2, 0, 1)
                img.close()
                sample_pucture.append(picture)
            sample_pucture = np.stack(sample_pucture, axis=0)
            return np.float32(self.val[index:index + self.win_size]), \
                   np.float32(self.test_labels[0:self.win_size]), sample_pucture

        elif (self.flag == 'test'):
            sample_raw = np.float32(self.test[index:index + self.win_size])
            ## 画图
            total_path = r''
            if not os.path.exists(total_path):
                os.makedirs(total_path)

            picture_path = os.path.join(total_path, f"{index}")
            if not os.path.exists(picture_path):
                os.makedirs(picture_path)

            # 创建图像(只能在完整训练一个epoch才可以省略，因为index是随机的，注意batch丢失问题
            if self.picture:
                for i in range(sample_raw.shape[1]):
                    time = [j for j in range(sample_raw.shape[0])]
                    line = sample_raw[:, i]
                    plt.figure(figsize=(2.24, 2.24))
                    plt.plot(time, line)
                    filename_png = f"sample_index_{index}_dimension_{i}.png"
                    plt.savefig(os.path.join(picture_path, filename_png))
                    plt.close()

                # 加载图像，并将其拼接为，[特征，通道，长，宽]
            sample_pucture = []
            for i in range(sample_raw.shape[1]):
                filename_png = f"sample_index_{index}_dimension_{i}.png"
                img_path = os.path.join(picture_path, filename_png)
                img = Image.open(img_path).convert('RGB')
                #                img = img.resize((224, 224))
                picture = np.array(img).transpose(2, 0, 1)
                img.close()
                sample_pucture.append(picture)
            sample_pucture = np.stack(sample_pucture, axis=0)
            return np.float32(self.test[index:index + self.win_size]), \
                   np.float32(self.test_labels[index:index + self.win_size]), sample_pucture

        else:
            return np.float32(self.test[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[
                index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])

class UCRSegLoader(object):
    def __init__(self, index, data_path, win_size, step, mode="train"):
        self.mode = mode
        self.step = step
        self.index = index
        self.win_size = win_size
        self.scaler = StandardScaler()
        data = np.load(data_path + "/UCR_"+str(index)+"_train.npy")
        self.scaler.fit(data)
        data = self.scaler.transform(data)
        test_data = np.load(data_path + "/UCR_"+str(index)+"_test.npy")
        self.test = self.scaler.transform(test_data)

        self.train = data
        self.val = self.test
        self.test_labels = np.load(data_path + "/UCR_"+str(index)+"_test_label.npy")
        if self.mode == "val":
            print("train:", self.train.shape)
            print("test:", self.test.shape)

    def __len__(self):
        if self.mode == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.mode == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.mode == 'val'):
            return np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.mode == 'test'):
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:
            return np.float32(self.test[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])
