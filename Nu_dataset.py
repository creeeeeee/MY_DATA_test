from nuscenes.prediction.input_representation.static_layers import correct_yaw
from nuscenes.map_expansion.map_api import NuScenesMap
from nuscenes.eval.prediction.splits import get_prediction_challenge_split
from nuscenes.eval.common.utils import quaternion_yaw
from pyquaternion import Quaternion
from nuscenes.prediction import PredictHelper
from typing import Dict, Tuple, Union, List
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
import os
import pickle
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from torch_geometric.data import Dataset, Data
from torch_geometric.data import InMemoryDataset
from collections.abc import Sequence


def get_edge(num_nodes, start=0):
    '''
    return a tensor(2, edges), indicing edge_index
    '''
    # 例如有5个
    # to=[0,1,2,3,4]
    to_ = np.arange(num_nodes, dtype=np.int64)
    # 2行0列
    edge = np.empty((2, 0))
    for i in range(num_nodes):
        from_ = np.ones(num_nodes, dtype=np.int64) * i
        from_de = np.hstack([from_[:i], from_[i + 1:]])  # 去掉from_[i]
        to_de = np.hstack([to_[:i], to_[i + 1:]])  # 去掉to_[i]

        edge = np.hstack((
            edge,
            np.vstack((  # 上from下to
                from_de,
                to_de,
            ))
        ))
    edge = edge + start

    return edge.astype(np.int64), num_nodes + start


def get_edge_mask(mask, start):
    # 不用num
    # 已知[0. 0. 0. 0. 0. 0. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.] mask

    list = []
    for i, j in enumerate(mask):
        if j == 1:
            list.append(i)
        else:
            continue
    to_ = np.array(list, dtype=np.int64)
    edge = np.empty((2, 0))
    for i, j in enumerate(list):
        from_ = np.ones(len(list), dtype=np.int64) * j
        from_de = np.hstack([from_[:i], from_[i + 1:]])  # 去掉from_[i]
        to_de = np.hstack([to_[:i], to_[i + 1:]])  # 去掉to_[i]
        edge = np.hstack((
            edge,
            np.vstack((  # 上from下to
                from_de,
                to_de,
            ))
        ))
    edge = edge + start

    return edge.astype(np.int64), 20 + start


class Nu_dataset(Dataset):
    def __init__(self, root, nuscenes, helper, data_split, transform=None, pre_transform=None, pre_filter=None):
        # data_dir:是存储数据集的文件夹路径
        # transform:数据转换函数，每一次获取数据时被调用。接受一个Data对象并返回一个转换后的Data对象。
        # pre_transform:数据转换函数，数据保存到文件前被调用。接受一个Data对象并返回一个转换后的Data对象
        super().__init__(root, transform, pre_transform, pre_filter)  # 继承父类所有方法

        self.nuscenes = nuscenes
        self.map_city = ['singapore-onenorth', 'singapore-hollandvillage',
                         'singapore-queenstown', 'boston-seaport']
        self.helper = PredictHelper(nuscenes)
        # 默认不采用地图拓展
        self.map_extent = False
        self.data = get_prediction_challenge_split(data_split, dataroot=root)


    @property
    def raw_file_names(self):  # 原始数据文件夹存放位置
        # return ['some_file_1', 'some_file_2', ...]
        """
        data_dir_ls = []
        for d_dir in os.listdir(self.root):
            if not d_dir.endswith('pkl'):
                continue
            if d_dir == 'norm_center_dict.pkl':
                continue
            data_dir_ls.append(d_dir)

        return data_dir_ls
        """
        pass

    @property
    def processed_file_names(self):  # 处理后保存的文件名
        """prept_ls = []
        if os.path.exists(self.processed_dir):
            for prepd in os.listdir(self.processed_dir):
                if prepd.startswith('data'):
                    prept_ls.append(prepd)
            return prept_ls
        else:
            return []"""
        pass

    def download(self):  # 这个例子中不是从网上下载的，所以这个函数pass掉
        pass

    def process(self):  # 处理数据的函数,最关键（怎么创建，怎么保存）
        idx = 0

        for raw_path in self.raw_paths:
            # Read data from `raw_path`.
            data = Data(...)

            if self.pre_filter is not None and not self.pre_filter(data):
                continue

            if self.pre_transform is not None:
                data = self.pre_transform(data)

            torch.save(data, os.path.join(self.processed_dir, f'data_{idx}.pt'))
            idx += 1




    # len和get是区别于in memory dataset的主要函数

    def len(self):  # 返回数据集中示例的数目。

        return len(self.processed_file_names)
        # return len(prept_ls)

    def get(self, idx):  # 实现加载单个图的逻辑。
        data = torch.load(os.path.join(self.processed_dir, 'data_{}.pt'.format(idx)))
        return data
