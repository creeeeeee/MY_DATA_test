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
        self.helper = helper
        self.maps = {i: NuScenesMap(map_name=i, dataroot=self.helper.data.dataroot) for i in self.map_city}
        # 默认不采用地图拓展
        self.map_extent = False
        self.data_split = get_prediction_challenge_split(data_split, dataroot=root)

        # time: -4s TO +2s
        self.obs = 5
        self.fur = 2

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
        # for idx in range(len(self.data_split)):
            # Read data from `raw_path`.
        for instance_sample in self.data_split:
            target_feats = self.get_target_agent(instance_sample)
            lane_feats = self.get_map(instance_sample)
            other_feats = self.get_other(instance_sample)

            # 更改结构,采用pyg data格式
            #

            all_feats = {
                "target": target_feats,
                "lane": lane_feats,
                "other": other_feats
            }
            data = Data(
                edge_index=torch.from_numpy(np.array()),
                x=torch.from_numpy(np.array()),
            )
            if self.pre_filter is not None and not self.pre_filter(data):
                continue

            if self.pre_transform is not None:
                data = self.pre_transform(data)

            torch.save(data, os.path.join(self.processed_dir, f'data_{idx}.pt'))
            idx += 1

    def len(self):  # 返回数据集中示例的数目。

        return len(self.processed_file_names)
        # return len(prept_ls)

    def get(self, idx):  # 实现加载单个图的逻辑。
        data = torch.load(os.path.join(self.processed_dir, 'data_{}.pt'.format(idx)))
        return data

    def get_target_agent(self, instance_sample):
        instance_token, sample_token = instance_sample.split("_")
        past = self.helper.get_past_for_agent(instance_token, sample_token, seconds=self.obs, in_agent_frame=True)
        past = np.flip(past, 0)
        past_pad = np.zeros((int(self.obs) * 2 + 1, 2)) # 0填充空余部分
        past_pad[-past.shape[0]-1: -1] = past
        past = past_pad

        past_ = self.helper.get_past_for_agent(instance_token, sample_token, seconds=self.obs, in_agent_frame=True, just_xy=False)
        states = np.zeros((2 * self.obs + 1, 3))
        states[-1, 0] = self.helper.get_velocity_for_agent(instance_token, sample_token)
        states[-1, 1] = self.helper.get_acceleration_for_agent(instance_token, sample_token)
        states[-1, 2] = self.helper.get_heading_change_rate_for_agent(instance_token, sample_token)

        for i in range(len(past_)):
            states[-(i + 2), 0] = self.helper.get_velocity_for_agent(instance_token, past_[i]['sample_token'])
            states[-(i + 2), 1] = self.helper.get_acceleration_for_agent(instance_token, past_[i]['sample_token'])
            states[-(i + 2), 2] = self.helper.get_heading_change_rate_for_agent(instance_token, past_[i]['sample_token'])
        states = np.nan_to_num(states)

        past = np.concatenate((past, states), axis=1)
        return past

    def get_map(self,instance_sample):
        instance_token, sample_token = instance_sample.split("_")
        map_id = self.helper.get_map_name_from_sample_token(sample_token)
        map_api = self.maps[map_id]

        sample_annotation = self.helper.get_sample_annotation(instance_token, sample_token)
        x, y = sample_annotation['translation'][:2]
        yaw = quaternion_yaw(Quaternion(sample_annotation['rotation']))
        yaw = correct_yaw(yaw)
        global_pose = (x, y, yaw) # 获取世界坐标系下的坐标和yaw

        lanes = self.get_lanes_around_agent(global_pose, map_api)
        polygons = self.get_polygons_around_agent(global_pose, map_api)

        lane_node_feats, _ = self.get_lane_node_feats(global_pose, lanes, polygons)
        lane_node_feats = self.discard_poses_outside_extent(lane_node_feats)
        # 空 pad 0
        if len(lane_node_feats) == 0:
            lane_node_feats = [np.zeros((1, 5))]

        lane_node_feats, lane_node_masks = self.list_to_tensor(lane_node_feats, self.max_nodes, self.polyline_length, 5)
        lane_feats = {
            'lane_node_feats': lane_node_feats,
            'lane_node_masks': lane_node_masks
        }

        return lane_feats
    def get_other(self,instance_sample):
        # 获得周围车辆或者行人的特征

        instance_token, sample_token = instance_sample.split("_")

        vehicles = self.get_agents_of_type(instance_token, sample_token, 'vehicle')
        pedestrians = self.get_agents_of_type(instance_token, sample_token, 'human')

        # Discard poses outside map extent
        vehicles = self.discard_poses_outside_extent(vehicles)
        pedestrians = self.discard_poses_outside_extent(pedestrians)

        # While running the dataset class in 'compute_stats' mode:
        if self.mode == 'compute_stats':
            return len(vehicles), len(pedestrians)

        # Convert to fixed size arrays for batching
        vehicles, vehicle_masks = self.list_to_tensor(vehicles, self.max_vehicles, self.obs * 2 + 1, 5)
        pedestrians, pedestrian_masks = self.list_to_tensor(pedestrians, self.max_pedestrians, self.obs * 2 + 1, 5)

        other_data = {
            'vehicles': vehicles,
            'vehicle_masks': vehicle_masks,
            'pedestrians': pedestrians,
            'pedestrian_masks': pedestrian_masks
        }

        return other_data


    def get_agents_of_type(self, instance_token, sample_token, agent_type: str):
        #
        # instance_token, sample_token = self.token_list[idx].split("_")
        # origin = self.get_target_agent_global_pose(idx)
        origin = self.helper.get_annotations_for_sample(sample_token)
        agent_details = self.helper.get_past_for_sample(instance_token, seconds=self.obs, in_agent_frame=False, just_xy=False)
        agent_hist = self.helper.get_past_for_sample(sample_token, seconds=self.obs, in_agent_frame=False, just_xy=True)
        present_time = self.helper.get_annotations_for_sample(sample_token)
        for annotation in present_time:
            ann_i_t = annotation['instance_token']
            if ann_i_t in agent_hist.keys():
                present_pose = np.asarray(annotation['translation'][0:2]).reshape(1, 2)
                if agent_hist[ann_i_t].any():
                    agent_hist[ann_i_t] = np.concatenate((present_pose, agent_hist[ann_i_t]))
                else:
                    agent_hist[ann_i_t] = present_pose

        agent_list = []
        agent_i_ts = []
        for k, v in agent_details.items():
            if v and agent_type in v[0]['category_name'] and v[0]['instance_token'] != i_t:
                agent_list.append(agent_hist[k])
                agent_i_ts.append(v[0]['instance_token'])

        for agent in agent_list:
            for n, pose in enumerate(agent):
                local_pose = self.global_to_local(origin, (pose[0], pose[1], 0))
                agent[n] = np.asarray([local_pose[0], local_pose[1]])

        for n, agent in enumerate(agent_list):
            xy = np.flip(agent, axis=0)
            motion_states = self.get_past_motion_states(agent_i_ts[n], s_t)
            motion_states = motion_states[-len(xy):, :]
            agent_list[n] = np.concatenate((xy, motion_states), axis=1)

        return agent_list


