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
    def __init__(self, root, nuscenes, helper, data_split, args, transform=None, pre_transform=None, pre_filter=None):
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
        self.map_extent = [ -50, 50, -20, 80 ]
        self.data_split = get_prediction_challenge_split(data_split, dataroot=root)  # list

        # time: -5s TO +2s
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
        return 1

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
        return []

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

    def get_target_agent_future(self, idx: int) -> np.ndarray:
        """
        Extracts future trajectory for target agent
        :param idx: data index
        :return fut: future trajectory for target agent, shape: [t_f * 2, 2]
        """
        i_t, s_t = self.token_list[idx].split("_")
        fut = self.helper.get_future_for_agent(i_t, s_t, seconds=self.t_f, in_agent_frame=True)

        return fut

    def get_target_agent_past(self, idx: int) -> np.ndarray:
        """
        Extracts future trajectory for target agent
        :param idx: data index
        :return fut: future trajectory for target agent, shape: [t_f * 2, 2]
        """
        i_t, s_t = self.token_list[idx].split("_")
        past = self.helper.get_past_for_agent(i_t, s_t, seconds=self.obs, in_agent_frame=True)

        return past

    def get_inputs(self, idx: int) -> Dict:
        """
        Gets model inputs for nuScenes single agent prediction
        :param idx: data index
        :return inputs: Dictionary with input representations
        """
        i_t, s_t = self.token_list[idx].split("_")
        map_representation,num_lane = self.get_map_representation(idx)
        surrounding_agent_representation,num_veh,num_ped = self.get_surrounding_agent_representation(idx)
        target_agent_representation = self.get_target_agent_representation(idx)
        inputs = {'instance_token': i_t,
                    'sample_token': s_t,
                    'map_representation': map_representation,
                    'surrounding_agent_representation': surrounding_agent_representation,
                    'target_agent_representation': target_agent_representation}
        return inputs

    def get_target_agent_global_pose(self, idx: int) -> Tuple[float, float, float]:
        """
        Returns global pose of target agent
        :param idx: data index
        :return global_pose: (x, y, yaw) or target agent in global co-ordinates
        """
        i_t, s_t = self.token_list[idx].split("_")
        sample_annotation = self.helper.get_sample_annotation(i_t, s_t)
        x, y = sample_annotation['translation'][:2]
        yaw = quaternion_yaw(Quaternion(sample_annotation['rotation']))
        yaw = correct_yaw(yaw)
        global_pose = (x, y, yaw)

        return global_pose

    def get_ground_truth(self, idx: int) -> Dict:
        """
        Gets ground truth labels for nuScenes single agent prediction
        :param idx: data index
        :return ground_truth: Dictionary with grund truth labels
        """
        target_agent_future = self.get_target_agent_future(idx)
        ground_truth = {'traj': target_agent_future}
        return ground_truth

    def get_target_agent_representation(self, idx: int) -> np.ndarray:
        """
        Extracts target agent representation
        :param idx: data index
        :return hist: track history for target agent, shape: [t_h * 2, 5]
        """
        i_t, s_t = self.token_list[idx].split("_")

        # x, y co-ordinates in agent's frame of reference
        hist = self.helper.get_past_for_agent(i_t, s_t, seconds=self.t_h, in_agent_frame=True)

        # Zero pad for track histories shorter than t_h 零填充很重要
        hist_zeropadded = np.zeros((int(self.t_h) * 2 + 1, 2))

        # Flip to have correct order of timestamps
        hist = np.flip(hist, 0)
        hist_zeropadded[-hist.shape[0]-1: -1] = hist
        hist = hist_zeropadded

        # Get velocity, acc and yaw_rate over past t_h sec
        motion_states = self.get_past_motion_states(i_t, s_t)
        hist = np.concatenate((hist, motion_states), axis=1)

        return hist

    def get_map_representation(self, idx: int) -> Tuple[Dict, int]:
        """
        Extracts map representation
        :param idx: data index
        :return: Returns an ndarray with lane node features, shape [max_nodes, polyline_length, 5] and an ndarray of
            masks of the same shape, with value 1 if the nodes/poses are empty,
        """
        i_t, s_t = self.token_list[idx].split("_")
        map_name = self.helper.get_map_name_from_sample_token(s_t)
        map_api = self.maps[map_name]

        # Get agent representation in global co-ordinates
        global_pose = self.get_target_agent_global_pose(idx)

        # Get lanes around agent within map_extent
        lanes = self.get_lanes_around_agent(global_pose, map_api)

        # Get relevant polygon layers from the map_api
        polygons = self.get_polygons_around_agent(global_pose, map_api)

        # Get vectorized representation of lanes
        lane_node_feats, _ = self.get_lane_node_feats(global_pose, lanes, polygons)

        # Discard lanes outside map extent
        lane_node_feats = self.discard_poses_outside_extent(lane_node_feats)

        # Add dummy node (0, 0, 0, 0, 0) if no lane nodes are found
        if len(lane_node_feats) == 0:
            lane_node_feats = [np.zeros((1, 5))]

        # While running the dataset class in 'compute_stats' mode:
        # if self.mode == 'compute_stats':
        #     return len(lane_node_feats)

        # Convert list of lane node feats to fixed size numpy array and masks
        lane_node_feats, lane_node_masks = self.list_to_tensor(lane_node_feats, self.max_nodes, self.polyline_length, 5)

        map_representation = {
            'lane_node_feats': lane_node_feats,
            'lane_node_masks': lane_node_masks
        }

        num_lane = len(lane_node_feats)

        return map_representation, num_lane

    #  Union[Tuple[int, int], Dict]:
    def get_surrounding_agent_representation(self, idx: int) -> \
            Tuple[Dict, int, int]:
        """
        Extracts surrounding agent representation
        :param idx: data index
        :return: ndarrays with surrounding pedestrian and vehicle track histories and masks for non-existent agents
        """

        # Get vehicles and pedestrian histories for current sample
        vehicles = self.get_agents_of_type(idx, 'vehicle')
        pedestrians = self.get_agents_of_type(idx, 'human')

        # Discard poses outside map extent
        vehicles = self.discard_poses_outside_extent(vehicles)
        pedestrians = self.discard_poses_outside_extent(pedestrians)

        # While running the dataset class in 'compute_stats' mode:
        # if self.mode == 'compute_stats':
        #     return len(vehicles), len(pedestrians)

        # Convert to fixed size arrays for batching
        vehicles, vehicle_masks = self.list_to_tensor(vehicles, self.max_vehicles, self.t_h * 2 + 1, 5)
        pedestrians, pedestrian_masks = self.list_to_tensor(pedestrians, self.max_pedestrians, self.t_h * 2 + 1, 5)

        surrounding_agent_representation = {
            'vehicles': vehicles,
            'vehicle_masks': vehicle_masks,
            'pedestrians': pedestrians,
            'pedestrian_masks': pedestrian_masks
        }
        num_veh = len(vehicles)
        num_ped = len(pedestrians)
        return surrounding_agent_representation, num_veh, num_ped

    def get_lanes_around_agent(self, global_pose: Tuple[float, float, float], map_api: NuScenesMap) -> Dict:
        """
        Gets lane polylines around the target agent
        :param global_pose: (x, y, yaw) or target agent in global co-ordinates
        :param map_api: nuScenes map api
        :return lanes: Dictionary of lane polylines
        """
        x, y, _ = global_pose
        radius = max(self.map_extent)
        lanes = map_api.get_records_in_radius(x, y, radius, ['lane', 'lane_connector'])
        lanes = lanes['lane'] + lanes['lane_connector']
        lanes = map_api.discretize_lanes(lanes, self.polyline_resolution)

        return lanes

    def get_polygons_around_agent(self, global_pose: Tuple[float, float, float], map_api: NuScenesMap) -> Dict:
        """
        Gets polygon layers around the target agent e.g. crosswalks, stop lines
        :param global_pose: (x, y, yaw) or target agent in global co-ordinates
        :param map_api: nuScenes map api
        :return polygons: Dictionary of polygon layers, each type as a list of shapely Polygons
        """
        x, y, _ = global_pose
        radius = max(self.map_extent)
        record_tokens = map_api.get_records_in_radius(x, y, radius, ['stop_line', 'ped_crossing'])
        polygons = {k: [] for k in record_tokens.keys()}
        for k, v in record_tokens.items():
            for record_token in v:
                polygon_token = map_api.get(k, record_token)['polygon_token']
                polygons[k].append(map_api.extract_polygon(polygon_token)) # 很重要

        return polygons

    def get_lane_node_feats(self, origin: Tuple, lanes: Dict[str, List[Tuple]],
                            polygons: Dict[str, List[Polygon]]) -> Tuple[List[np.ndarray], List[str]]:
        """
        Generates vector HD map representation in the agent centric frame of reference
        :param origin: (x, y, yaw) of target agent in global co-ordinates
        :param lanes: lane centerline poses in global co-ordinates
        :param polygons: stop-line and cross-walk polygons in global co-ordinates
        :return:
        """

        # Convert lanes to list
        lane_ids = [k for k, v in lanes.items()]
        lanes = [v for k, v in lanes.items()]

        # Get flags indicating whether a lane lies on stop lines or crosswalks
        lane_flags = self.get_lane_flags(lanes, polygons)

        # Convert lane polylines to local coordinates:
        lanes = [np.asarray([self.global_to_local(origin, pose) for pose in lane]) for lane in lanes]

        # Concatenate lane poses and lane flags
        lane_node_feats = [np.concatenate((lanes[i], lane_flags[i]), axis=1) for i in range(len(lanes))]

        # Split lane centerlines into smaller segments:
        lane_node_feats, lane_node_ids = self.split_lanes(lane_node_feats, self.polyline_length, lane_ids)

        return lane_node_feats, lane_node_ids

    def get_agents_of_type(self, idx: int, agent_type: str) -> List[np.ndarray]:
        """
        Returns surrounding agents of a particular class for a given sample
        :param idx: data index
        :param agent_type: 'human' or 'vehicle'
        :return: list of ndarrays of agent track histories.
        """
        i_t, s_t = self.token_list[idx].split("_")

        # Get agent representation in global co-ordinates
        origin = self.get_target_agent_global_pose(idx)

        # Load all agents for sample
        agent_details = self.helper.get_past_for_sample(s_t, seconds=self.t_h, in_agent_frame=False, just_xy=False)
        agent_hist = self.helper.get_past_for_sample(s_t, seconds=self.t_h, in_agent_frame=False, just_xy=True)

        # Add present time to agent histories        # 当前时间
        present_time = self.helper.get_annotations_for_sample(s_t)
        for annotation in present_time:
            ann_i_t = annotation['instance_token']
            if ann_i_t in agent_hist.keys():
                present_pose = np.asarray(annotation['translation'][0:2]).reshape(1, 2)
                if agent_hist[ann_i_t].any():
                    agent_hist[ann_i_t] = np.concatenate((present_pose, agent_hist[ann_i_t]))
                else:
                    agent_hist[ann_i_t] = present_pose

        # Filter for agent type
        agent_list = []
        agent_i_ts = []
        for k, v in agent_details.items():
            if v and agent_type in v[0]['category_name'] and v[0]['instance_token'] != i_t: # 这里是用来判断v存在并且属于这一类，而且要去掉目标instance token
                agent_list.append(agent_hist[k])
                agent_i_ts.append(v[0]['instance_token'])

        # Convert to target agent's frame of reference
        for agent in agent_list:
            for n, pose in enumerate(agent):
                local_pose = self.global_to_local(origin, (pose[0], pose[1], 0))
                agent[n] = np.asarray([local_pose[0], local_pose[1]])

        # Flip history to have most recent time stamp last and extract past motion states
        for n, agent in enumerate(agent_list):
            xy = np.flip(agent, axis=0)
            motion_states = self.get_past_motion_states(agent_i_ts[n], s_t)
            motion_states = motion_states[-len(xy):, :]
            agent_list[n] = np.concatenate((xy, motion_states), axis=1)

        return agent_list

    def discard_poses_outside_extent(self, pose_set: List[np.ndarray],
                                     ids: List[str] = None) -> Union[List[np.ndarray],
                                                                     Tuple[List[np.ndarray], List[str]]]:
        """
        Discards lane or agent poses outside predefined extent in target agent's frame of reference.
        :param pose_set: agent or lane polyline poses
        :param ids: annotation record tokens for pose_set. Only applies to lanes.
        :return: Updated pose set
        """
        updated_pose_set = []
        updated_ids = []

        for m, poses in enumerate(pose_set):
            flag = False
            for n, pose in enumerate(poses):
                if self.map_extent[0] <= pose[0] <= self.map_extent[1] and \
                        self.map_extent[2] <= pose[1] <= self.map_extent[3]:
                    flag = True

            if flag:
                updated_pose_set.append(poses)
                if ids is not None:
                    updated_ids.append(ids[m])

        if ids is not None:
            return updated_pose_set, updated_ids
        else:
            return updated_pose_set

    def load_stats(self) -> Dict[str, int]:
        """
        Function to load dataset statistics like max surrounding agents, max nodes, max edges etc.
        读取状态文件
        """
        filename = os.path.join(self.data_dir, 'stats.pickle')
        if not os.path.isfile(filename):
            raise Exception('Could not find dataset statistics. Please run the dataset in compute_stats mode')

        with open(filename, 'rb') as handle:
            stats = pickle.load(handle)

        return stats

    def get_past_motion_states(self, i_t, s_t):
        """
        Returns past motion states: v, a, yaw_rate for a given instance and sample token over self.t_h seconds
        """
        motion_states = np.zeros((2 * self.t_h + 1, 3))
        motion_states[-1, 0] = self.helper.get_velocity_for_agent(i_t, s_t)
        motion_states[-1, 1] = self.helper.get_acceleration_for_agent(i_t, s_t)
        motion_states[-1, 2] = self.helper.get_heading_change_rate_for_agent(i_t, s_t)
        hist = self.helper.get_past_for_agent(i_t, s_t, seconds=self.t_h, in_agent_frame=True, just_xy=False)

        for k in range(len(hist)):
            motion_states[-(k + 2), 0] = self.helper.get_velocity_for_agent(i_t, hist[k]['sample_token'])
            motion_states[-(k + 2), 1] = self.helper.get_acceleration_for_agent(i_t, hist[k]['sample_token'])
            motion_states[-(k + 2), 2] = self.helper.get_heading_change_rate_for_agent(i_t, hist[k]['sample_token'])

        motion_states = np.nan_to_num(motion_states)
        return motion_states

    def save_data(self, idx: int, data: Dict):
        """
        Saves extracted pre-processed data
        :param idx: data index
        :param data: pre-processed data
        """
        filename = os.path.join(self.data_dir, self.token_list[idx] + '.pickle')
        with open(filename, 'wb') as handle:
            pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def load_data(self, idx: int) -> Dict:
        """
        Function to load extracted data.
        :param idx: data index
        :return data: Dictionary with batched tensors
        """
        filename = os.path.join(self.data_dir, self.token_list[idx] + '.pickle')


        if not os.path.isfile(filename):
            print(filename)
            raise Exception('Could not find data. Please run the dataset in extract_data mode')

        with open(filename, 'rb') as handle:
            data = pickle.load(handle)
        return data

    @staticmethod
    def global_to_local(origin: Tuple, global_pose: Tuple) -> Tuple:
        """
        Converts pose in global co-ordinates to local co-ordinates.
        :param origin: (x, y, yaw) of origin in global co-ordinates
        :param global_pose: (x, y, yaw) in global co-ordinates
        :return local_pose: (x, y, yaw) in local co-ordinates
        """
        # Unpack
        global_x, global_y, global_yaw = global_pose
        origin_x, origin_y, origin_yaw = origin

        # Translate
        local_x = global_x - origin_x
        local_y = global_y - origin_y

        # Rotate
        global_yaw = correct_yaw(global_yaw)
        theta = np.arctan2(-np.sin(global_yaw-origin_yaw), np.cos(global_yaw-origin_yaw))

        r = np.asarray([[np.cos(np.pi/2 - origin_yaw), np.sin(np.pi/2 - origin_yaw)],
                        [-np.sin(np.pi/2 - origin_yaw), np.cos(np.pi/2 - origin_yaw)]])
        local_x, local_y = np.matmul(r, np.asarray([local_x, local_y]).transpose())

        local_pose = (local_x, local_y, theta)

        return local_pose

    @staticmethod
    def split_lanes(lanes: List[np.ndarray], max_len: int, lane_ids: List[str]) -> Tuple[List[np.ndarray], List[str]]:
        """
        Splits lanes into roughly equal sized smaller segments with defined maximum length
        :param lanes: list of lane poses
        :param max_len: maximum admissible length of polyline
        :param lane_ids: list of lane ID tokens
        :return lane_segments: list of smaller lane segments
                lane_segment_ids: list of lane ID tokens corresponding to original lane that the segment is part of
        """
        lane_segments = []
        lane_segment_ids = []
        for idx, lane in enumerate(lanes):
            n_segments = int(np.ceil(len(lane) / max_len))
            n_poses = int(np.ceil(len(lane) / n_segments))
            for n in range(n_segments):
                lane_segment = lane[n * n_poses: (n+1) * n_poses]
                lane_segments.append(lane_segment)
                lane_segment_ids.append(lane_ids[idx])

        return lane_segments, lane_segment_ids

    @staticmethod
    def get_lane_flags(lanes: List[List[Tuple]], polygons: Dict[str, List[Polygon]]) -> List[np.ndarray]:
        """
        Returns flags indicating whether each pose on lane polylines lies on polygon map layers
        like stop-lines or cross-walks
        :param lanes: list of lane poses
        :param polygons: dictionary of polygon layers
        :return lane_flags: list of ndarrays with flags
        """

        lane_flags = [np.zeros((len(lane), len(polygons.keys()))) for lane in lanes]
        for lane_num, lane in enumerate(lanes):
            for pose_num, pose in enumerate(lane):
                point = Point(pose[0], pose[1])
                for n, k in enumerate(polygons.keys()):
                    polygon_list = polygons[k]
                    for polygon in polygon_list:
                        if polygon.contains(point):
                            lane_flags[lane_num][pose_num][n] = 1
                            break

        return lane_flags

    @staticmethod
    def list_to_tensor(feat_list: List[np.ndarray], max_num: int, max_len: int,
                       feat_size: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Converts a list of sequential features (e.g. lane polylines or agent history) to fixed size numpy arrays for
        forming mini-batches

        :param feat_list: List of sequential features
        :param max_num: Maximum number of sequences in List
        :param max_len: Maximum length of each sequence
        :param feat_size: Feature dimension
        :return: 1) ndarray of features of shape [max_num, max_len, feat_dim]. Has zeros where elements are missing,
            2) ndarray of binary masks of shape [max_num, max_len, feat_dim]. Has ones where elements are missing.
        """
        feat_array = np.zeros((max_num, max_len, feat_size))
        mask_array = np.ones((max_num, max_len, feat_size))
        for n, feats in enumerate(feat_list):
            feat_array[n, :len(feats), :] = feats
            mask_array[n, :len(feats), :] = 0

        return feat_array, mask_array

    @staticmethod
    def flip_horizontal(data: Dict):
        """
        Helper function to randomly flip some samples across y-axis for data augmentation
        :param data: Dictionary with inputs and ground truth values.
        :return: data: Dictionary with inputs and ground truth values fligpped along y-axis.
        """
        # Flip target agent
        hist = data['inputs']['target_agent_representation']
        hist[:, 0] = -hist[:, 0]  # x-coord
        hist[:, 4] = -hist[:, 4]  # yaw-rate
        data['inputs']['target_agent_representation'] = hist

        # Flip lane node features
        lf = data['inputs']['map_representation']['lane_node_feats']
        lf[:, :, 0] = -lf[:, :, 0]  # x-coord
        lf[:, :, 2] = -lf[:, :, 2]  # yaw
        data['inputs']['map_representation']['lane_node_feats'] = lf

        # Flip surrounding agents
        vehicles = data['inputs']['surrounding_agent_representation']['vehicles']
        vehicles[:, :, 0] = -vehicles[:, :, 0]  # x-coord
        vehicles[:, :, 4] = -vehicles[:, :, 4]  # yaw-rate
        data['inputs']['surrounding_agent_representation']['vehicles'] = vehicles

        peds = data['inputs']['surrounding_agent_representation']['pedestrians']
        peds[:, :, 0] = -peds[:, :, 0]  # x-coord
        peds[:, :, 4] = -peds[:, :, 4]  # yaw-rate
        data['inputs']['surrounding_agent_representation']['pedestrians'] = peds

        # Flip groud truth trajectory
        fut = data['ground_truth']['traj']
        fut[:, 0] = -fut[:, 0]  # x-coord
        data['ground_truth']['traj'] = fut

        return data


