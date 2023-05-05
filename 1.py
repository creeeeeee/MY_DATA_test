from nuscenes.prediction.input_representation.static_layers import correct_yaw
from nuscenes.map_expansion.map_api import NuScenesMap
from nuscenes.eval.common.utils import quaternion_yaw
from pyquaternion import Quaternion
from nuscenes.prediction import PredictHelper
from nuscenes.eval.prediction.splits import get_prediction_challenge_split
import numpy as np
from typing import Dict, Tuple, Union, List
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
from nuscenes import NuScenes
import os
import pickle
import torch

#%%
DATAROOT='/home/labone/labone/jcy/nuscenes/data/mini2/'
nuscenes = NuScenes('v1.0-mini', dataroot=DATAROOT)

#%%
mini_train = get_prediction_challenge_split("mini_train", dataroot=DATAROOT)

helper = PredictHelper(nuscenes)
instance_token, sample_token = mini_train[0].split("_")
annotation = helper.get_sample_annotation(instance_token, sample_token)

#%%
map_locs = ['singapore-onenorth', 'singapore-hollandvillage', 'singapore-queenstown', 'boston-seaport']
maps = {i: NuScenesMap(map_name=i, dataroot=helper.data.dataroot) for i in map_locs}

map_name = helper.get_map_name_from_sample_token(sample_token)
map_api = maps[map_name]

sample_annotation = helper.get_sample_annotation(instance_token, sample_token)
x, y = sample_annotation['translation'][:2]
yaw = quaternion_yaw(Quaternion(sample_annotation['rotation']))
yaw = correct_yaw(yaw)
global_pose = (x, y, yaw)


#%%
layer_ = ['drivable_area', # 可驾驶区域
 'road_segment',  #
 'road_block',
 'lane',
 'ped_crossing',
 'walkway',
 'stop_line',
 'carpark_area',
 'road_divider',
 'lane_divider',
 'traffic_light']

radius = max([-50, 50, -20, 80])
lanes = map_api.get_records_in_radius(x, y, radius, ['lane', 'lane_connector'])

#%%
sample_polygon = map_api.polygon[3]

#%%
lanes = lanes['lane'] + lanes['lane_connector']
polyline_resolution = 1

#%%
lanes = map_api.discretize_lanes(lanes, polyline_resolution) # 连接表

#%%
record_tokens = map_api.get_records_in_radius(x, y, radius, ['stop_line', 'ped_crossing'])
polygons = {k: [] for k in record_tokens.keys()}
for k, v in record_tokens.items():
    for record_token in v:
        polygon_token = map_api.get(k, record_token)['polygon_token']
        polygons[k].append(map_api.extract_polygon(polygon_token))

#%%

def get_lane_flags(lanes: List[List[Tuple]], polygons: Dict[str, List[Polygon]]) -> List[np.ndarray]:
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

#%%
lane_ids = [k for k, v in lanes.items()]
lanes = [v for k, v in lanes.items()]

lane_flags = get_lane_flags(lanes, polygons)

#%%
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
    theta = np.arctan2(-np.sin(global_yaw - origin_yaw), np.cos(global_yaw - origin_yaw))

    r = np.asarray([[np.cos(np.pi / 2 - origin_yaw), np.sin(np.pi / 2 - origin_yaw)],
                    [-np.sin(np.pi / 2 - origin_yaw), np.cos(np.pi / 2 - origin_yaw)]])
    local_x, local_y = np.matmul(r, np.asarray([local_x, local_y]).transpose())

    local_pose = (local_x, local_y, theta)

    return local_pose

#%%
lanes = [np.asarray([global_to_local(global_pose, pose) for pose in lane]) for lane in lanes]

#%%
lane_node_feats = [np.concatenate((lanes[i], lane_flags[i]), axis=1) for i in range(len(lanes))]
#%%
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
            lane_segment = lane[n * n_poses: (n + 1) * n_poses]
            lane_segments.append(lane_segment)
            lane_segment_ids.append(lane_ids[idx])

    return lane_segments, lane_segment_ids

lane_node_feats, lane_node_ids = split_lanes(lane_node_feats, 20, lane_ids)



#%%
#nuscenes.list_scenes() # 10个场景
my_scene = nuscenes.scene[0]
first_sample_token = my_scene['first_sample_token']
my_sample = nuscenes.get('sample', first_sample_token)

#%%
my_annotation_token = my_sample['anns'][19]
my_annotation_metadata = nuscenes.get('sample_annotation', my_annotation_token)
len(mini_train)

#%%
instance_token, sample_token = mini_train[3].split("_")
future_xy_local = helper.get_future_for_agent(instance_token, sample_token, seconds=4, in_agent_frame=True)
past = helper.get_past_for_agent(instance_token, sample_token, seconds=2, in_agent_frame=True)
sample = helper.get_annotations_for_sample(sample_token)



#%%
from nuscenes.map_expansion.map_api import NuScenesMap
nusc_map = NuScenesMap(map_name='singapore-onenorth', dataroot=DATAROOT)

#%%
x, y, yaw = 395, 1095, 0
closest_lane = nusc_map.get_closest_lane(x, y, radius=2)
lane_record = nusc_map.get_arcline_path(closest_lane)

#%%
from nuscenes.map_expansion import arcline_path_utils
poses = arcline_path_utils.discretize_lane(lane_record, resolution_meters=1)

closest_pose_on_lane, distance_along_lane = arcline_path_utils.project_pose_to_lane((x, y, yaw), lane_record)

arcline_path_utils.get_curvature_at_distance_along_lane(distance_along_lane, lane_record)
#%%
import matplotlib.pyplot as plt

from nuscenes.prediction.input_representation.static_layers import StaticLayerRasterizer
from nuscenes.prediction.input_representation.agents import AgentBoxesWithFadedHistory
from nuscenes.prediction.input_representation.interface import InputRepresentation
from nuscenes.prediction.input_representation.combinators import Rasterizer

static_layer_rasterizer = StaticLayerRasterizer(helper)
agent_rasterizer = AgentBoxesWithFadedHistory(helper, seconds_of_history=1)
mtp_input_representation = InputRepresentation(static_layer_rasterizer, agent_rasterizer, Rasterizer())

instance_token_img, sample_token_img = 'bc38961ca0ac4b14ab90e547ba79fbb6', '7626dde27d604ac28a0240bdd54eba7a'
anns = [ann for ann in nuscenes.sample_annotation if ann['instance_token'] == instance_token_img]
img = mtp_input_representation.make_input_representation(instance_token_img, sample_token_img)

plt.imshow(img)
plt.show()

#%%

import matplotlib.pyplot as plt
import tqdm
import numpy as np

from nuscenes.map_expansion.map_api import NuScenesMap
from nuscenes.map_expansion import arcline_path_utils
from nuscenes.map_expansion.bitmap import BitMap

nusc_map = NuScenesMap(dataroot='/home/labone/labone/jcy/nuscenes/data/mini2/', map_name='singapore-onenorth')
#%%
sample_drivable_area = nusc_map.drivable_area[0]
fig, ax = nusc_map.render_record('drivable_area', sample_drivable_area['token'],other_layers=[])

plt.figure()
plt.subplots_adjust(bottom=0.3)
plt.show()

#%%

sample_road_segment = nusc_map.road_segment[600]
sample_intersection_road_segment = nusc_map.road_segment[3]
fig, ax = nusc_map.render_record('road_segment', sample_intersection_road_segment['token'], other_layers=[])
plt.figure()
plt.show()

#%%
bitmap = BitMap(nusc_map.dataroot, nusc_map.map_name, 'basemap')
fig, ax = nusc_map.render_layers(['lane'], figsize=1, bitmap=bitmap)

plt.figure()
plt.show()
#%%
fig, ax = nusc_map.render_record('stop_line', nusc_map.stop_line[14]['token'], other_layers=[], bitmap=bitmap)

plt.figure()
plt.show()
#%%
patch_box = (300, 1700, 100, 100)
patch_angle = 0  # Default orientation where North is up
layer_names = ['drivable_area', 'walkway']
canvas_size = (1000, 1000)
map_mask = nusc_map.get_map_mask(patch_box, patch_angle, layer_names, canvas_size)
map_mask[0]



