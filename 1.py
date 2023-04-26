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
nuscenes.list_scenes()
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

motion_s = helper.get_past_motion_states(instance_token, sample_token)

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
fig, ax = nusc_map.render_layers(nusc_map.non_geometric_layers, figsize=1)
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



