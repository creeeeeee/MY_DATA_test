import torch.utils.data as torch_data
from typing import List, Dict
import torch
import os
import pickle
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

from Nu_dataset import Nu_dataset

DATAROOT='/home/labone/labone/jcy/nuscenes/data/mini2/'

def preprocess():
    nuscenes = NuScenes('v1.0-mini', dataroot=DATAROOT)
    helper = PredictHelper(nuscenes)

    train_data = Nu_dataset(DATAROOT,nuscenes, helper, 'mini_train')
    val_data = Nu_dataset(DATAROOT, nuscenes, helper, 'train_val')
    test_data = Nu_dataset(DATAROOT, nuscenes, helper, 'mini_val')

if __name__ == "__main__":
    preprocess()