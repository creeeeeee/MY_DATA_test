from nuscenes.prediction import PredictHelper
from nuscenes import NuScenes
from dataset.interface import TrajectoryDataset
import torch
import os
import pickle
from typing import List, Dict
import torch.utils.data as torch_data
from dataset.nuScenes_vector import NuScenesVector
from dataset.Nu_dataset import Nu_dataset
from nuscenes.eval.prediction.splits import get_prediction_challenge_split

DATAROOT='/home/labone/labone/jcy/nuscenes/data/mini2/'

def preprocess():
    nuscenes = NuScenes('v1.0-mini', dataroot=DATAROOT)
    helper = PredictHelper(nuscenes)

    arg = {'split': 'mini_train',
           't_h': 1,
           't_f': 2,
           'map_extent': [-50, 50, -20, 80],
           'polyline_resolution': 1,
           'polyline_length': 20,
           'traversal_horizon': 15,
           'random_flips': True
           }
    data_split = get_prediction_challenge_split('mini_train', dataroot=DATAROOT)
    train_data = Nu_dataset(DATAROOT, nuscenes, helper, data_split,args)

    # data_split = get_prediction_challenge_split('mini_train', dataroot=DATAROOT)  # list
    #
    #
    # train_data = NuScenesVector('compute_stats', DATAROOT, arg, helper)
    # compute_dataset_stats([train_data], 4, 0, verbose=True)
    # train_set = NuScenesVector('extract_data', DATAROOT, arg, helper)
    #
    # y = train_set[3]
    #
    # real_data = NuScenesVector('load_data', DATAROOT, arg, helper)


    # val_data = Nu_dataset(DATAROOT, nuscenes, helper, 'train_val')
    # test_data = Nu_dataset(DATAROOT, nuscenes, helper, 'mini_val')

    x=1


def compute_dataset_stats(dataset_splits: List[TrajectoryDataset], batch_size: int, num_workers: int, verbose=False):
    """
    计算数据集状态
    :param dataset_splits: List of dataset objects usually corresponding to the train, val and test splits 分割后的数据
    :param batch_size: Batch size for dataloader
    :param num_workers: Number of workers for dataloader = 0
    :param verbose: Whether to print progress 过程是否输出
    """
    # Check if all datasets have been initialized with the correct mode
    for dataset in dataset_splits:
        if dataset.mode != 'compute_stats':
            raise Exception('Dataset mode should be compute_stats')

    # Initialize data loaders
    data_loaders = []
    for dataset in dataset_splits:
        dl = torch_data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        data_loaders.append(dl)

    # Initialize dataset statistics
    stats = {}

    # For printing progress S=[x,y,v,a,w,I]
    print("Computing dataset stats...")
    num_mini_batches = sum([len(data_loader) for data_loader in data_loaders])
    mini_batch_count = 0

    # Loop over splits and mini-batches
    for data_loader in data_loaders:
        for i, mini_batch_stats in enumerate(data_loader):
            for k, v in mini_batch_stats.items():
                if k in stats.keys():
                    stats[k] = max(stats[k], torch.max(v).item())
                else:
                    stats[k] = torch.max(v).item()

            # Show progress
            if verbose:
                print("mini batch " + str(mini_batch_count + 1) + '/' + str(num_mini_batches) )
                mini_batch_count += 1

    # Save stats
    filename = os.path.join(dataset_splits[0].data_dir, 'stats.pickle')
    with open(filename, 'wb') as handle:
        pickle.dump(stats, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    preprocess()