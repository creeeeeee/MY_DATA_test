from nuscenes.prediction import PredictHelper
from nuscenes import NuScenes

from dataset.Nu_dataset import Nu_dataset

DATAROOT='/home/labone/labone/jcy/nuscenes/data/mini2/'

def preprocess():
    nuscenes = NuScenes('v1.0-mini', dataroot=DATAROOT)
    helper = PredictHelper(nuscenes)

    train_data = Nu_dataset(DATAROOT,nuscenes, helper, 'mini_train')
    val_data = Nu_dataset(DATAROOT, nuscenes, helper, 'train_val')
    test_data = Nu_dataset(DATAROOT, nuscenes, helper, 'mini_val')

if __name__ == "__main__":
    preprocess()