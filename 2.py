#%%
import matplotlib.pyplot as plt
import tqdm
import numpy as np

from nuscenes.map_expansion.map_api import NuScenesMap
from nuscenes.map_expansion import arcline_path_utils
from nuscenes.map_expansion.bitmap import BitMap

#%%
nusc_map = NuScenesMap(dataroot='/home/labone/labone/jcy/nuscenes/data/mini2/', map_name='singapore-onenorth')
bitmap = BitMap(nusc_map.dataroot, nusc_map.map_name, 'basemap')
#%%
fig, ax = nusc_map.render_layers(nusc_map.non_geometric_layers,figsize=1)
plt.savefig('/home/labone/labone/jcy/testdata/1.jpg')
plt.subplots_adjust(left=0.05, bottom=0.09, right=0.08, top=0.9)
plt.show()
#%%
fig, ax = nusc_map.render_record('stop_line', nusc_map.stop_line[14]['token'], other_layers=[], bitmap=bitmap)
plt.tight_layout()
plt.show()
# plt.savefig('/home/labone/labone/jcy/testdata/2.jpg')
#%%
patch_box = (300, 1700, 100, 100)
patch_angle = 0  # Default orientation where North is up
layer_names = ['drivable_area', 'walkway']
canvas_size = (1000, 1000)
map_mask = nusc_map.get_map_mask(patch_box, patch_angle, layer_names, canvas_size)

#%%
figsize = (12, 4)
fig, ax = nusc_map.render_map_mask(patch_box, patch_angle, layer_names, canvas_size, figsize=figsize, n_row=1)
plt.show()
#%%
my_patch = (300, 1000, 500, 1200)

fig, ax = nusc_map.render_map_patch(my_patch, nusc_map.non_geometric_layers, figsize=(10, 10), bitmap=bitmap)
plt.savefig('/home/labone/labone/jcy/testdata/1.jpg')

