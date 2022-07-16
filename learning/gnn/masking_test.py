import pickle
import numpy as np 

pik_file = "/home/alex/drake-tamp/experiments/jobs/small-set-rgbd/oracle/1_0_1_40.yaml_logs/2022-06-21-00:55:52.405_labels.pkl" 
with open(pik_file, "rb") as file:
    data = pickle.load(file)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda import stream
from torch.nn.utils.rnn import pad_sequence
from torch_geometric.data import Batch
from torch_geometric.nn import GCNConv, MetaLayer
from torch_scatter import scatter_mean

def get_perception(perception_raw):
    # import matplotlib.pyplot as plt 
    # plt.subplot(121)
    # plt.imshow(problem_info.perception['color'])
    # plt.title('Color image')
    # plt.subplot(122)
    # plt.imshow(np.squeeze(problem_info.perception['depth']))
    # plt.title('Depth image')
    # plt.savefig('/home/alex/drake-tamp/learning/gnn/test_rgbd_0.png')
    
    #TODO do clipping and normalization
    from numpy import inf 
    #the 4th channel is alpha we don't care 
    col = perception_raw['color'][:, :, :3]
    dep = perception_raw['depth'][:, :, ]

    #change everything at infinite distance to just be the ground 
    dep[dep == inf] = np.max(dep[dep < 1000])

    #stack the depth and rgb data
    perception = torch.from_numpy(np.concatenate((col, dep), axis=2))

    #normalizing
    input = torch.reshape(perception, (640000, 4))
    means = torch.mean(input, axis=0)
    std = torch.std(input, dim=0)
    normalized = (input - means)/std
    input = torch.reshape(normalized, (800, 800, 4))

    #reshaping to NCWH format for convolution 
    perception = input.permute(2, 0, 1)
    perception = perception.reshape([1, 4, 800, 800])
    perception = torch.nn.functional.interpolate(perception, (200, 200))
    return perception


perception = get_perception(data['problem_info'].perception )

import matplotlib.pyplot as plt

plt.imshow(data['problem_info'].perception['color'])
plt.show()

#return mask for object given object coords in image space 
import cv2
def get_object_mask(rgb, coords):
    mask = np.zeros(rgb.shape)
    ret = cv2.floodFill(rgb, mask, coords, (250, 250, 250))
    plt.imshow(ret)
    plt.show()

rgb = data['problem_info'].perception['color']
get_object_mask(rgb, (100, 400))