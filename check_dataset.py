# %%
# import public packages
import torch
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch
from PIL import Image
# %%
# import custom packages
import dataset
import model
# %%
INIT_PADDING = 80
# %%
train_ds_0    = dataset.Tusimple_Train_Padding_Dataset(padding=0)
train_ds_80   = dataset.Tusimple_Train_Padding_Dataset(padding=80)
train_ds_160  = dataset.Tusimple_Train_Padding_Dataset(padding=160)
train_ds_240  = dataset.Tusimple_Train_Padding_Dataset(padding=240)
# %%
img_0 = np.transpose(train_ds_0[0][0], (1,2,0))
img_80 = np.transpose(train_ds_80[0][0], (1,2,0))
img_160 = np.transpose(train_ds_160[0][0], (1,2,0))
img_240 = np.transpose(train_ds_240[0][0], (1,2,0))
# %%
label_0   = train_ds_0[0][1]
label_80  = train_ds_80[0][1]
label_160 = train_ds_160[0][1]
label_240 = train_ds_240[0][1]
# %%
label_lane_img_0 = img_0.numpy().copy()
for one_sample_lane in label_0:

    lane_coordinate = list(zip(one_sample_lane*1280, [x for x in range(160+INIT_PADDING, 720-5+INIT_PADDING, 5)]))
    lane_coordinate = np.int32(lane_coordinate)
    lane_coordinate = lane_coordinate[[2*x for x in range(56)]]
    lane_coordinate = lane_coordinate[lane_coordinate[:,0]>0]
    
    cv2.polylines(label_lane_img_0, [np.int32(lane_coordinate)], isClosed=False, color=(1,0,0), thickness=10)
# %%
label_lane_img_80 = img_80.numpy().copy()
for one_sample_lane in label_80:

    lane_coordinate = list(zip(one_sample_lane*1280, [x for x in range(160+INIT_PADDING, 720-5+INIT_PADDING, 5)]))
    lane_coordinate = np.int32(lane_coordinate)
    lane_coordinate = lane_coordinate[[2*x for x in range(56)]]
    lane_coordinate = lane_coordinate[lane_coordinate[:,0]>0]
    
    cv2.polylines(label_lane_img_80, [np.int32(lane_coordinate)], isClosed=False, color=(1,0,0), thickness=10)
# %%
label_lane_img_160 = img_160.numpy().copy()
for one_sample_lane in label_160:

    lane_coordinate = list(zip(one_sample_lane*1280, [x for x in range(160+INIT_PADDING, 720-5+INIT_PADDING, 5)]))
    lane_coordinate = np.int32(lane_coordinate)
    lane_coordinate = lane_coordinate[[2*x for x in range(56)]]
    lane_coordinate = lane_coordinate[lane_coordinate[:,0]>0]
    
    cv2.polylines(label_lane_img_160, [np.int32(lane_coordinate)], isClosed=False, color=(1,0,0), thickness=10)
# %%
label_lane_img_240 = img_240.numpy().copy()
for one_sample_lane in label_240:

    lane_coordinate = list(zip(one_sample_lane*1280, [x for x in range(160+INIT_PADDING, 720-5+INIT_PADDING, 5)]))
    lane_coordinate = np.int32(lane_coordinate)
    lane_coordinate = lane_coordinate[[2*x for x in range(56)]]
    lane_coordinate = lane_coordinate[lane_coordinate[:,0]>0]
    
    cv2.polylines(label_lane_img_240, [np.int32(lane_coordinate)], isClosed=False, color=(1,0,0), thickness=10)