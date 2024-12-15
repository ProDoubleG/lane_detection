# %%
# import public packages
import torch
import os
import numpy as np
import cv2
import torch
from PIL import Image
# %%
# import custom packages
import dataset
import model
import config
import utils
# %%
# fix seed
utils.setting.fix_seed(config.SEED)
# %%
NUM_ARM = config.NUM_ARM
MODEL_NAME = config.MODEL_NAME
LEARNING_RATE = config.LEARNING_RATE
INIT_PADDING = config.INIT_PADDING
HEIGHT_TOP_CROP = config.HEIGHT_TOP_CROP
HEIGHT_BOTTOM_CROP = config.HEIGHT_BOTTOM_CROP
PATCH_WIDTH = config.PATCH_WIDTH
TILE_HORIZONTAL_NUM = config.TILE_HORIZONTAL_NUM
TILE_VERTICAL_NUM   = config.TILE_VERTICAL_NUM
CLF_PATCH_RADIUS = config.CLF_PATCH_RADIUS
MODEL_SAVE_DIR   = config.MODEL_SAVE_DIR
SET_CLF_DS_DIR = config.SET_CLF_DS_DIR
PT_CLF_THRES   = config.PT_CLF_THRES
# %%
train_ds    = dataset.Tusimple_Train_Padding_Dataset(padding=0)
train_loader = torch.utils.data.DataLoader(train_ds, batch_size=1, shuffle=False)
# %%
# Load model on GPU
point_reg = model.LanePointRegressor(num_arm=NUM_ARM)
point_clf = model.LanePointClassifier(num_arm=NUM_ARM)
set_sup   = model.LaneSetSuppressor()

point_reg.load_state_dict(torch.load(os.path.join(MODEL_SAVE_DIR, MODEL_NAME, "point_regressor.pt")))
point_clf.load_state_dict(torch.load(os.path.join(MODEL_SAVE_DIR, MODEL_NAME, "point_classifier.pt")))
USE_CUDA = torch.cuda.is_available()

if USE_CUDA:
    device = torch.device('cuda:0')
    torch.cuda.set_device(device)
else:
    device = torch.device('cpu')

if USE_CUDA:
    point_reg.to(device)
    point_clf.to(device)
    set_sup.to(device)
# %%
# create data dir
set_clf_false_dir = f'{SET_CLF_DS_DIR}/{MODEL_NAME}/false'
set_clf_true_dir  = f'{SET_CLF_DS_DIR}/{MODEL_NAME}/true'

if os.path.exists(set_clf_true_dir):
    os.system(f"rm -rf {set_clf_true_dir}")
os.makedirs(set_clf_true_dir)

if os.path.exists(set_clf_false_dir):
    os.system(f"rm -rf {set_clf_false_dir}")
os.makedirs(set_clf_false_dir)
# %%
# get prediciton
with torch.no_grad():

    for loader_idx, data in enumerate(train_loader):

        print('\r',f"writing data for classifier ...  {loader_idx+1}/{len(train_loader)}", end ="")
        inputs, label_lanes, binary_label =  data

        point_reg_outputs = point_reg(inputs.to(device))
        point_clf_outputs = point_clf(inputs.to(device), point_reg_outputs)
        set_sup_outputs   = set_sup(point_clf_outputs, point_reg_outputs)
        inputs = inputs[0].numpy()
        inputs = np.transpose(inputs, (1,2,0))

        point_reg_outputs  = point_reg_outputs[0].detach().cpu().numpy()
        point_clf_outputs  = point_clf_outputs[0].detach().cpu().numpy()

        label_lanes  = label_lanes[0].numpy()

        # draw label with 50-radius 
        label_lanes_img = np.zeros((720+INIT_PADDING, 1280))
        for label_lane_idx, one_label_lane_set in enumerate(label_lanes):
            
            if np.max(one_label_lane_set) == -1:

                continue
            label_lane_coordinate = list(zip(one_label_lane_set*1280, [x for x in range(HEIGHT_TOP_CROP+INIT_PADDING, HEIGHT_BOTTOM_CROP+INIT_PADDING-5, 5)]))
            label_lane_coordinate = np.int32(label_lane_coordinate)
            label_lane_coordinate = label_lane_coordinate[label_lane_coordinate[:,0]>0]
            # label_lanes_img[label_lane_coordinate[:,1], label_lane_coordinate[:,0]] = 1
            for coordinate in label_lane_coordinate:
                cv2.circle(label_lanes_img, (coordinate[0], coordinate[1]), radius=CLF_PATCH_RADIUS, color=1, thickness=-1)
        
        
        # draw pred on label-img
        pred_lanes_img = np.zeros((720+INIT_PADDING, 1280))
        pred_to_label_list = list()

        for pred_lane_idx, one_pred_lane_set in enumerate(point_reg_outputs):


            pred_lane_coordinates = list(zip(one_pred_lane_set*1280, [x for x in range(HEIGHT_TOP_CROP+INIT_PADDING, HEIGHT_BOTTOM_CROP+INIT_PADDING-5, 5)]))
            pred_lane_coordinates = np.int32(pred_lane_coordinates)
            pred_lane_coordinates[np.where(point_clf_outputs[pred_lane_idx] < config.PT_CLF_THRES)] = -2
            pred_lane_coordinates = pred_lane_coordinates[[2*x for x in range(56)]]
            pred_lane_coordinates = pred_lane_coordinates[pred_lane_coordinates[:,0]>0]
            score = label_lanes_img[pred_lane_coordinates[:,1], pred_lane_coordinates[:,0]]

            # #
            # test = np.zeros((720+INIT_PADDING, 1280))
            # for coor in pred_lane_coordinates:
            #     cv2.circle(test, (coor[0], coor[1]), radius=10, thickness=-1, color=1)

            # #
            if len(score) < 5:
                pred_to_label_list.append(False)
                continue

            if sum(score)/len(score) > 0.5:
                pred_to_label_list.append(True)
            else:
                pred_to_label_list.append(False)
        
        
        # draw image
        for pred_lane_idx, one_pred_lane_set in enumerate(point_reg_outputs):
            
            if pred_lane_idx in set_sup_outputs:

                pass

            else:
                pred_lane_coordinates = list(one_pred_lane_set*1280)
                pred_lane_coordinates = np.int32(pred_lane_coordinates)
                pred_lane_coordinates[np.where(point_clf_outputs[pred_lane_idx] < PT_CLF_THRES)] = -2
                pred_lane_coordinates = pred_lane_coordinates[[2*x for x in range(56)]]
                h_samples = [x for x in range(HEIGHT_TOP_CROP, HEIGHT_BOTTOM_CROP, 10)]
                
                if len(pred_lane_coordinates) < 5:
                    continue
                
                assert TILE_VERTICAL_NUM*TILE_HORIZONTAL_NUM == 56

                lane_img = np.zeros((56, PATCH_WIDTH, PATCH_WIDTH,3))
                arranged_lane_img = np.zeros((TILE_VERTICAL_NUM*PATCH_WIDTH,TILE_HORIZONTAL_NUM*PATCH_WIDTH,3))
                for _idx in range(56):
                    if pred_lane_coordinates[_idx] == -2:
                        lane_img[_idx] = np.zeros((PATCH_WIDTH,PATCH_WIDTH,3))
                        
                    else:
                        segment = inputs[np.clip(int(h_samples[_idx]-(PATCH_WIDTH/2)+INIT_PADDING), 0, 720+INIT_PADDING):np.clip(int(h_samples[_idx]+(PATCH_WIDTH/2)+INIT_PADDING), 0, 720+INIT_PADDING), np.clip(int(pred_lane_coordinates[_idx]-(PATCH_WIDTH/2)), 0, 1280):np.clip(int(pred_lane_coordinates[_idx]+(PATCH_WIDTH/2)), 0, 1280)]
                        segment = cv2.resize(segment,(PATCH_WIDTH,PATCH_WIDTH))
                        lane_img[_idx] = segment
                
                for _idx in range(56):
                    arranged_lane_img[PATCH_WIDTH*int(_idx/TILE_HORIZONTAL_NUM):PATCH_WIDTH*(int(_idx/TILE_HORIZONTAL_NUM)+1), PATCH_WIDTH*int(_idx%TILE_HORIZONTAL_NUM):PATCH_WIDTH*(int(_idx%TILE_HORIZONTAL_NUM)+1)] = lane_img[_idx]

                if pred_to_label_list[pred_lane_idx]:
                    img = Image.fromarray(np.array(arranged_lane_img*255, np.uint8))
                    img.save(os.path.join(set_clf_true_dir, f'{loader_idx}_{pred_lane_idx}.png'))

                elif not pred_to_label_list[pred_lane_idx]:
                    img = Image.fromarray(np.array(arranged_lane_img*255, np.uint8))
                    img.save(os.path.join(set_clf_false_dir, f'{loader_idx}_{pred_lane_idx}.png'))