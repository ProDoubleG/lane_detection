# %%
# import public package
import torch
import os
import numpy as np
import cv2
from PIL import Image
# %%
# import custom package
import dataset
import model
import config
# %%
# Set Parameters
BATCH_SIZE = config.BATCH_SIZE
NUM_ARM    = config.NUM_ARM
MODEL_NAME = config.MODEL_NAME
LEARNING_RATE  = config.LEARNING_RATE
RESIZE_HEIGHT = config.RESIZE_HEIGHT
RESIZE_WIDTH  = config.RESIZE_WIDTH
HEIGHT_TOP_CROP  = config.HEIGHT_TOP_CROP
INIT_PADDING   = config.INIT_PADDING
PATCH_WIDTH    = config.PATCH_WIDTH
MODEL_SAVE_DIR = config.MODEL_SAVE_DIR
TILE_HORIZONTAL_NUM = config.TILE_HORIZONTAL_NUM
TILE_VERTICAL_NUM   = config.TILE_VERTICAL_NUM
HEIGHT_TOP_CROP     = config.HEIGHT_TOP_CROP
HEIGHT_BOTTOM_CROP  = config.HEIGHT_BOTTOM_CROP
PT_CLF_THRES        = config.PT_CLF_THRES
SET_CLF_THRES       = config.SET_CLF_THRES
# %%
# Get dataset
valid_dataset = dataset.Tusimple_Valid_Padding_Dataset()
# %%
# Load model on GPU
point_regressor  = model.LanePointRegressor(num_arm=NUM_ARM)
point_classifier = model.LanePointClassifier(num_arm=NUM_ARM)
set_suppressor   = model.LaneSetSuppressor()
set_classifier   = model.LaneSetClassifier()

point_regressor.load_state_dict( torch.load(os.path.join(MODEL_SAVE_DIR, MODEL_NAME, "point_regressor.pt")))
point_classifier.load_state_dict(torch.load(os.path.join(MODEL_SAVE_DIR, MODEL_NAME, "point_classifier.pt")))
set_classifier.load_state_dict(  torch.load(os.path.join(MODEL_SAVE_DIR, MODEL_NAME, "set_classifier.pt")))

USE_CUDA = torch.cuda.is_available()

if USE_CUDA:
    device = torch.device('cuda:0')
    torch.cuda.set_device(device)
else:
    device = torch.device('cpu')

if USE_CUDA:
    point_regressor.to(device)
    point_classifier.to(device)
    set_classifier.to(device)
# %%
# make inference
model_save_dir = f"{MODEL_SAVE_DIR}/{MODEL_NAME}"
color_list = [(255/255, 0, 0), (0, 255/255, 0),  (0, 0, 255/255),(255/255, 255/255, 0), (0, 255/255, 255/255), (255/255, 0, 255/255), (255, 165, 0), (128/255, 255/255, 0), (0, 0, 128/255), (128/255, 0, 128/255), (255/255, 192/255, 203/255), (128/255, 128/255, 128/255)]

if not os.path.exists(model_save_dir):
    os.makedirs(model_save_dir)

forward_data    = valid_dataset[2510]
right_turn_data = valid_dataset[1302]
left_turn_data  = valid_dataset[1250]

forward_hidden_data     = valid_dataset[2350]
right_turn_hidden_data  = valid_dataset[2278]
left_turn_hidden_data   = valid_dataset[980]

forward_many_lane  = valid_dataset[2082]
forward_not_lane   = valid_dataset[2400]
forward_many_cars  = valid_dataset[2022]

datalist = [
    forward_data,
    right_turn_data,
    left_turn_data,
    forward_hidden_data,
    right_turn_hidden_data,
    left_turn_hidden_data,
    forward_many_lane,
    forward_not_lane,
    forward_many_cars]

img_name_list = [
    "forward_infernce",
    "right_curve_inference",
    "left_curve_inference",
    "forward_obscured_inference",
    "right_curve_obscured_inference",
    "left_curve_obscured_inference",
    "forward_multi_lane_inference",
    "forward_no_lane_inference",
    "forward_multi_car_inference"
]

for idx, data in enumerate(datalist):

    img_folder_path = os.path.join(model_save_dir, img_name_list[idx])
    if not os.path.exists(img_folder_path):
        os.makedirs(img_folder_path)

    original_img, inputs, numeric_label, binary_label = data
    
    inputs = inputs.unsqueeze(dim=0)
    numeric_label = numeric_label.unsqueeze(dim=0)
    binary_label  = binary_label.unsqueeze(dim=0)

    point_reg_outputs = point_regressor(inputs.to(device))
    point_clf_outputs = point_classifier(inputs.to(device), point_reg_outputs)
    set_sup_outputs = set_suppressor(point_reg_outputs, point_clf_outputs)
    # set_clf_outputs = set_classifier()

    original_img = original_img.numpy()
    point_reg_outputs = point_reg_outputs[0].detach().cpu().numpy()
    point_clf_outputs = point_clf_outputs[0].detach().cpu().numpy()
    
    numeric_lanes   = numeric_label[0].numpy()

    #################### plot original image #######################
    img = Image.fromarray(np.array(original_img*255, np.uint8))
    img.save(os.path.join(img_folder_path, f'0_original_img.png'))
    ################################################################


    #################### plot label image ##########################
    label_lane_img = original_img.copy()
    for one_sample_lane in numeric_lanes:

        lane_coordinate = list(zip(one_sample_lane*1280, [x for x in range(HEIGHT_TOP_CROP, HEIGHT_BOTTOM_CROP-5, 5)]))
        lane_coordinate = np.int32(lane_coordinate)
        lane_coordinate = lane_coordinate[[2*x for x in range(56)]]
        lane_coordinate = lane_coordinate[lane_coordinate[:,0]>0]
        
        cv2.polylines(label_lane_img, [np.int32(lane_coordinate)], isClosed=False, color=(1,0,0), thickness=10)

    img = Image.fromarray(np.array(label_lane_img*255, np.uint8))
    img.save(os.path.join(img_folder_path, f'1_label_img.png'))
    #################################################################

    #################### plot point regressor output img ############
    point_reg_output_img = original_img.copy()
    for _idx, one_sample_pred_lane in enumerate(point_reg_outputs):

        lane_coordinate = list(zip(one_sample_pred_lane*1280, [x for x in range(HEIGHT_TOP_CROP, HEIGHT_BOTTOM_CROP-5, 5)]))
        lane_coordinate = np.int32(lane_coordinate)
        lane_coordinate = lane_coordinate[[2*x for x in range(56)]]
        lane_coordinate = lane_coordinate[lane_coordinate[:,0]>0]
        
        cv2.polylines(point_reg_output_img, [np.int32(lane_coordinate)], isClosed=False, color=color_list[_idx], thickness=10)

    img = Image.fromarray(np.array(point_reg_output_img*255, np.uint8))
    img.save(os.path.join(img_folder_path, f'2_point_reg_img.png'))
    ###################################################################

    ############## plot point classifier output img ###################
    point_clf_output_img = original_img.copy()
    for _idx, one_sample_pred_lane in enumerate(point_reg_outputs):
        lane_coordinate = list(zip(one_sample_pred_lane*1280, [x for x in range(HEIGHT_TOP_CROP, HEIGHT_BOTTOM_CROP-5, 5)]))
        lane_coordinate = np.int32(lane_coordinate)
        lane_coordinate[np.where(point_clf_outputs[_idx] < PT_CLF_THRES)] = -2
        lane_coordinate = lane_coordinate[[2*x for x in range(56)]]
        lane_coordinate = lane_coordinate[lane_coordinate[:,0]>0]
        
        cv2.polylines(point_clf_output_img, [np.int32(lane_coordinate)], isClosed=False, color=color_list[_idx], thickness=10)

    img = Image.fromarray(np.array(point_clf_output_img*255, np.uint8))
    img.save(os.path.join(img_folder_path, f'3_point_clf_img.png'))
    #####################################################################

    ################# plot set suppressor output img ####################
    set_sup_output_img = original_img.copy()
    for _idx, one_sample_pred_lane in enumerate(point_reg_outputs):

        if _idx in set_sup_outputs:
            continue
        lane_coordinate = list(zip(one_sample_pred_lane*1280, [x for x in range(HEIGHT_TOP_CROP, HEIGHT_BOTTOM_CROP-5, 5)]))
        lane_coordinate = np.int32(lane_coordinate)
        lane_coordinate[np.where(point_clf_outputs[_idx] < PT_CLF_THRES)] = -2
        lane_coordinate = lane_coordinate[[2*x for x in range(56)]]
        lane_coordinate = lane_coordinate[lane_coordinate[:,0]>0]
        
        cv2.polylines(set_sup_output_img, [np.int32(lane_coordinate)], isClosed=False, color=color_list[_idx], thickness=10)

    img = Image.fromarray(np.array(set_sup_output_img*255, np.uint8))
    img.save(os.path.join(img_folder_path, f'4_set_sup_img.png'))
    #####################################################################

    ########################## Lane Set Classifier ######################
    is_lanes = list()
    for pred_lane_idx, one_pred_lane_set in enumerate(point_reg_outputs):
        
        pred_lane_coordinates = list(one_pred_lane_set*1280)
        pred_lane_coordinates = np.int32(pred_lane_coordinates)
        pred_lane_coordinates[np.where(point_clf_outputs[pred_lane_idx] < PT_CLF_THRES)] = -2
        pred_lane_coordinates = pred_lane_coordinates[[2*x for x in range(56)]]
        h_samples = [x for x in range(HEIGHT_TOP_CROP, HEIGHT_BOTTOM_CROP, 10)]
        
        lane_img = np.zeros((56, PATCH_WIDTH, PATCH_WIDTH,3))
        arranged_lane_img = np.zeros((TILE_VERTICAL_NUM*PATCH_WIDTH,TILE_HORIZONTAL_NUM*PATCH_WIDTH,3))

        for _idx in range(56):
            if pred_lane_coordinates[_idx] == -2:
                lane_img[_idx] = np.zeros((PATCH_WIDTH,PATCH_WIDTH,3))
                
            else:
                segment = original_img[np.clip(int(h_samples[_idx]-(PATCH_WIDTH/2)), 0, RESIZE_HEIGHT):np.clip(int(h_samples[_idx]+(PATCH_WIDTH/2)), 0, RESIZE_HEIGHT), np.clip(int(pred_lane_coordinates[_idx]-(PATCH_WIDTH/2)), 0, RESIZE_WIDTH):np.clip(int(pred_lane_coordinates[_idx]+(PATCH_WIDTH/2)), 0, RESIZE_WIDTH)]
                segment = cv2.resize(segment,(PATCH_WIDTH,PATCH_WIDTH))
                lane_img[_idx] = segment
        
        for _idx in range(56):
            arranged_lane_img[PATCH_WIDTH*int(_idx/TILE_HORIZONTAL_NUM):PATCH_WIDTH*(int(_idx/TILE_HORIZONTAL_NUM)+1), PATCH_WIDTH*int(_idx%TILE_HORIZONTAL_NUM):PATCH_WIDTH*(int(_idx%TILE_HORIZONTAL_NUM)+1)] = lane_img[_idx]
        
        arranged_lane_img = np.transpose(arranged_lane_img, (2,0,1))
        arranged_lane_img = torch.FloatTensor(arranged_lane_img)
        arranged_lane_img = torch.unsqueeze(arranged_lane_img, 0)
        
        set_classifier_outputs = set_classifier(arranged_lane_img.to(device))
        set_classifier_outputs = set_classifier_outputs.detach().cpu().numpy()

        if SET_CLF_THRES > set_classifier_outputs:
            pass
            # set_classifier_outputs = 0
            is_lanes.append(False)
        else:
            # set_classifier_outputs = 1
            is_lanes.append(True)
    #########################################################################

    ############## plot set classifier output ###############################
    set_clf_output_img = original_img.copy()
    for _idx, one_sample_pred_lane in enumerate(point_reg_outputs):

        if _idx in set_sup_outputs:
            continue

        if not is_lanes[_idx]:
            continue

        lane_coordinate = list(zip(one_sample_pred_lane*1280, [x for x in range(HEIGHT_TOP_CROP, HEIGHT_BOTTOM_CROP-5, 5)]))
        lane_coordinate = np.int32(lane_coordinate)
        lane_coordinate[np.where(point_clf_outputs[_idx] < PT_CLF_THRES)] = -2
        lane_coordinate = lane_coordinate[[2*x for x in range(56)]]
        lane_coordinate = lane_coordinate[lane_coordinate[:,0]>0]
        
        cv2.polylines(set_clf_output_img, [np.int32(lane_coordinate)], isClosed=False, color=color_list[_idx], thickness=10)

    img = Image.fromarray(np.array(set_clf_output_img*255, np.uint8))
    img.save(os.path.join(img_folder_path, f'5_set_clf_img.png'))
    ###########################################################################