# %%
# import public package
import torch
import os
import numpy as np
import cv2
# %%
# import custom package
import dataset
import model
import config
# %%
# Set Parameters
RESIZE_HEIGHT = config.RESIZE_HEIGHT
RESIZE_WIDTH  = config.RESIZE_WIDTH

HEIGHT_TOP_CROP    = config.HEIGHT_TOP_CROP
HEIGHT_BOTTOM_CROP = config.HEIGHT_BOTTOM_CROP
NUM_LANE_POINT     = config.NUM_LANE_POINT
ENLENGTHEN_NUM_LANE_POINT = config.ENLENGTHEN_NUM_LANE_POINT
NUM_ARM        = config.NUM_ARM
MODEL_NAME     = config.MODEL_NAME
LEARNING_RATE  = config.LEARNING_RATE
MODEL_SAVE_DIR = config.MODEL_SAVE_DIR
JSON_DIR = str(os.path.join(MODEL_SAVE_DIR, MODEL_NAME, "pred_file.json"))

INIT_PADDING = config.INIT_PADDING
PATCH_WIDTH  = config.PATCH_WIDTH
TILE_HORIZONTAL_NUM = config.TILE_HORIZONTAL_NUM
TILE_VERTICAL_NUM   = config.TILE_VERTICAL_NUM

SET_CLF_THRES = config.SET_CLF_THRES
PT_CLF_THRES  = config.PT_CLF_THRES
# %%
valid_dataset = dataset.Tusimple_Valid_Padding_Dataset(argsort=False)
valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=1, shuffle=False)
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
# remove previous json
if os.path.isfile(JSON_DIR):
    os.remove(JSON_DIR)
# %%
# make json
accuracy_list  = list()
precision_list = list()
recall_list    = list()
f1_score_list  = list()

with torch.no_grad():

    for data_idx, valid_data in enumerate(valid_loader):

        print('\r',f"writing json ...  {data_idx+1}/{len(valid_loader)}", end ="")
        original_img, inputs, label_lanes, binary_label =  valid_data

        pt_reg_outputs = point_regressor(inputs.to(device))
        pt_clf_outputs = point_classifier(inputs.to(device), pt_reg_outputs)

        set_sup_outputs = set_suppressor(pt_reg_outputs, pt_clf_outputs)

        original_img = original_img[0].numpy()
        pt_reg_outputs  = pt_reg_outputs[0].detach().cpu().numpy()
        pt_clf_outputs  = pt_clf_outputs[0].detach().cpu().numpy()
        label_lanes  = label_lanes[0].numpy()
        inputs = inputs[0].numpy()
        inputs = np.transpose(inputs, (1,2,0))
        
        ##### Lane Set Classifier
        not_lanes = list()
        for pred_lane_idx, one_pred_lane_set in enumerate(pt_reg_outputs):
            
            pred_lane_coordinates = list(one_pred_lane_set*RESIZE_WIDTH)
            pred_lane_coordinates = np.int32(pred_lane_coordinates)
            pred_lane_coordinates[np.where(pt_clf_outputs[pred_lane_idx] < PT_CLF_THRES)] = -2
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
                set_classifier_outputs = 0
            else:
                set_classifier_outputs = 1
            
            not_lanes.append(set_classifier_outputs)

        lanes = list()

        for pred_lane_idx, one_pred_lane_set in enumerate(pt_reg_outputs):
            
            if pred_lane_idx in set_sup_outputs:
                continue
            
            if not_lanes[pred_lane_idx] != 1:
                continue
            
            pred_lane_coordinates = list(zip(one_pred_lane_set*RESIZE_WIDTH, [x for x in range(HEIGHT_TOP_CROP, HEIGHT_BOTTOM_CROP-5, 5)]))
            pred_lane_coordinates = np.int32(pred_lane_coordinates)
            pred_lane_coordinates[np.where(pt_clf_outputs[pred_lane_idx] < PT_CLF_THRES), 0] = -2
            pred_lane_coordinates = pred_lane_coordinates[[2*x for x in range(56)]]
            pred_lane_coordinates[pred_lane_coordinates[:,0]<0, 0] = -2
           
            lanes.append(list(pred_lane_coordinates[:,0]))
            h_samples = [x for x in range(HEIGHT_TOP_CROP, HEIGHT_BOTTOM_CROP, 10)]
            raw_file  = valid_dataset.merged_label[data_idx]['raw_file']

        json_line = dict()
        json_line["lanes"]     = lanes
        json_line["h_samples"] = h_samples
        json_line["raw_files"] = raw_file
        json_line["run_time"]  = 0

        with open(JSON_DIR, "a") as f:
            f.write('{'+'"lanes": '+f'{lanes}, '+'"h_samples": '+f'{h_samples}, '+'"raw_file": '+f'"{raw_file}"'+'}\n')
# %%