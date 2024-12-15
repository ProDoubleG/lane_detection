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
import config
import utils.setting
# %%
# fix seed
utils.setting.fix_seed(config.SEED)
# %%
# import parameters
BATCH_SIZE = config.BATCH_SIZE
NUM_ARM = config.NUM_ARM
MODEL_NAME = config.MODEL_NAME
LEARNING_RATE = config.LEARNING_RATE
INIT_PADDING = config.INIT_PADDING
PATIENCE_LIMIT = config.PATIENCE_LIMIT
PT_CLF_THRES = config.PT_CLF_THRES
# MODEL_SAVE_DIR = config.MODEL_SAVE_DIR
MODEL_SAVE_DIR = "/home/tusimple_data/model_analysis/"
# %%
# load dataset with augmentation
train_ds_0    = dataset.Tusimple_Train_Padding_Dataset(padding=0)
train_ds_80   = dataset.Tusimple_Train_Padding_Dataset(padding=80)
train_ds_160  = dataset.Tusimple_Train_Padding_Dataset(padding=160)
train_ds_240  = dataset.Tusimple_Train_Padding_Dataset(padding=240)

train_ds = train_ds_0  + train_ds_80 + train_ds_160 + train_ds_240

train_loader = torch.utils.data.DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
# %%
# Loader test
for i in train_loader:
    input, lanes, binary_lanes = i
    break
# %%
# Plot dataset
numeric_lanes  = lanes[0].numpy()
numeric_lanes  = numeric_lanes[np.max(numeric_lanes, axis=1)!=-1]
original_img   = input[0].detach().numpy()
original_img   = np.transpose(original_img, (1,2,0))

label_lane_img = original_img.copy()
for one_sample_lane in numeric_lanes:

    lane_coordinate = list(zip(one_sample_lane*1280, [x for x in range(160+INIT_PADDING, 720-5+INIT_PADDING, 5)]))
    lane_coordinate = np.int32(lane_coordinate)
    lane_coordinate = lane_coordinate[[2*x for x in range(56)]]
    lane_coordinate = lane_coordinate[lane_coordinate[:,0]>0]
            
    cv2.polylines(label_lane_img, [np.int32(lane_coordinate)], isClosed=False, color=(1,0,0), thickness=10)
# %%
# Test Model Test
# point_reg = model.LanePointRegressor()
# reg_output  = point_reg(input)

# point_clf = model.LanePointClassifier()
# clf_output = point_clf(input, reg_output)

# del point_reg, reg_output, point_clf, clf_output
# %%
# Load model on GPU
point_reg = model.LanePointRegressor(num_arm=NUM_ARM)
point_clf = model.LanePointClassifier(num_arm=NUM_ARM)

USE_CUDA = torch.cuda.is_available()

if USE_CUDA:
    device = torch.device('cuda:0')
    torch.cuda.set_device(device)
else:
    device = torch.device('cpu')

if USE_CUDA:
    point_reg.to(device)
    point_clf.to(device)
# %%
# Define BCE loss of Classifier model
class LaneBCELoss(torch.nn.Module):
    def __init__(self):
        super(LaneBCELoss, self).__init__()

    def forward(self, model_output, targets, assigned_idx_list, smooth=1):  
        
        tot_BCE_loss = 0.0 
        for batch_idx in range(len(targets)):
            assigned_idx = 0
            
            target_grid = targets[batch_idx]       
            output_grid  = model_output[batch_idx] 
            assigned_idx =  assigned_idx_list[batch_idx]

            BCE_loss  = 0
            idx = 0
            lane_cnt = 0
            for single_target_grid in target_grid:

                if torch.sum(single_target_grid)==0:
                    pass
                else:
                    lane_cnt += 1
                    
                    single_output_grid  = output_grid[assigned_idx[idx]]
                    
                    single_target_grid = single_target_grid.view(-1)
                    single_output_grid = single_output_grid.view(-1)

                    BCE = torch.nn.functional.binary_cross_entropy(single_output_grid, single_target_grid, reduction='mean')
                    
                    BCE_loss += BCE

                    idx += 1

            BCE_loss = BCE_loss/lane_cnt
            tot_BCE_loss += BCE_loss

        tot_BCE_loss = tot_BCE_loss/(len(targets))
        
        return tot_BCE_loss
# %%
# Define L1 loss of Regressor model
class Grid_L1(torch.nn.Module):
    def __init__(self):
        super(Grid_L1, self).__init__()
        self.positive_grid_mask = None
        self.weight = None
    def forward(self, model_output, target_lanes):
        
        output_lanes = model_output
        tot_L1_loss = 0
        batch_lane_idx = list()
        for batch_idx in range(len(target_lanes)):

            target_grid = target_lanes[batch_idx]
            input_grid  = output_lanes[batch_idx]
        
            L1_loss  = 0
            lane_cnt = 0
            
            paired_input_lane_idx = list()

            for label_lane_idx, single_target_grid in enumerate(target_grid):
                
                if torch.max(single_target_grid)==-1:
                    pass

                else:
                    lane_cnt += 1
                    # L1 = torch.abs(single_target_grid - input_grid)
                    # positive_mask = torch.where(single_target_grid!=-1)
                    positive_grid_mask = torch.ones_like(single_target_grid, requires_grad = False)
                    positive_grid_mask[torch.where(single_target_grid==-1)] = 0

                    
                    L1 = torch.abs(single_target_grid - input_grid)*positive_grid_mask # matrix element-wise multiplication

                    # weight...
                    # L1 = L1*torch.tensor([1+ 1/x for x in range(1,112)]).to(device) <-- this is not right

                    # L1 = torch.mean(L1, dim=1)/torch.sum(positive_grid_mask, dim=1)
                    L1 = torch.sum(L1, dim=1)/torch.sum(positive_grid_mask)

                    for argidx in range(len(L1)):
                        min_idx = torch.argsort(L1)[argidx]
                        if min_idx in paired_input_lane_idx:
                            pass
                        else:
                            paired_input_lane_idx.append(min_idx)
                            break
                    
                    L1_loss += L1[min_idx]

            batch_lane_idx.append(paired_input_lane_idx)
            L1_loss = L1_loss/lane_cnt
            tot_L1_loss += L1_loss

        tot_L1_loss = tot_L1_loss/(len(target_lanes))

        return tot_L1_loss, batch_lane_idx # each indicies represent the input and values are the paired label index
# %%
# Get optimizer and loss function class
reg_optimizer    = torch.optim.Adam(point_reg.parameters(), lr=LEARNING_RATE)
clf_optimizer    = torch.optim.Adam(point_clf.parameters(),  lr=LEARNING_RATE)
l1_loss_fn         = Grid_L1()
cross_entropy_loss = LaneBCELoss()
# %%
# Train Regressor and Classifier
EPOCH = 200

clf_loss_list  = list()
reg_loss_list    = list()
clf_train_threshold  = 100
best_loss = 1e4
best_epoch = 0

current_patience = 0

for epoch in range(EPOCH):
    print(f"---------------Epoch : {epoch+1}/{EPOCH}--------------------")
    train_clf_loss = 0.0
    train_reg_loss = 0.0
    clf_train_idx = 0
    
    for train_idx, data in enumerate(train_loader, 0):
        reg_optimizer.zero_grad()
        clf_optimizer.zero_grad()

        print('\r',f"training {train_idx+1}/{len(train_loader)}, CE_loss: {train_clf_loss/(clf_train_idx+1):0.5f} L1_loss: {train_reg_loss/(train_idx+1):0.5f}", end =" ")

        inputs, lanes, binary_lanes = data
        
        reg_outputs = point_reg(inputs.to(device))
        
        reg_loss, assigned_lane_idx = l1_loss_fn(reg_outputs, lanes.to(device))
        reg_loss.backward()
        reg_optimizer.step()
        train_reg_loss += reg_loss.item()
        
        if clf_train_threshold < 0.01:
            clf_train_idx += 1
            reg_outputs = reg_outputs.detach()
            clf_outputs = point_clf(inputs.to(device), reg_outputs.to(device))
            
            clf_loss    = cross_entropy_loss(clf_outputs, binary_lanes.to(device), assigned_lane_idx)
            clf_loss.backward()
            clf_optimizer.step()
            train_clf_loss += clf_loss.item()

        else:
            pass
        
    if PATIENCE_LIMIT > current_patience:
        
        if train_reg_loss/(train_idx+1) < best_loss*0.9:
            
            # save model

            best_loss = train_reg_loss/(train_idx+1)
            best_epoch = epoch+1
            best_reg_model = point_reg
            best_clf_model = point_clf

            current_patience = 0
        else:
            current_patience += 1
    
    else:
        break
        
    clf_train_threshold = train_reg_loss/(train_idx+1)

    # print("clf_train_threshold: ", clf_train_threshold)
    clf_loss_list.append(train_clf_loss/(clf_train_idx+1))
    reg_loss_list.append(train_reg_loss/(train_idx+1))
    print("")
# %%
# save model
model_save_dir = f"{MODEL_SAVE_DIR}/{MODEL_NAME}"

if not os.path.exists(model_save_dir):
    os.makedirs(model_save_dir)

torch.save(best_reg_model.state_dict(), os.path.join(model_save_dir, "point_regressor.pt"))
torch.save(best_clf_model.state_dict(), os.path.join(model_save_dir, "point_classifier.pt"))

plt.plot(reg_loss_list)
plt.xlabel("epoches")
plt.ylabel("L1 loss")
plt.savefig(os.path.join(model_save_dir, "L1_loss.png"))

plt.plot(clf_loss_list)
plt.xlabel("epoches")
plt.ylabel("bce loss")
plt.savefig(os.path.join(model_save_dir, "BCE_loss.png"))

plt.clf()
# %%
point_reg = model.LanePointRegressor(num_arm=NUM_ARM)
point_clf = model.LanePointClassifier(num_arm=NUM_ARM)

point_reg.load_state_dict(torch.load(os.path.join("/home/tusimple_data/model_analysis", MODEL_NAME, "point_regressor.pt")))
point_clf.load_state_dict(torch.load(os.path.join("/home/tusimple_data/model_analysis", MODEL_NAME, "point_classifier.pt")))
USE_CUDA = torch.cuda.is_available()

if USE_CUDA:
    device = torch.device('cuda:0')
    torch.cuda.set_device(device)
else:
    device = torch.device('cpu')

if USE_CUDA:
    point_reg.to(device)
    point_clf.to(device)
# %%
# SET_CLF_THRESHOLD = 0.9

model_save_dir = f"/home/tusimple_data/model_analysis/{MODEL_NAME}"

color_list = [(255/255, 0, 0), (0, 255/255, 0),  (0, 0, 255/255),(255/255, 255/255, 0), (0, 255/255, 255/255), (255/255, 0, 255/255), (255, 165, 0), (128/255, 255/255, 0), (0, 0, 128/255), (128/255, 0, 128/255), (255/255, 192/255, 203/255), (128/255, 128/255, 128/255)]

if not os.path.exists(model_save_dir):
    os.makedirs(model_save_dir)

valid_dataset = dataset.Tusimple_Valid_Padding_Dataset()

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

    point_reg_outputs = point_reg(inputs.to(device))
    point_clf_outputs = point_clf(inputs.to(device), point_reg_outputs)

    original_img = original_img.numpy()
    point_reg_outputs = point_reg_outputs[0].detach().cpu().numpy()
    point_clf_outputs = point_clf_outputs[0].detach().cpu().numpy()
    # set_sup_outputs   = set_sup_outputs.detach().cpu().numpy()
    numeric_lanes   = numeric_label[0].numpy()

    # original_img = np.transpose(original_img, (1,2,0))
    label_lane_img = original_img.copy()
    
    numeric_lanes  = numeric_lanes[np.max(numeric_lanes, axis=1)!=-1]

    ## plot original image
    label_lane_img = original_img.copy()

    img = Image.fromarray(np.array(original_img*255, np.uint8))
    img.save(os.path.join(img_folder_path, f'0_original_img.png'))
    ##
    label_lane_img = original_img.copy()
    for one_sample_lane in numeric_lanes:

        lane_coordinate = list(zip(one_sample_lane*1280, [x for x in range(160, 720-5, 5)]))
        lane_coordinate = np.int32(lane_coordinate)
        lane_coordinate = lane_coordinate[[2*x for x in range(56)]]
        lane_coordinate = lane_coordinate[lane_coordinate[:,0]>0]
        
        cv2.polylines(label_lane_img, [np.int32(lane_coordinate)], isClosed=False, color=(1,0,0), thickness=10)

    img = Image.fromarray(np.array(label_lane_img*255, np.uint8))
    img.save(os.path.join(img_folder_path, f'1_label_img.png'))

    # plot point regressor output img
    point_reg_output_img = original_img.copy()
    for _idx, one_sample_pred_lane in enumerate(point_reg_outputs):

        lane_coordinate = list(zip(one_sample_pred_lane*1280, [x for x in range(160, 720-5, 5)]))
        lane_coordinate = np.int32(lane_coordinate)
        lane_coordinate = lane_coordinate[[2*x for x in range(56)]]
        lane_coordinate = lane_coordinate[lane_coordinate[:,0]>0]
        
        cv2.polylines(point_reg_output_img, [np.int32(lane_coordinate)], isClosed=False, color=color_list[_idx], thickness=10)

    img = Image.fromarray(np.array(point_reg_output_img*255, np.uint8))
    img.save(os.path.join(img_folder_path, f'2_point_reg_img.png'))

    point_clf_output_img = original_img.copy()
    for _idx, one_sample_pred_lane in enumerate(point_reg_outputs):
        lane_coordinate = list(zip(one_sample_pred_lane*1280, [x for x in range(160, 720-5, 5)]))
        lane_coordinate = np.int32(lane_coordinate)
        lane_coordinate[np.where(point_clf_outputs[_idx] < PT_CLF_THRES)] = -2
        lane_coordinate = lane_coordinate[[2*x for x in range(56)]]
        lane_coordinate = lane_coordinate[lane_coordinate[:,0]>0]
        
        cv2.polylines(point_clf_output_img, [np.int32(lane_coordinate)], isClosed=False, color=color_list[_idx], thickness=10)

    img = Image.fromarray(np.array(point_clf_output_img*255, np.uint8))
    img.save(os.path.join(img_folder_path, f'3_point_clf_img.png'))