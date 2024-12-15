# import public package
import matplotlib.pyplot
import torch
import json
import cv2
import numpy
import os
from itertools import chain, repeat
import matplotlib

# import custom package
import config
RESIZE_HEIGHT = config.RESIZE_HEIGHT
RESIZE_WIDTH  = config.RESIZE_WIDTH
INIT_PADDING = config.INIT_PADDING

HEIGHT_TOP_CROP = 160
HEIGHT_BOTTOM_CROP = 720
NUM_LANE_POINT = len(range(HEIGHT_TOP_CROP, HEIGHT_BOTTOM_CROP, 10))
ENLENGTHEN_NUM_LANE_POINT = len(range(HEIGHT_TOP_CROP, HEIGHT_BOTTOM_CROP-5, 5))

TUSIMPLE_DIR = "/home/tusimple_data/TUSimple"
TUSIMPLE_FALSE_LANE_DS_DIR = "/home/tusimple_data/classifier"
MODEL_NAME = config.MODEL_NAME
NUM_ARM = 6

class Tusimple_Train_Dataset(torch.utils.data.Dataset):
    def __init__(self):
        super(Tusimple_Train_Dataset, self).__init__()

        self.TUSIMPLE_DIR = TUSIMPLE_DIR
        self.resize_shape = (RESIZE_HEIGHT, RESIZE_WIDTH)

        self.merged_label = []
        label_list = [x for x in os.listdir(os.path.join(self.TUSIMPLE_DIR, 'train_set')) if 'json' in x]
        
        for label in label_list:
            with open(os.path.join(self.TUSIMPLE_DIR, 'train_set', label), 'r') as f:
                for line in f:
                    self.merged_label.append(json.loads(line.rstrip()))

    def __getitem__(self, index):
        
        raw_file = self.merged_label[index]["raw_file"]
        lanes = numpy.array(self.merged_label[index]["lanes"])
        h_samples = numpy.array(self.merged_label[index]["h_samples"])

        enranged_lanes = numpy.zeros((len(lanes), 56))

        if numpy.min(h_samples) != 160:

            h_samples  = numpy.array([x for x in range(HEIGHT_TOP_CROP, HEIGHT_BOTTOM_CROP, 10)])

            for lane_idx in range(len(lanes)):
                enranged_lanes[lane_idx] = numpy.append([-2 for _ in range(160, 240, 10)], lanes[lane_idx], axis = 0)

        else:
            enranged_lanes = lanes
        
        raw_file = self.merged_label[index]["raw_file"]

        input_color_img = cv2.imread(os.path.join(self.TUSIMPLE_DIR,'train_set',raw_file),cv2.IMREAD_COLOR)
        

        input_color_img = cv2.resize(input_color_img, (self.resize_shape[1], self.resize_shape[0]))
        input_color_img = cv2.cvtColor(input_color_img, cv2.COLOR_RGB2BGR)
        input_color_img = numpy.array(input_color_img)/255.0

        label_lane = numpy.zeros((6,NUM_LANE_POINT))
        for idx in range(len(enranged_lanes)):
            label_lane[idx] = enranged_lanes[idx]

        label_lane[label_lane==0] = -2
        
        label_enranged_lane = numpy.empty((6, ENLENGTHEN_NUM_LANE_POINT))

        for _idx in range(ENLENGTHEN_NUM_LANE_POINT):
            
            if _idx % 2 == 0:
                label_enranged_lane[:, _idx] = label_lane[:,_idx//2]
            else:
                label_enranged_lane[:, _idx] = (label_lane[:,_idx//2] + label_lane[:,(_idx//2)+1])/2

        for _lane_idx in range(6):

            if numpy.max(label_lane[_lane_idx]) == -2:
                label_enranged_lane[_lane_idx, :] = -1
                pass
            else:

                is_lane = numpy.where(label_lane[_lane_idx] > 0)[0]
                
                if is_lane[0] > 0:
                    label_enranged_lane[_lane_idx][:is_lane[0]*2] = -1
                else:
                    pass

                if is_lane[-1] < NUM_LANE_POINT-1:
                    label_enranged_lane[_lane_idx][is_lane[-1]*2+1:] = -1
                else:
                    pass

        label_enranged_lane = numpy.clip(label_enranged_lane, 0, 1280)
        label_enranged_lane = label_enranged_lane/1280.0

        label_enranged_lane[label_enranged_lane==0] = -1

        label_binary_lane = numpy.zeros_like(label_enranged_lane)

        label_binary_lane[numpy.where(label_enranged_lane>0)] = 1

        input_color_img = numpy.transpose(input_color_img, (2,0,1))

        return torch.FloatTensor(input_color_img), torch.FloatTensor(label_enranged_lane), torch.FloatTensor(label_binary_lane)
    def __len__(self): 
        return len(self.merged_label)
    
class Tusimple_MultiScale_Train_Dataset(torch.utils.data.Dataset):
    def __init__(self):
        super(Tusimple_MultiScale_Train_Dataset, self).__init__()

        self.TUSIMPLE_DIR = TUSIMPLE_DIR
        self.resize_shape = (RESIZE_HEIGHT, RESIZE_WIDTH)

        self.merged_label = []
        label_list = [x for x in os.listdir(os.path.join(self.TUSIMPLE_DIR, 'train_set')) if 'json' in x]
        
        for label in label_list:
            with open(os.path.join(self.TUSIMPLE_DIR, 'train_set', label), 'r') as f:
                for line in f:
                    self.merged_label.append(json.loads(line.rstrip()))

    def __getitem__(self, index):
        
        raw_file = self.merged_label[index]["raw_file"]
        lanes = numpy.array(self.merged_label[index]["lanes"])
        h_samples = numpy.array(self.merged_label[index]["h_samples"])

        enranged_lanes = numpy.zeros((len(lanes), 56))

        if numpy.min(h_samples) != 160:

            h_samples  = numpy.array([x for x in range(HEIGHT_TOP_CROP, HEIGHT_BOTTOM_CROP, 10)])

            for lane_idx in range(len(lanes)):
                enranged_lanes[lane_idx] = numpy.append([-2 for _ in range(160, 240, 10)], lanes[lane_idx], axis = 0)

        else:
            enranged_lanes = lanes
        
        raw_file = self.merged_label[index]["raw_file"]

        input_color_img = cv2.imread(os.path.join(self.TUSIMPLE_DIR,'train_set',raw_file),cv2.IMREAD_COLOR)

        input_color_img_scale_1on1 =  cv2.resize(input_color_img, (self.resize_shape[1],        self.resize_shape[0]))
        input_color_img_scale_1on2 =  cv2.resize(input_color_img, (int(self.resize_shape[1]/2), int(self.resize_shape[0]/2)))
        input_color_img_scale_1on4 =  cv2.resize(input_color_img, (int(self.resize_shape[1]/4), int(self.resize_shape[0]/4)))
        input_color_img_scale_1on8 =  cv2.resize(input_color_img, (int(self.resize_shape[1]/8), int(self.resize_shape[0]/8)))

        input_color_img_scale_1on1 = cv2.cvtColor(input_color_img_scale_1on1, cv2.COLOR_RGB2BGR)
        input_color_img_scale_1on2 = cv2.cvtColor(input_color_img_scale_1on2, cv2.COLOR_RGB2BGR)
        input_color_img_scale_1on4 = cv2.cvtColor(input_color_img_scale_1on4, cv2.COLOR_RGB2BGR)
        input_color_img_scale_1on8 = cv2.cvtColor(input_color_img_scale_1on8, cv2.COLOR_RGB2BGR)

        input_color_img_scale_1on1 = numpy.array(input_color_img_scale_1on1)/255.0
        input_color_img_scale_1on2 = numpy.array(input_color_img_scale_1on2)/255.0
        input_color_img_scale_1on4 = numpy.array(input_color_img_scale_1on4)/255.0
        input_color_img_scale_1on8 = numpy.array(input_color_img_scale_1on8)/255.0

        label_lane = numpy.zeros((6,NUM_LANE_POINT))
        for idx in range(len(enranged_lanes)):
            label_lane[idx] = enranged_lanes[idx]

        label_lane[label_lane==0] = -2
        
        label_enranged_lane = numpy.empty((6, ENLENGTHEN_NUM_LANE_POINT))

        for _idx in range(ENLENGTHEN_NUM_LANE_POINT):
            
            if _idx % 2 == 0:
                label_enranged_lane[:, _idx] = label_lane[:,_idx//2]
            else:
                label_enranged_lane[:, _idx] = (label_lane[:,_idx//2] + label_lane[:,(_idx//2)+1])/2

        for _lane_idx in range(6):

            if numpy.max(label_lane[_lane_idx]) == -2:
                label_enranged_lane[_lane_idx, :] = -1
                pass
            else:

                is_lane = numpy.where(label_lane[_lane_idx] > 0)[0]
                
                if is_lane[0] > 0:
                    label_enranged_lane[_lane_idx][:is_lane[0]*2] = -1
                else:
                    pass

                if is_lane[-1] < NUM_LANE_POINT-1:
                    label_enranged_lane[_lane_idx][is_lane[-1]*2+1:] = -1
                else:
                    pass

        label_enranged_lane = numpy.clip(label_enranged_lane, 0, 1280)
        label_enranged_lane = label_enranged_lane/1280.0

        label_enranged_lane[label_enranged_lane==0] = -1

        label_binary_lane = numpy.zeros_like(label_enranged_lane)

        label_binary_lane[numpy.where(label_enranged_lane>0)] = 1

        input_color_img_scale_1on1 = numpy.transpose(input_color_img_scale_1on1, (2,0,1))
        input_color_img_scale_1on2 = numpy.transpose(input_color_img_scale_1on2, (2,0,1))
        input_color_img_scale_1on4 = numpy.transpose(input_color_img_scale_1on4, (2,0,1))
        input_color_img_scale_1on8 = numpy.transpose(input_color_img_scale_1on8, (2,0,1))

        return torch.FloatTensor(input_color_img_scale_1on1), torch.FloatTensor(input_color_img_scale_1on2), torch.FloatTensor(input_color_img_scale_1on4), torch.FloatTensor(input_color_img_scale_1on8), torch.FloatTensor(label_enranged_lane), torch.FloatTensor(label_binary_lane)
    def __len__(self): 
        return len(self.merged_label)


class Tusimple_Train_Padding_Dataset(torch.utils.data.Dataset):
    def __init__(self, padding=0, init_padding=INIT_PADDING):
        super(Tusimple_Train_Padding_Dataset, self).__init__()


        self.TUSIMPLE_DIR = TUSIMPLE_DIR
        self.resize_shape = (RESIZE_HEIGHT, RESIZE_WIDTH)
        self.init_padding  = init_padding
        self.padding      = padding

        self.merged_label = list()
        label_list = [x for x in os.listdir(os.path.join(self.TUSIMPLE_DIR, 'train_set')) if 'json' in x]
        
        for label in label_list:
            with open(os.path.join(self.TUSIMPLE_DIR, 'train_set', label), 'r') as f:
                for line in f:
                    self.merged_label.append(json.loads(line.rstrip()))

    def __getitem__(self, index):
        
        raw_file = self.merged_label[index]["raw_file"]
        lanes = numpy.array(self.merged_label[index]["lanes"])
        h_samples = numpy.array(self.merged_label[index]["h_samples"])

        enranged_lanes = numpy.zeros((len(lanes), 56))

        if numpy.min(h_samples) != 160:

            h_samples  = numpy.array([x for x in range(HEIGHT_TOP_CROP, HEIGHT_BOTTOM_CROP, 10)])

            for lane_idx in range(len(lanes)):
                enranged_lanes[lane_idx] = numpy.append([-2 for _ in range(160, 240, 10)], lanes[lane_idx], axis = 0)

        else:
            enranged_lanes = lanes
        
        raw_file = self.merged_label[index]["raw_file"]

        input_color_img = cv2.imread(os.path.join(self.TUSIMPLE_DIR,'train_set',raw_file),cv2.IMREAD_COLOR)
 
        input_color_img = cv2.resize(input_color_img, (self.resize_shape[1], self.resize_shape[0]))
        input_color_img = cv2.cvtColor(input_color_img, cv2.COLOR_RGB2BGR)
        input_color_img = numpy.array(input_color_img)/255.0

        
        input_padded_img = numpy.zeros((720+self.init_padding, 1280, 3))
        
        input_padded_img[self.init_padding+self.padding:] = input_color_img[0:720-self.padding]

        label_lane = numpy.zeros((6,NUM_LANE_POINT))
        for idx in range(len(enranged_lanes)):
            label_lane[idx] = enranged_lanes[idx]
        
        # padding label
  
        label_lane = numpy.delete(label_lane, tuple([x for x in range(-self.padding//10, 0, 1)]), axis=1)
 
        padding_lable = numpy.zeros((6, self.padding//10))
        label_lane = numpy.concatenate((padding_lable, label_lane), axis=1)

        label_lane[label_lane==0] = -2

        label_enranged_lane = numpy.empty((6, ENLENGTHEN_NUM_LANE_POINT))
        
        for _idx in range(ENLENGTHEN_NUM_LANE_POINT):
            
            if _idx % 2 == 0:
                label_enranged_lane[:, _idx] = label_lane[:,_idx//2]
            else:
                label_enranged_lane[:, _idx] = (label_lane[:,_idx//2] + label_lane[:,(_idx//2)+1])/2

        for _lane_idx in range(6):

            if numpy.max(label_lane[_lane_idx]) == -2:
                label_enranged_lane[_lane_idx, :] = -1
                pass
            else:

                is_lane = numpy.where(label_lane[_lane_idx] > 0)[0]
                
                if is_lane[0] > 0:
                    label_enranged_lane[_lane_idx][:is_lane[0]*2] = -1
                else:
                    pass

                if is_lane[-1] < NUM_LANE_POINT-1:
                    label_enranged_lane[_lane_idx][is_lane[-1]*2+1:] = -1
                else:
                    pass

        label_enranged_lane = numpy.clip(label_enranged_lane, 0, 1280)
        label_enranged_lane = label_enranged_lane/1280.0

        label_enranged_lane[label_enranged_lane==0] = -1

        label_binary_lane = numpy.zeros_like(label_enranged_lane)

        label_binary_lane[numpy.where(label_enranged_lane>0)] = 1

        input_padded_img = numpy.transpose(input_padded_img, (2,0,1))

        return torch.FloatTensor(input_padded_img), torch.FloatTensor(label_enranged_lane), torch.FloatTensor(label_binary_lane)
    def __len__(self):
        return len(self.merged_label)
    

class Tusimple_Valid_Dataset(torch.utils.data.Dataset):
    def __init__(self):
        super(Tusimple_Valid_Dataset, self).__init__()

        self.TUSIMPLE_DIR = TUSIMPLE_DIR
        self.resize_shape = (RESIZE_HEIGHT, RESIZE_WIDTH)

        self.merged_label = []
        
        with open(os.path.join(self.TUSIMPLE_DIR, 'test_label_new.json'), 'r') as f:
            for line in f:
                self.merged_label.append(json.loads(line.rstrip()))
        file_idx = list()
        for ith_label in self.merged_label:
            file_idx.append(ith_label["raw_file"])

        file_idx = numpy.argsort(file_idx)
        
        self.merged_label = numpy.array(self.merged_label)[file_idx]

    def __getitem__(self, index):
        
        raw_file = self.merged_label[index]["raw_file"]
        lanes = numpy.array(self.merged_label[index]["lanes"])
        h_samples = numpy.array(self.merged_label[index]["h_samples"])
        raw_file = self.merged_label[index]["raw_file"]

        input_color_img = cv2.imread(os.path.join(self.TUSIMPLE_DIR,'test_set',raw_file),cv2.IMREAD_COLOR)

        input_color_img = cv2.resize(input_color_img, (self.resize_shape[1], self.resize_shape[0]))
        input_color_img = cv2.cvtColor(input_color_img, cv2.COLOR_RGB2BGR)
        input_color_img = numpy.array(input_color_img)/255.0

        label_lane = numpy.zeros((6,NUM_LANE_POINT))
        for idx in range(len(lanes)):
            label_lane[idx] = lanes[idx]
        
        label_lane[label_lane==0] = -2

        label_enranged_lane = numpy.empty((6, ENLENGTHEN_NUM_LANE_POINT))

        for _idx in range(ENLENGTHEN_NUM_LANE_POINT):
            
            if _idx % 2 == 0:
                label_enranged_lane[:, _idx] = label_lane[:,_idx//2]
            else:
                label_enranged_lane[:, _idx] = (label_lane[:,_idx//2] + label_lane[:,(_idx//2)+1])/2

        for _lane_idx in range(6):

            if numpy.max(label_lane[_lane_idx]) == -2:
                label_enranged_lane[_lane_idx, :] = -1
                pass
            else:

                is_lane = numpy.where(label_lane[_lane_idx] > 0)[0]
                
                if is_lane[0] > 0:
                    label_enranged_lane[_lane_idx][:is_lane[0]*2] = -1
                else:
                    pass

                if is_lane[-1] < NUM_LANE_POINT-1:
                    label_enranged_lane[_lane_idx][is_lane[-1]*2+1:] = -1
                else:
                    pass

        
        label_enranged_lane = numpy.clip(label_enranged_lane, 0, 1280)
        label_enranged_lane = label_enranged_lane/1280.0

        label_enranged_lane[label_enranged_lane==0] = -1

        label_binary_lane = numpy.zeros_like(label_enranged_lane)

        label_binary_lane[numpy.where(label_enranged_lane>0)] = 1

        input_color_img = numpy.transpose(input_color_img, (2,0,1))

        return torch.FloatTensor(input_color_img), torch.FloatTensor(input_color_img),  torch.FloatTensor(label_enranged_lane), torch.FloatTensor(label_binary_lane)
    def __len__(self):
        return len(self.merged_label)

class Tusimple_MultiScale_Valid_Dataset(torch.utils.data.Dataset):
    def __init__(self):
        super(Tusimple_MultiScale_Valid_Dataset, self).__init__()

        self.TUSIMPLE_DIR = TUSIMPLE_DIR
        self.resize_shape = (RESIZE_HEIGHT, RESIZE_WIDTH)

        self.merged_label = []
        
        with open(os.path.join(self.TUSIMPLE_DIR, 'test_label_new.json'), 'r') as f:
            for line in f:
                self.merged_label.append(json.loads(line.rstrip()))
        file_idx = list()
        for ith_label in self.merged_label:
            file_idx.append(ith_label["raw_file"])

        file_idx = numpy.argsort(file_idx)
        
        self.merged_label = numpy.array(self.merged_label)[file_idx]

    def __getitem__(self, index):
        
        raw_file = self.merged_label[index]["raw_file"]
        lanes = numpy.array(self.merged_label[index]["lanes"])
        h_samples = numpy.array(self.merged_label[index]["h_samples"])
        raw_file = self.merged_label[index]["raw_file"]

        input_color_img = cv2.imread(os.path.join(self.TUSIMPLE_DIR, raw_file),cv2.IMREAD_COLOR)

        input_color_img_scale_1on1 =  cv2.resize(input_color_img, (self.resize_shape[1],        self.resize_shape[0]))
        input_color_img_scale_1on2 =  cv2.resize(input_color_img, (int(self.resize_shape[1]/2), int(self.resize_shape[0]/2)))
        input_color_img_scale_1on4 =  cv2.resize(input_color_img, (int(self.resize_shape[1]/4), int(self.resize_shape[0]/4)))
        input_color_img_scale_1on8 =  cv2.resize(input_color_img, (int(self.resize_shape[1]/8), int(self.resize_shape[0]/8)))

        input_color_img_scale_1on1 = cv2.cvtColor(input_color_img_scale_1on1, cv2.COLOR_RGB2BGR)
        input_color_img_scale_1on2 = cv2.cvtColor(input_color_img_scale_1on2, cv2.COLOR_RGB2BGR)
        input_color_img_scale_1on4 = cv2.cvtColor(input_color_img_scale_1on4, cv2.COLOR_RGB2BGR)
        input_color_img_scale_1on8 = cv2.cvtColor(input_color_img_scale_1on8, cv2.COLOR_RGB2BGR)

        input_color_img_scale_1on1 = numpy.array(input_color_img_scale_1on1)/255.0
        input_color_img_scale_1on2 = numpy.array(input_color_img_scale_1on2)/255.0
        input_color_img_scale_1on4 = numpy.array(input_color_img_scale_1on4)/255.0
        input_color_img_scale_1on8 = numpy.array(input_color_img_scale_1on8)/255.0

        label_lane = numpy.zeros((6,NUM_LANE_POINT))
        for idx in range(len(lanes)):
            label_lane[idx] = lanes[idx]
        
        label_lane[label_lane==0] = -2

        label_enranged_lane = numpy.empty((6, ENLENGTHEN_NUM_LANE_POINT))

        for _idx in range(ENLENGTHEN_NUM_LANE_POINT):
            
            if _idx % 2 == 0:
                label_enranged_lane[:, _idx] = label_lane[:,_idx//2]
            else:
                label_enranged_lane[:, _idx] = (label_lane[:,_idx//2] + label_lane[:,(_idx//2)+1])/2

        for _lane_idx in range(6):

            if numpy.max(label_lane[_lane_idx]) == -2:
                label_enranged_lane[_lane_idx, :] = -1
                pass
            else:

                is_lane = numpy.where(label_lane[_lane_idx] > 0)[0]
                
                if is_lane[0] > 0:
                    label_enranged_lane[_lane_idx][:is_lane[0]*2] = -1
                else:
                    pass

                if is_lane[-1] < NUM_LANE_POINT-1:
                    label_enranged_lane[_lane_idx][is_lane[-1]*2+1:] = -1
                else:
                    pass

        
        label_enranged_lane = numpy.clip(label_enranged_lane, 0, 1280)
        label_enranged_lane = label_enranged_lane/1280.0

        label_enranged_lane[label_enranged_lane==0] = -1

        label_binary_lane = numpy.zeros_like(label_enranged_lane)

        label_binary_lane[numpy.where(label_enranged_lane>0)] = 1

        input_color_img_scale_1on1 = numpy.transpose(input_color_img_scale_1on1, (2,0,1))
        input_color_img_scale_1on2 = numpy.transpose(input_color_img_scale_1on2, (2,0,1))
        input_color_img_scale_1on4 = numpy.transpose(input_color_img_scale_1on4, (2,0,1))
        input_color_img_scale_1on8 = numpy.transpose(input_color_img_scale_1on8, (2,0,1))

        return torch.FloatTensor(input_color_img_scale_1on1), torch.FloatTensor(input_color_img_scale_1on2), torch.FloatTensor(input_color_img_scale_1on4), torch.FloatTensor(input_color_img_scale_1on8),  torch.FloatTensor(label_enranged_lane), torch.FloatTensor(label_binary_lane)
    def __len__(self):
        return len(self.merged_label)

class Tusimple_Valid_Padding_Dataset(torch.utils.data.Dataset):
    def __init__(self, init_padding=80, argsort=True):
        super(Tusimple_Valid_Padding_Dataset, self).__init__()

        self.TUSIMPLE_DIR = TUSIMPLE_DIR
        self.resize_shape = (RESIZE_HEIGHT, RESIZE_WIDTH)
        self.init_padding  = init_padding

        self.merged_label = list()
        
        with open(os.path.join(self.TUSIMPLE_DIR, 'test_label_new.json'), 'r') as f:
            for line in f:
                self.merged_label.append(json.loads(line.rstrip()))
        file_idx = list()
        for ith_label in self.merged_label:
            file_idx.append(ith_label["raw_file"])

        if argsort:
            file_idx = numpy.argsort(file_idx)
            self.merged_label = numpy.array(self.merged_label)[file_idx]
        else:
            self.merged_label = numpy.array(self.merged_label)
        
        # for label in label_list:
        #     with open(os.path.join(self.TUSIMPLE_DIR, 'test_set', label), 'r') as f:
        #         for line in f:
        #             self.merged_label.append(json.loads(line.rstrip()))

    def __getitem__(self, index):
        
        raw_file = self.merged_label[index]["raw_file"]
        lanes = numpy.array(self.merged_label[index]["lanes"])
        h_samples = numpy.array(self.merged_label[index]["h_samples"])

        enranged_lanes = numpy.zeros((len(lanes), 56))

        if numpy.min(h_samples) != 160:

            h_samples  = numpy.array([x for x in range(HEIGHT_TOP_CROP, HEIGHT_BOTTOM_CROP, 10)])

            for lane_idx in range(len(lanes)):
                enranged_lanes[lane_idx] = numpy.append([-2 for _ in range(160, 240, 10)], lanes[lane_idx], axis = 0)

        else:
            enranged_lanes = lanes
        
        raw_file = self.merged_label[index]["raw_file"]

        input_color_img = cv2.imread(os.path.join(self.TUSIMPLE_DIR,raw_file),cv2.IMREAD_COLOR)
        

        input_color_img = cv2.resize(input_color_img, (self.resize_shape[1], self.resize_shape[0]))
        input_color_img = cv2.cvtColor(input_color_img, cv2.COLOR_RGB2BGR)
        input_color_img = numpy.array(input_color_img)/255.0

        
        input_padded_img = numpy.zeros((720+self.init_padding, 1280, 3))
        
        input_padded_img[self.init_padding:] = input_color_img[0:720]

        label_lane = numpy.zeros((6,NUM_LANE_POINT))
        for idx in range(len(enranged_lanes)):
            label_lane[idx] = enranged_lanes[idx]
        
        label_lane[label_lane==0] = -2

        label_enranged_lane = numpy.empty((6, ENLENGTHEN_NUM_LANE_POINT))
        
        for _idx in range(ENLENGTHEN_NUM_LANE_POINT):
            
            if _idx % 2 == 0:
                label_enranged_lane[:, _idx] = label_lane[:,_idx//2]
            else:
                label_enranged_lane[:, _idx] = (label_lane[:,_idx//2] + label_lane[:,(_idx//2)+1])/2

        for _lane_idx in range(6):

            if numpy.max(label_lane[_lane_idx]) == -2:
                label_enranged_lane[_lane_idx, :] = -1
                pass
            else:

                is_lane = numpy.where(label_lane[_lane_idx] > 0)[0]
                
                if is_lane[0] > 0:
                    label_enranged_lane[_lane_idx][:is_lane[0]*2] = -1
                else:
                    pass

                if is_lane[-1] < NUM_LANE_POINT-1:
                    label_enranged_lane[_lane_idx][is_lane[-1]*2+1:] = -1
                else:
                    pass

        label_enranged_lane = numpy.clip(label_enranged_lane, 0, 1280)
        label_enranged_lane = label_enranged_lane/1280.0

        label_enranged_lane[label_enranged_lane==0] = -1

        label_binary_lane = numpy.zeros_like(label_enranged_lane)

        label_binary_lane[numpy.where(label_enranged_lane>0)] = 1

        input_padded_img = numpy.transpose(input_padded_img, (2,0,1))

        return torch.FloatTensor(input_color_img), torch.FloatTensor(input_padded_img), torch.FloatTensor(label_enranged_lane), torch.FloatTensor(label_binary_lane)
    def __len__(self):
        return len(self.merged_label)
    
class Tusimple_TrueLane_Dataset(torch.utils.data.Dataset):
    def __init__(self):
        super(Tusimple_TrueLane_Dataset, self).__init__()

        # self.SET_CLF_DIR = f"/home/tusimple_data/classifier/multi_arm_{NUM_ARM}"
        self.SET_CLF_DIR = str(os.path.join(TUSIMPLE_FALSE_LANE_DS_DIR, MODEL_NAME))
        true_list        = [os.path.join(self.SET_CLF_DIR, 'true', png) for png in os.listdir(os.path.join(self.SET_CLF_DIR, 'true'))]

        self.png_list = true_list

    def __getitem__(self, index):
        
        input_color_img = matplotlib.pyplot.imread(self.png_list[index])
        
        input_color_img = numpy.transpose(input_color_img, (2,0,1))

        return torch.FloatTensor(input_color_img), torch.FloatTensor([1])
    
    def __len__(self):
        return len(self.png_list)
    
class Tusimple_FalseLane_Dataset(torch.utils.data.Dataset):
    def __init__(self, ratio=8):
        super(Tusimple_FalseLane_Dataset, self).__init__()

        # self.SET_CLF_DIR = f"/home/tusimple_data/classifier/multi_arm_{NUM_ARM}"
        self.SET_CLF_DIR = str(os.path.join(TUSIMPLE_FALSE_LANE_DS_DIR, MODEL_NAME))
        false_list       = [os.path.join(self.SET_CLF_DIR, 'false', png) for png in os.listdir(os.path.join(self.SET_CLF_DIR, 'false'))]
        false_list       = list(chain.from_iterable(repeat(false_list, ratio)))
        self.png_list = false_list

    def __getitem__(self, index):
        
        input_color_img = matplotlib.pyplot.imread(self.png_list[index])

        input_color_img = numpy.transpose(input_color_img, (2,0,1))

        return torch.FloatTensor(input_color_img), torch.FloatTensor([0])
    
    def __len__(self):
        return len(self.png_list)