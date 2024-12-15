import torch
import numpy as np
from torchvision import models

NUM_ARM = 6
RESIZE_HEIGHT = 720
RESIZE_WIDTH  = 1280

HEIGHT_TOP_CROP = 160
HEIGHT_BOTTOM_CROP = 720
NUM_LANE_POINT = len(range(HEIGHT_TOP_CROP, HEIGHT_BOTTOM_CROP, 10))
ENLENGTHEN_NUM_LANE_POINT = len(range(HEIGHT_TOP_CROP, HEIGHT_BOTTOM_CROP-5, 5))

class LanePointRegressor(torch.nn.Module):
    def __init__(self, num_arm=NUM_ARM):
        super(LanePointRegressor, self).__init__()
        self.resnet101 = models.resnet101()
        self.num_arm = num_arm
        self.resnet_embedding_conv = torch.nn.Conv2d(in_channels=2048, out_channels=256, kernel_size=1)

        self.avg_pool2d = torch.nn.AdaptiveAvgPool2d((16, 32))
        self.bridge_conv2d = torch.nn.Conv2d(in_channels=256, out_channels=self.num_arm*8, kernel_size=3, padding='same')

        self.avg_pool1d = torch.nn.AdaptiveAvgPool1d((256))
        self.bridge_conv1d = torch.nn.Conv1d(in_channels=self.num_arm*8, out_channels=self.num_arm, kernel_size=3, padding='same')
        
        self.arm_fc = torch.nn.Linear(in_features=256, out_features=111)
       
    def forward(self, x):
        x = self.resnet101.conv1(x) # (half scale, 64 channel)
        # print("conv out: ", x.shape)
        x = self.resnet101.bn1(x)
        x = self.resnet101.relu(x)
        # print("before maxpool : ", x.shape)
        x = self.resnet101.maxpool(x)

        # print("after max pool : ", x.shape)
        x = self.resnet101.layer1(x)
        # print("layer 1 : ", x.shape)
        x = self.resnet101.layer2(x)
        # print("layer 2 : ", x.shape)
        x = self.resnet101.layer3(x)
        # print("layer 3 : ", x.shape)
        x = self.resnet101.layer4(x)
        # print("layer 4 : ", x.shape)
        
        x = self.resnet_embedding_conv(x)
        x = self.avg_pool2d(x)
        x = self.bridge_conv2d(x)
        x = torch.flatten(x, start_dim=2, end_dim=3)
        x = self.avg_pool1d(x)
        x = self.bridge_conv1d(x)
        x = torch.nn.functional.sigmoid(self.arm_fc(x))

        return x
    
class LanePointRegressor_MutliScale(torch.nn.Module):
    def __init__(self, num_arm=NUM_ARM):
        super(LanePointRegressor_MutliScale, self).__init__()

        self.resnet101 = models.resnet101()
        self.num_arm = num_arm

        self.conv1    = torch.nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(7,7), stride=(2,2), padding=(3,3), bias=False)
        self.bn1      = torch.nn.BatchNorm2d(num_features=64)
        self.relu1    = torch.nn.ReLU(inplace=True)
        self.maxpool1 = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)

        self.conv2 = torch.nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(5,5), stride=(2,2), padding=(2,2), bias=False)
        self.bn2   = torch.nn.BatchNorm2d(num_features=64)
        self.relu2 = torch.nn.ReLU(inplace=True)

        self.conv3 = torch.nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(3,3), stride=(2,2), padding=(1,1), bias=False)
        self.bn3   = torch.nn.BatchNorm2d(num_features=64)
        self.relu3 = torch.nn.ReLU(inplace=True)

        self.conv4 = torch.nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(2,2), stride=(2,2), bias=False)
        self.bn4   = torch.nn.BatchNorm2d(num_features=64)
        self.relu4 = torch.nn.ReLU(inplace=True)

        self.concat_conv_1 = torch.nn.Conv2d(in_channels=256+64,  out_channels=256,  kernel_size=(1,1), bias=False) # 64+64  to 64
        self.concat_conv_2 = torch.nn.Conv2d(in_channels=512+64, out_channels=512, kernel_size=(1,1), bias=False) # 256+64 to 256
        self.concat_conv_3 = torch.nn.Conv2d(in_channels=1024+64, out_channels=1024, kernel_size=(1,1), bias=False)

        self.resnet_embedding_conv = torch.nn.Conv2d(in_channels=2048, out_channels=256, kernel_size=1)
        self.avg_pool2d = torch.nn.AdaptiveAvgPool2d((16, 32))
        self.bridge_conv2d = torch.nn.Conv2d(in_channels=256, out_channels=self.num_arm*8, kernel_size=3, padding='same')

        self.avg_pool1d = torch.nn.AdaptiveAvgPool1d((256))
        self.bridge_conv1d = torch.nn.Conv1d(in_channels=self.num_arm*8, out_channels=self.num_arm, kernel_size=3, padding='same')
        
        self.arm_fc = torch.nn.Linear(in_features=256, out_features=111)
       
    def forward(self, input_scale_1on1, input_scale_1on2, input_scale_1on4, input_scale_1on8):

        x1 = self.conv1(input_scale_1on1)
        x1 = self.bn1(x1)
        x1 = self.relu1(x1)
        x1 = self.maxpool1(x1)

        x2 = self.conv2(input_scale_1on2)
        x2 = self.bn2(x2)
        x2 = self.relu2(x2)

        x3 = self.conv3(input_scale_1on4)
        x3 = self.bn3(x3)
        x3 = self.relu3(x3)

        x4 = self.conv4(input_scale_1on8)
        x4 = self.bn4(x4)
        x4 = self.relu4(x4)

        x = self.resnet101.layer1(x1)
        x = torch.concat((x, x2), dim=1)
        x = self.concat_conv_1(x)

        x = self.resnet101.layer2(x)
        x = torch.concat((x, x3), dim=1)
        x = self.concat_conv_2((x))

        x = self.resnet101.layer3(x)
        x = torch.concat((x, x4), dim=1)
        x = self.concat_conv_3(x)

        x = self.resnet101.layer4(x)
        
        x = self.resnet_embedding_conv(x)
        x = self.avg_pool2d(x)
        x = self.bridge_conv2d(x)
        x = torch.flatten(x, start_dim=2, end_dim=3)
        x = self.avg_pool1d(x)
        x = self.bridge_conv1d(x)
        x = torch.nn.functional.sigmoid(self.arm_fc(x))

        return x
    
class LanePointClassifier(torch.nn.Module):
    def __init__(self, num_arm=NUM_ARM):
        super(LanePointClassifier, self).__init__()

        self.resnet101 = models.resnet101()
        self.num_arm = num_arm
        self.resnet_embedding_conv = torch.nn.Conv2d(in_channels=2048, out_channels=256, kernel_size=1)

        self.avg_pool2d = torch.nn.AdaptiveAvgPool2d((16, 32))
        self.bridge_conv2d = torch.nn.Conv2d(in_channels=256, out_channels=self.num_arm*8, kernel_size=3, padding='same')

        self.avg_pool1d = torch.nn.AdaptiveAvgPool1d((256))
        self.bridge_conv1d = torch.nn.Conv1d(in_channels=self.num_arm*8, out_channels=self.num_arm, kernel_size=3, padding='same')
        
        self.bridge_fc = torch.nn.Linear(in_features=256, out_features=111)

        self.arm_conv1d_1 = torch.nn.Conv1d(in_channels=self.num_arm*2, out_channels=self.num_arm*8, kernel_size=3, padding='same')
        self.arm_fc_1  = torch.nn.Linear(in_features=111,out_features=256)
        self.arm_conv1d_2 = torch.nn.Conv1d(in_channels=self.num_arm*8, out_channels=self.num_arm, kernel_size=1, padding='same')
        self.arm_fc_2  = torch.nn.Linear(in_features=256,out_features=111)

    def forward(self, input_img, point_reg_output):

        x = self.resnet101.conv1(input_img)
        x = self.resnet101.bn1(x)
        x = self.resnet101.relu(x)
        x = self.resnet101.maxpool(x)

        x = self.resnet101.layer1(x)
        x = self.resnet101.layer2(x)
        x = self.resnet101.layer3(x)
        x = self.resnet101.layer4(x)
        
        x = self.resnet_embedding_conv(x)
        x = self.avg_pool2d(x)
        x = self.bridge_conv2d(x)
        x = torch.flatten(x, start_dim=2, end_dim=3)
        x = self.avg_pool1d(x)
        x = self.bridge_conv1d(x)
        x = self.bridge_fc(x)

        x = torch.concat((point_reg_output, x), dim=1)
        x = self.arm_conv1d_1(x)
        x = self.arm_fc_1(x)
        x = self.arm_conv1d_2(x)
        x = torch.nn.functional.sigmoid(self.arm_fc_2(x))

        return x
    
# %%
# Lane suppression
class LaneSetSuppressor(torch.nn.Module):
    def __init__(self):
        super(LaneSetSuppressor, self).__init__()
        self.point_clf_threshold   = 0.6
        self.suppression_threshold = 0.05

    def getSameElement(self, a, b):
        indices = torch.zeros_like(a, dtype=torch.uint8)

        for elem in b:
            indices = indices | (a == elem).type(torch.uint8)

        intersection = a[indices.type(torch.bool)]

        return intersection
    
    def forward(self, point_reg_output, point_clf_output):

        # suppression_output = point_reg_output.clone().detach()
        
        for batch_idx in range(len(point_reg_output)):

            suppressed_lane_list = list()

            for lane_idx, one_sample_pred_lane in enumerate(point_reg_output[batch_idx]):
                diff = torch.abs(point_reg_output[batch_idx] - one_sample_pred_lane)
                for diff_idx, _diff in enumerate(diff):
                    if lane_idx == diff_idx:
                        pass
                    else:
                        _diff = _diff[self.getSameElement(torch.where(point_clf_output[batch_idx][diff_idx] > self.point_clf_threshold)[0], torch.where(point_clf_output[batch_idx][lane_idx] > self.point_clf_threshold)[0])]
                        if torch.mean(_diff) < self.suppression_threshold:
                            if torch.mean(point_clf_output[batch_idx][diff_idx][point_clf_output[batch_idx][diff_idx] > self.point_clf_threshold]) > torch.mean(point_clf_output[batch_idx][lane_idx][point_clf_output[batch_idx][lane_idx] > self.point_clf_threshold]):
                                suppressed_lane_list.append(lane_idx)
                            else:
                                suppressed_lane_list.append(diff_idx)

            suppressed_lane_list = np.unique(suppressed_lane_list)
            
            # suppression_output[batch_idx][suppressed_lane_list] = -1

        return suppressed_lane_list
# %%
# Lane Set Classifier
class LaneSetClassifier(torch.nn.Module):
    def __init__(self):
        super(LaneSetClassifier, self).__init__()

        self.resnet101 = models.resnet101()

        self.resnet_embedding_conv = torch.nn.Conv2d(in_channels=2048, out_channels=256, kernel_size=1)
        
        self.avg_pool2d = torch.nn.AdaptiveAvgPool2d((4, 4))
        self.bridge_conv2d = torch.nn.Conv2d(in_channels=256, out_channels=NUM_ARM*8, kernel_size=3, padding='same')

        self.linear = torch.nn.Linear(in_features=NUM_ARM*8*4*4, out_features=1)
        
    def forward(self, input_img):
        
        x = self.resnet101.conv1(input_img)
        x = self.resnet101.bn1(x)
        x = self.resnet101.relu(x)
        x = self.resnet101.maxpool(x)

        x = self.resnet101.layer1(x)
        x = self.resnet101.layer2(x)
        x = self.resnet101.layer3(x)
        x = self.resnet101.layer4(x)

        x = self.resnet_embedding_conv(x)
        x = self.avg_pool2d(x)
        x = self.bridge_conv2d(x)
        x = torch.flatten(x, start_dim=1, end_dim=-1)
        out = torch.nn.functional.sigmoid(self.linear(x))
        
        return out