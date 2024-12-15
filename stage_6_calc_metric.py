# %%
import json
import numpy as np
import cv2
import matplotlib.pyplot as plt
import sys
sys.path.append("../")
from evaluate.lane import LaneEval
import os
import config
# %%
NUM_ARM    = config.NUM_ARM
MODEL_NAME = config.MODEL_NAME
LEARNING_RATE  = config.LEARNING_RATE
MODEL_SAVE_DIR = config.MODEL_SAVE_DIR
JSON_DIR = str(os.path.join(MODEL_SAVE_DIR, MODEL_NAME, "pred_file.json"))
TUSIMPLE_DIR  = config.TUSIMPLE_DIR
TEST_LABEL_JSON_DIR = f'{TUSIMPLE_DIR}/test_label_new.json'
# %%
json_pred = [json.loads(line) for line in open(JSON_DIR).readlines()]
json_gt = [json.loads(line) for line in open(TEST_LABEL_JSON_DIR)]

pred, gt = json_pred[0], json_gt[0]
pred_lanes = pred['lanes']
# run_time = [0 for _ in range(lenpred['lanes'])]
gt_lanes = gt['lanes']
# y_samples = pred['h_samples']
y_samples = gt['h_samples']
raw_file = gt['raw_file']
# %%
import os
img = plt.imread(os.path.join(TUSIMPLE_DIR, raw_file))
plt.imshow(img)
plt.show()
# %%
gt_lanes_vis = [[(x, y) for (x, y) in zip(lane, y_samples) if x >= 0] for lane in gt_lanes]
img_vis = img.copy()

for lane in gt_lanes_vis:
    for pt in lane:
        cv2.circle(img_vis, pt, radius=5, color=(255, 0, 0), thickness=-1)

plt.imshow(img_vis)
plt.show()
# %%
pred_lanes_vis = [[(x, y) for (x, y) in zip(lane, y_samples) if x >= 0] for lane in pred_lanes]
img_vis = img.copy()

for lane in pred_lanes_vis:
    for pt in lane:
        cv2.circle(img_vis, pt, radius=5, color=(255, 0, 0), thickness=-1)
    break
plt.imshow(img_vis)
plt.show()
# %%
gt_lanes_vis = [[(x, y) for (x, y) in zip(lane, y_samples) if x >= 0] for lane in gt_lanes]
pred_lanes_vis = [[(x, y) for (x, y) in zip(lane, y_samples) if x >= 0] for lane in pred_lanes]
img_vis = img.copy()

for lane in gt_lanes_vis:
    cv2.polylines(img_vis, np.int32([lane]), isClosed=False, color=(0,255,0), thickness=5)
for lane in pred_lanes_vis:
    cv2.polylines(img_vis, np.int32([lane]), isClosed=False, color=(0,0,255), thickness=2)

plt.imshow(img_vis)
plt.show()
# %%
np.random.shuffle(pred_lanes)
# Overall Accuracy, False Positive Rate, False Negative Rate
print(LaneEval.bench(pred_lanes, gt_lanes, y_samples, 0.1))
# %%
acc_list = list()
fpr_list = list()
fnr_list = list()

for pred, gt in zip(json_pred, json_gt):
    pred_lanes = pred['lanes']
    gt_lanes = gt['lanes']
    y_samples = gt['h_samples']
    acc, fpr, fnr = LaneEval.bench(pred_lanes, gt_lanes, y_samples, 0.1)
    acc_list.append(acc)
    fpr_list.append(fpr)
    fnr_list.append(fnr)

print(np.mean(acc_list))
print(np.mean(fpr_list))
print(np.mean(fnr_list))
# %%

# 6arm

#0.8581532855773512
#0.13820393002635994
#0.21594176851186198
# %%

# 6arm - 20 patch
# 0.893583003834172
# 0.21369518332135154
# 0.1673256649892164

# %%

# 6arm - 50 patch
# 0.8833177142172468
# 0.16690031152647972
# 0.1811346752935538

# %%
# 12arm 50 patch
# 0.8267722347745712
# 0.31237119578241074
# 0.32422717469446444