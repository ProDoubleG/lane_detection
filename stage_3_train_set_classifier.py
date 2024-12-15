# %%
# Package imports
import torch
import os
import numpy as np
import dataset
import model
import config
import utils
import utils.setting
# %%
utils.setting.fix_seed(config.SEED)
# %%
BATCH_SIZE    = config.BATCH_SIZE
LEARNING_RATE = config.LEARNING_RATE
NUM_ARM       = config.NUM_ARM
MODEL_NAME    = config.MODEL_NAME
MODEL_SAVE_DIR  = config.MODEL_SAVE_DIR
PATIENCE_LIMIT  = config.PATIENCE_LIMIT
# %%
# get dataset and loader
true_dataset  = dataset.Tusimple_TrueLane_Dataset()
false_dataset = dataset.Tusimple_FalseLane_Dataset(ratio=7)
dataset_all = true_dataset + false_dataset
train_loader = torch.utils.data.DataLoader(dataset_all, batch_size=BATCH_SIZE, shuffle=True)
# %%
# Loader test
for i in train_loader:
    _input, _label = i
    break
del _input, _label
# %%
# Load model on GPU
set_clf = model.LaneSetClassifier()

USE_CUDA = torch.cuda.is_available()

if USE_CUDA:
    device = torch.device('cuda:0')
    torch.cuda.set_device(device)
else:
    device = torch.device('cpu')

if USE_CUDA:
    set_clf.to(device)
# %%
# Get optimizer and loss function class
set_clf_optimizer = torch.optim.Adam(set_clf.parameters(), lr=LEARNING_RATE)
set_clf_loss_fn   = torch.nn.BCELoss()
# %%
# Train Regressor and Classifier
EPOCH = 100

set_clf_loss_list  = list()

for epoch in range(EPOCH):
    print(f"---------------Epoch : {epoch+1}/{EPOCH}--------------------")
    train_ce_loss = 0.0
    
    for train_idx, data in enumerate(train_loader, 0):
        set_clf_optimizer.zero_grad()

        print('\r',f"training {train_idx+1}/{len(train_loader)}, BCE_loss: {train_ce_loss/(train_idx+1):0.5f}", end =" ")

        inputs, labels = data
        
        set_clf_outputs = set_clf(inputs.to(device))
        
        ce_loss = set_clf_loss_fn(set_clf_outputs, labels.to(device))

        ce_loss.backward()
        set_clf_optimizer.step()

        train_ce_loss += ce_loss.item()
    
    if PATIENCE_LIMIT > current_patience:
        
        if train_ce_loss/(train_idx+1) < best_loss*0.9:
            
            # save model

            best_loss = train_ce_loss/(train_idx+1)
            best_epoch = epoch+1
            best_model = set_clf

            current_patience = 0
        else:
            current_patience += 1
    
    else:
        break

    set_clf_loss_list.append(train_ce_loss/(train_idx+1))
    print("")
# %%
torch.save(best_model.state_dict(), os.path.join(os.path.join((os.path.join(MODEL_SAVE_DIR, MODEL_NAME, "set_classifier.pt")))))