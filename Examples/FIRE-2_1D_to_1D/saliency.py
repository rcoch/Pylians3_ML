import numpy as np
import sys,os
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torchvision.utils as vutils
import data
import architecture
import shap
import pandas as pd

#################################### INPUT ##########################################
# data parameters
fin  = 'my_fire2_data.h5'
seed = 5
realizations = 2000

# architecture parameters
h1 = 2000
dropout_rate = 0.3

# training parameters
batch_size = 256

# name of output files
name   = '1hd_100_0.0_0.0'
fout   = 'results/%s.txt'%name
fmodel = 'models/1hd_2000_0.3_1e-5.pt'

feature_names=["redshift","SFR_inst","SFR_10","SFR_100","M_dust","M_gas","M_star"]
#####################################################################################

# use GPUs if available
GPU = torch.cuda.is_available()
device = torch.device("cuda" if GPU else "cpu")
print('GPU:',GPU)
print('Training on',device)
cudnn.benchmark = True      #May train faster but cost more memory

# define loss function
criterion = nn.MSELoss()

# get the data
test_loader  = data.create_dataset('test', seed, fin, batch_size)
train_loader  = data.create_dataset('train', seed, fin, batch_size)

# get the parameters of the trained model
model = architecture.model_1hl(7, h1, 1, dropout_rate)

model.load_state_dict(torch.load(fmodel, map_location=torch.device(device)))
model.to(device=device)

# grab 1 element
model.eval()
for x,y in test_loader:
    with torch.no_grad():
        x = x.to(device)
        y = y.to(device)
        break
        
model.eval()
for x_train,y_train in train_loader:
    with torch.no_grad():
        x_train = x_train.to(device)
        y_train = y_train.to(device)
        break

x_train_array= x_train.numpy()
print ("np.shape(x_train_array)",np.shape(x_train_array))

e = shap.DeepExplainer(model,torch.from_numpy(x_train_array[np.random.choice(np.arange(len(x_train_array)), 256, replace=False)]).to(device))

x_samples = x_train_array[np.random.choice(np.arange(len(x_train_array)), 256, replace=False)]
print(len(x_samples))
shap_values = e.shap_values(torch.from_numpy(x_samples).to(device))
shap.summary_plot(shap_values, features=x_samples, feature_names = feature_names)

df = pd.DataFrame({
    "mean_abs_shap": np.mean(np.abs(shap_values), axis=0),
    "stdev_abs_shap": np.std(np.abs(shap_values), axis=0),
    "name": feature_names
})

print (df)
