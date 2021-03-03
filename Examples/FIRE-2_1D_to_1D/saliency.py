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
model_params = 7
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
model = architecture.model_1hl(model_params, h1, 1, dropout_rate)

model.load_state_dict(torch.load(fmodel, map_location=torch.device(device)))
model.to(device=device)

# grab some data
model.eval()
for x_train,y_train in train_loader:
    with torch.no_grad():
        x_train = x_train.to(device)
        y_train = y_train.to(device)
        break

x_train_array= x_train.numpy()
print ("np.shape(x_train_array)",np.shape(x_train_array))

# DeepExplainer approximates SHAP values for deep learning models.
# Can pass the entire training set as data, but could be slow - 100 samples will give a good estimate, 1000 samples a very good estimate.
e = shap.DeepExplainer(model,torch.from_numpy(x_train_array[np.random.choice(np.arange(len(x_train_array)), 256, replace=False)]).to(device))

x_samples = x_train_array[np.random.choice(np.arange(len(x_train_array)), 256, replace=False)]

shap_values = e.shap_values(torch.from_numpy(x_samples).to(device))

# print the results
df = pd.DataFrame({
    "mean_abs_shap": np.mean(np.abs(shap_values), axis=0),
    "stdev_abs_shap": np.std(np.abs(shap_values), axis=0),
    "name": feature_names
})
print (df)

# make a shap summary plot. this shows from top to bottom the most important features in the model.
shap.summary_plot(shap_values, features=x_samples, feature_names = feature_names)

# make shap dependence plots
# each dot is a single row from the dataset. The x axis shows the value of the feature, and the y axis represents how much knowing that feature changes the output for the model for that sample's prediction. Vertical dispersion of data represents interaction effects.
# the interaction index here (i.e. the color coding) is chosen automatically as what seems to be the strongest interaction.

for i in range(0,model_params):
    shap.dependence_plot(i, shap_values, x_samples, feature_names=feature_names)


