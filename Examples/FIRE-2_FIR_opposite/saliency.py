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
import matplotlib.pyplot as plt
from matplotlib import cm

#################################### INPUT ##########################################
# data parameters
fin  = 'my_fire2_data_fir_subset.h5'
seed = 5
realizations = 2000
model_params = 8
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
feature_names=["100","150","185","230","345","460","650","870"]

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
model = architecture.model_1hl(model_params, h1, 7, dropout_rate)

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
e = shap.DeepExplainer(model,torch.from_numpy(x_train_array[np.random.choice(np.arange(len(x_train_array)), 100, replace=False)]).to(device))

x_samples = x_train_array[np.random.choice(np.arange(len(x_train_array)), 100, replace=False)]

shap_values = e.shap_values(torch.from_numpy(x_samples).to(device))
print ("shap_values",np.shape(shap_values))
#shap_values = shap_values[10]

choice_wavelengths = [0,1,2,3,4,5,6,7]
#alma_freqs = np.array([100,150,185,230,345,460,650,870])
#alma_wavelengths = 3*1e8*1./(alma_freqs*1e9)*1e6

feature_names=["redshift","SFR_inst","SFR_10","SFR_100","M_dust","M_gas","M_star"]
choice_features = [0,1,2,3,4,5,6]

choice_shap_100 = []
choice_shap_150 = []
choice_shap_185 = []
choice_shap_230 = []
choice_shap_345 = []
choice_shap_460 = []
choice_shap_650 = []
choice_shap_870 = []

for index in choice_features:
    choice_shap_values = shap_values[index]
    #print ("choice_shap_values",choice_shap_values)
    
    #shap_means = np.mean(np.abs(choice_shap_values), axis=0)
    
    shap_means = np.mean(np.abs(choice_shap_values), axis=0)*1./np.sum(np.mean(np.abs(choice_shap_values), axis=0)) # normalise - is this ok?
    #shap_stds = np.std(np.abs(choice_shap_values), axis=0)*1./np.sum(np.mean(np.abs(choice_shap_values), axis=0)) # normalise - is this ok?
    
    
    choice_shap_100.append(shap_means[0])
    choice_shap_150.append(shap_means[1])
    choice_shap_185.append(shap_means[2])
    choice_shap_230.append(shap_means[3])
    choice_shap_345.append(shap_means[4])
    choice_shap_460.append(shap_means[5])
    choice_shap_650.append(shap_means[6])
    choice_shap_870.append(shap_means[7])

    # print the results
    '''
    df = pd.DataFrame({
        "mean_abs_shap": np.mean(np.abs(choice_shap_values), axis=0),
        "stdev_abs_shap": np.std(np.abs(choice_shap_values), axis=0),
        "name": feature_names
    })
    print (df)
    '''
    
cm_subsection = np.linspace(0., 1., len(choice_wavelengths))
wavelength_colors = [cm.Dark2(x) for x in cm_subsection]

plt.plot(choice_features,choice_shap_100,color=wavelength_colors[0],label='100GHz')
plt.plot(choice_features,choice_shap_150,color=wavelength_colors[1],label='150GHz')
plt.plot(choice_features,choice_shap_185,color=wavelength_colors[2],label='185GHz')
plt.plot(choice_features,choice_shap_230,color=wavelength_colors[3],label='230GHz')
plt.plot(choice_features,choice_shap_345,color=wavelength_colors[4],label='345GHz')
plt.plot(choice_features,choice_shap_460,color=wavelength_colors[5],label='460GHz')
plt.plot(choice_features,choice_shap_650,color=wavelength_colors[6],label='650GHz')
plt.plot(choice_features,choice_shap_870,color=wavelength_colors[7],label='870GHz')

plt.xticks(choice_features, ("redshift","SFR_inst","SFR_10","SFR_100","M_dust","M_gas","M_star"))


axes = plt.gca()
#axes.set_yscale("log")
#axes.set_xscale("log")

plt.legend()
plt.show()


# make a shap summary plot. this shows from top to bottom the most important features in the model.
#shap.summary_plot(choice_shap_values, features=x_samples, feature_names = feature_names)

stop
# make shap dependence plots
# each dot is a single row from the dataset. The x axis shows the value of the feature, and the y axis represents how much knowing that feature changes the output for the model for that sample's prediction. Vertical dispersion of data represents interaction effects.
# the interaction index here (i.e. the color coding) is chosen automatically as what seems to be the strongest interaction.

for i in range(0,model_params):
    shap.dependence_plot(i, shap_values, x_samples, feature_names=feature_names)

# can also force the interaction index - by passing interaction_index
