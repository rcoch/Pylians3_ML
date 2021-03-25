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

feature_names=["redshift","SFR(inst)","SFR(10Myr)","SFR(100Myr)",r"$M_{\rm{dust}}$",r"$M_{\rm{gas}}$",r"$M_{\star}$"]
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
model = architecture.model_1hl(model_params, h1, 8, dropout_rate)

model.load_state_dict(torch.load(fmodel, map_location=torch.device(device)))
model.to(device=device)

#print ("shape of train loader", np.shape(train_loader))
# grab some data
model.eval()
for x_train,y_train in train_loader:
    #print ("here")
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
alma_freqs = np.array([100,150,185,230,345,460,650,870])
alma_wavelengths = 3*1e8*1./(alma_freqs*1e9)*1e6

choice_shap_redshift = []
choice_shap_sfr_inst = []
choice_shap_sfr_10 = []
choice_shap_sfr_100 = []
choice_shap_mdust = []
choice_shap_mgas = []
choice_shap_mstar = []

choice_shap_redshift_std = []
choice_shap_sfr_inst_std = []
choice_shap_sfr_10_std = []
choice_shap_sfr_100_std = []
choice_shap_mdust_std = []
choice_shap_mgas_std = []
choice_shap_mstar_std = []


for index in choice_wavelengths:
    choice_shap_values = shap_values[index]
    #print ("choice_shap_values",choice_shap_values)
    
    shap_means = np.mean(np.abs(choice_shap_values), axis=0)*1./np.sum(np.mean(np.abs(choice_shap_values), axis=0)) # normalise - is this ok?
    shap_stds = np.std(np.abs(choice_shap_values), axis=0)*1./np.sum(np.mean(np.abs(choice_shap_values), axis=0)) # normalise - is this ok?

    choice_shap_redshift.append(shap_means[0])
    choice_shap_sfr_inst.append(shap_means[1])
    choice_shap_sfr_10.append(shap_means[2])
    choice_shap_sfr_100.append(shap_means[3])
    choice_shap_mdust.append(shap_means[4])
    choice_shap_mgas.append(shap_means[5])
    choice_shap_mstar.append(shap_means[6])
    
    choice_shap_redshift_std.append(shap_stds[0])
    choice_shap_sfr_inst_std.append(shap_stds[1])
    choice_shap_sfr_10_std.append(shap_stds[2])
    choice_shap_sfr_100_std.append(shap_stds[3])
    choice_shap_mdust_std.append(shap_stds[4])
    choice_shap_mgas_std.append(shap_stds[5])
    choice_shap_mstar_std.append(shap_stds[6])
    

    # print the results
    print ("wavelength",alma_wavelengths[index])
    df = pd.DataFrame({
        "mean_abs_shap": np.mean(np.abs(choice_shap_values), axis=0),
        "stdev_abs_shap": np.std(np.abs(choice_shap_values), axis=0),
        "name": feature_names
    })
    
    shap.summary_plot(choice_shap_values, features=x_samples, feature_names = feature_names)
    #plt.savefig("summary_plot"+str(index)+".pdf")
    #plt.clf()
    print (df)
    
    
    
plt.figure(figsize=(5,5))
cm_subsection = np.linspace(0., 1., len(choice_wavelengths))
wavelength_colors = [cm.Dark2(x) for x in cm_subsection]

plt.plot(alma_wavelengths,choice_shap_redshift,color=wavelength_colors[0],label='redshift')
#plt.fill_between(alma_wavelengths,np.array(choice_shap_redshift)-np.array(choice_shap_redshift_std),np.array(choice_shap_redshift)+np.array(choice_shap_redshift_std),color=wavelength_colors[0],alpha=0.1)
plt.plot(alma_wavelengths,choice_shap_sfr_inst,color=wavelength_colors[1],label='SFR(inst)')
#plt.fill_between(alma_wavelengths,np.array(choice_shap_sfr_inst)-np.array(choice_shap_sfr_inst_std),np.array(choice_shap_sfr_inst)+np.array(choice_shap_sfr_inst_std),color=wavelength_colors[1],alpha=0.1)
plt.plot(alma_wavelengths,choice_shap_sfr_10,color=wavelength_colors[2],label='SFR(10Myr)')
#plt.fill_between(alma_wavelengths,np.array(choice_shap_sfr_10)-np.array(choice_shap_sfr_10_std),np.array(choice_shap_sfr_10)+np.array(choice_shap_sfr_10_std),color=wavelength_colors[2],alpha=0.1)
plt.plot(alma_wavelengths,choice_shap_sfr_100,color=wavelength_colors[3],label='SFR(100Myr)')
#plt.fill_between(alma_wavelengths,np.array(choice_shap_sfr_100)-np.array(choice_shap_sfr_100_std),np.array(choice_shap_sfr_100)+np.array(choice_shap_sfr_100_std),color=wavelength_colors[3],alpha=0.1)
plt.plot(alma_wavelengths,choice_shap_mdust,color=wavelength_colors[4],label=r'$M_{\rm{dust}}$')
#plt.fill_between(alma_wavelengths,np.array(choice_shap_mdust)-np.array(choice_shap_mdust_std),np.array(choice_shap_mdust)+np.array(choice_shap_mdust_std),color=wavelength_colors[4],alpha=0.1)
plt.plot(alma_wavelengths,choice_shap_mgas,color=wavelength_colors[5],label=r'$M_{\rm{gas}}$')
#plt.fill_between(alma_wavelengths,np.array(choice_shap_mgas)-np.array(choice_shap_mgas_std),np.array(choice_shap_mgas)+np.array(choice_shap_mgas_std),color=wavelength_colors[5],alpha=0.1)
plt.plot(alma_wavelengths,choice_shap_mstar,color=wavelength_colors[6],label=r'$M_{\rm{\star}}$')
#plt.fill_between(alma_wavelengths,np.array(choice_shap_mstar)-np.array(choice_shap_mstar_std),np.array(choice_shap_mstar)+np.array(choice_shap_mstar_std),color=wavelength_colors[6],alpha=0.1)

axes = plt.gca()
#axes.set_yscale("log")
axes.set_xscale("log")

plt.xlabel(r'$\lambda_{\rm{obs}}/\mu\rm{m}$',size=16)
plt.ylabel('Importance of feature',size=16)

#plt.legend()
plt.legend(loc='upper center',bbox_to_anchor=(0.5, 1.25),fancybox=False, shadow=False, ncol=3)
plt.tight_layout()
plt.savefig("importance_vs_wavelength.pdf")
plt.clf()

# make a shap summary plot. this shows from top to bottom the most important features in the model.
#shap.summary_plot(choice_shap_values, features=x_samples, feature_names = feature_names)

stop
# make shap dependence plots
# each dot is a single row from the dataset. The x axis shows the value of the feature, and the y axis represents how much knowing that feature changes the output for the model for that sample's prediction. Vertical dispersion of data represents interaction effects.
# the interaction index here (i.e. the color coding) is chosen automatically as what seems to be the strongest interaction.

for i in range(0,model_params):
    shap.dependence_plot(i, shap_values, x_samples, feature_names=feature_names)

# can also force the interaction index - by passing interaction_index
