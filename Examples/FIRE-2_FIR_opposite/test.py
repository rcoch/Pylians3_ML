import numpy as np
import torch
import sys,os,h5py
sys.path.append('../')
import data as data
import architecture
import matplotlib.pyplot as plt
from matplotlib import cm


#################################### INPUT ##########################################
# data parameters
fin  = 'my_fire2_data_fir_subset.h5'
seed = 5

# architecture parameters
h1 = 2000
dropout_rate = 0.3

# training parameters
batch_size = 256

# name of output files
name   = '1hd_100_0.0_0.0'
fout   = 'results/%s.txt'%name
fmodel = 'models/1hd_2000_0.3_1e-5.pt'#models/%s.pt'%name
#####################################################################################

# get GPU if available
if torch.cuda.is_available():
    print("CUDA Available")
    device = torch.device('cuda')
else:
    print("CUDA Not Available")
    device = torch.device('cpu')

# get the test dataset
test_loader  = data.create_dataset('test', seed, fin, batch_size)

# get the number of elements in the test set
size = 0
for x, y in test_loader:
    size += x.shape[0]

# define the array with the results
pred = np.zeros((size), dtype=np.float32)
true = np.zeros((size), dtype=np.float32)

# get the parameters of the trained model
model = architecture.model_1hl(8, h1, 7, dropout_rate)

model.load_state_dict(torch.load(fmodel, map_location=torch.device(device)))
model.to(device=device)

alma_freqs = np.array([100,150,185,230,345,460,650,870])
wavelengths_correct = 3*1e8*1./(alma_freqs*1e9)*1e6

wavelengths_correct = np.linspace(0,8,7)

f     = h5py.File(fin, 'r')
redshift    = f['A_all_z'][:]
SFR    = f['A_all_sfr'][:];
SFR_10    = f['A_all_sfr_10'][:];
SFR_100   = f['A_all_sfr_100'][:];
M_dust     = f['A_all_dust_mass'][:];
M_gas  = f['A_all_gas_mass'][:];
M_star  = f['A_all_stellar_mass'][:];
fluxes = f['A_all_alma_fir_phot'][:];
phys_array = np.vstack([redshift,SFR,SFR_10,SFR_100,M_dust,M_gas,M_star]).T
        

cm_subsection = np.linspace(0., 1., len(wavelengths_correct)+3)
wavelength_colors = [cm.gist_heat_r(x) for x in cm_subsection]

labels = ["redshift", "SFR (inst)", "SFR (10Myr)", "SFR (100Myr)", "Mdust", "Mgas", "Mstellar"]

for j in range(0,len(wavelengths_correct)):

    model.eval()
    for x, y in test_loader:
        with torch.no_grad():
            x    = x.to(device)
            y    = y.to(device)
            y_NN = model(x)
            
            my_x_array = []
            my_y_array = []
            
            for i in range(0,len(x)):
                print ("input params", x[i])
                modelled_spectrum = model(x[i]).numpy()
                
                if j==0:
                    my_x_array.append(y[i][j])
                    my_y_array.append(model(x[i])[j])
                if j>0:
                    my_x_array.append(np.power(10,y[i][j])-1)
                    my_y_array.append(np.power(10,model(x[i])[j])-1)
                
            plt.plot(my_x_array,my_y_array,'.', color=wavelength_colors[j+1],label=labels[j])
             
            line_to_plot = np.linspace(np.min(my_x_array),np.max(my_x_array),10)
            plt.plot(line_to_plot,line_to_plot,'k--')
            
            if j>0:
                plt.yscale("log")
                plt.xscale("log")

            plt.xlabel(r'$\rm{quant}_{\rm{SKIRT}}$',size=16)
            plt.ylabel(r'$\rm{quant}_{\rm{NN}}$',size=16)
            plt.legend()
        plt.show()
        plt.clf()
    
stop

# get mean and std of S_850
f = h5py.File(fin, 'r')
fluxes = f['A_all_fluxes'][:]
f.close()

#S850 = np.log10(1.0 + S850)
#mean, std = np.mean(S850), np.std(S850)
#print("std",std)

# compute the rmse; de-normalize
#error_norm = ((pred - true))**2
#pred  = pred*std + mean
#true  = true*std + mean
#error = (pred - true)**2

#print('Error^2 norm      = %.3e'%np.mean(error_norm))
#print('Error             = %.3e'%np.sqrt(np.mean(error)))

plt.plot(pred,true,'r.')
#plt.plot(np.log10(pred),np.log10(true),'r.')
plt.show()

pred_new = np.power(10,pred)-1
true_new = np.power(10,true)-1

print ("number of points plotted", len(pred_new))

plt.plot(pred_new,true_new,'r.')
x = np.linspace(0,2)
plt.plot(x,x,'k--')
axes=plt.gca()
axes.set_xlabel(r"$S_{850,\rm{predicted,\,NN}}$",fontsize=16)
axes.set_ylabel(r"$S_{850,\rm{true,\,SKIRT}}$",fontsize=16)

plt.show()

# save results to file
#results = np.zeros((size,10))
#results[:,0:5]  = true
#results[:,5:10] = pred
#np.savetxt(fout, results)


print (np.shape(pred))

'''
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

print(confusion_matrix(pred,true))
print(classification_report(pred,true))
print(accuracy_score(pred,true))
'''
