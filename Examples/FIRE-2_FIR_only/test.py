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
model = architecture.model_1hl(7, h1, 8, dropout_rate)

model.load_state_dict(torch.load(fmodel, map_location=torch.device(device)))
model.to(device=device)

alma_freqs = np.array([100,150,185,230,345,460,650,870])
wavelengths_correct = 3*1e8*1./(alma_freqs*1e9)*1e6

# get redshift normalisation factor
f     = h5py.File(fin, 'r')
redshift    = f['A_all_z'][:]
print ("redshift", redshift)
mean_redshift = np.mean(redshift)
std_redshift = np.std(redshift)

# loop over the different batches and get the prediction
offset = 0
model.eval()
for x, y in test_loader:
    with torch.no_grad():
        x    = x.to(device)
        y    = y.to(device)
        y_NN = model(x)
        
        for i in range(0,2):#len(x)):
            print ("input params", x[i])
            redshift_here = x[i][0].numpy()
            denormed_redshift_here = redshift_here*std_redshift + mean_redshift
            #redshift_here = np.power(10,redshift_here)-1
            modelled_spectrum = model(x[i]).numpy()
            
            plt.plot(wavelengths_correct,model(x[i]),'r-',label='NN estimate')
            plt.plot(wavelengths_correct,y[i],'b-',label='SKIRT output, z='+str(np.round(denormed_redshift_here,2)))

            axes = plt.gca()
            axes.set_ylim([1e-6,10])
            plt.yscale("log")
            plt.xlabel(r'$\lambda_{\rm{obs}}/\mu m$',size=16)
            plt.ylabel(r'$S_{\rm{obs}}/mJy$',size=16)
            plt.legend()
            plt.show()
            plt.clf()
        
        print ('shape of each output')
        print (np.shape(x))
        print (np.shape(y))
        print (np.shape(y_NN))




    
cm_subsection = np.linspace(0., 1., len(wavelengths_correct)+3)
wavelength_colors = [cm.gist_heat_r(x) for x in cm_subsection]
wavelengths_labels = np.round(wavelengths_correct,0).astype(int)

for j in range(0,len(wavelengths_correct)):

    model.eval()
    for x, y in test_loader:
        with torch.no_grad():
            x    = x.to(device)
            y    = y.to(device)
            y_NN = model(x)
            
            x_array = []
            y_array = []
            for i in range(0,len(x)):
                print ("input params", x[i])

                modelled_spectrum = model(x[i]).numpy()
                x_array.append(y[i][j])
                y_array.append(model(x[i])[j])

        plt.plot(x_array,y_array,'.', color=wavelength_colors[j+1],label=str(wavelengths_labels[j])+r"$\mu\rm{m}$")
        print ("number of points being plotted here", len(x_array))

    
        if j>1:
            threshold=0.01
        else:
            threshold=0.01

        pos_indices = np.where(np.array(x_array)>threshold)
        print ("percentage of predicted fluxes that are positive", len(np.array(y_array)[pos_indices])*100./len(y_array))
        #print ("y_array",y_array)
        pred_div_true = np.array(y_array)[pos_indices]*1./np.array(x_array)[pos_indices]
        mean_pred_div_true = np.mean(pred_div_true)
        median_pred_div_true = np.median(pred_div_true)
        std_pred_div_true = np.std(pred_div_true)
        print ("mean,median,std", mean_pred_div_true,median_pred_div_true,std_pred_div_true)
    
        axes = plt.gca()
        plt.text(0.02,2e-4,"mean, std: "+str(np.round(mean_pred_div_true,2))+", "+str(np.round(std_pred_div_true,2)),fontsize=16)

        plt.yscale("log")
        plt.xscale("log")

        plt.ylim([1e-4,10])
        plt.xlim([1e-4,10])
        
        xline = np.logspace(-4,1)
        plt.plot(xline,xline,'k--')
        
        plt.xlabel(r'$S_{\rm{SKIRT}}/mJy$',size=16)
        plt.ylabel(r'$S_{\rm{NN}}/mJy$',size=16)
        plt.legend(fontsize=16)
        plt.tight_layout()
        #plt.show()
        plt.savefig("nn_performance_wavlength_"+str(j)+".pdf")
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
