import numpy as np
import torch
import sys,os,h5py
sys.path.append('../')
import data as data
import architecture
import matplotlib.pyplot as plt


#################################### INPUT ##########################################
# data parameters
fin  = 'my_fire2_data_spectrum.h5'
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
model = architecture.model_1hl(7, h1, 90, dropout_rate)

model.load_state_dict(torch.load(fmodel, map_location=torch.device(device)))
model.to(device=device)

wavelengths_file= "/Users/rachelcochrane/Documents/Pylians/prepping_FIRE_data/wave_orig.dat"
# load in these wavelengths as an array
wavelengths_correct = np.genfromtxt(wavelengths_file, dtype=None,delimiter=' ')
wavelengths_correct = np.log10(wavelengths_correct)

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
        
        for i in range(0,len(x)):
            print ("input params", x[i])
            redshift_here = x[i][0].numpy()
            denormed_redshift_here = redshift_here*std_redshift + mean_redshift
            #redshift_here = np.power(10,redshift_here)-1
            modelled_spectrum = model(x[i]).numpy()

            plt.plot(wavelengths_correct,model(x[i]),'r-',label='NN estimate')
            plt.plot(wavelengths_correct,y[i],'b-',label='SKIRT output, z='+str(denormed_redshift_here))

            axes = plt.gca()
            axes.set_ylim([1e-6,10])
            plt.yscale("log")
            plt.xlabel(r'$\lambda_{\rm{rest}}/\mu m$',size=16)
            plt.ylabel(r'$S_{\rm{rest}}/mJy$',size=16)
            plt.legend()
            plt.show()
            plt.clf()
        
        print ('shape of each output')
        print (np.shape(x))
        print (np.shape(y))
        print (np.shape(y_NN))

        '''
        length = x.shape[0]
        a = torch.reshape(y_NN, [-1])
        pred[offset:offset+length] = a.cpu().numpy()
        #a = torch.reshape(y_NN, [-1])
        #print (a)
        #pred[offset:offset+length] = a.cpu().numpy()
        print ("pred", pred)
        
        print ("y", np.shape(y))
        b = torch.reshape(y, [-1])
        true[offset:offset+length] = b.cpu().numpy()
        print ("true", true)
        offset += length
        '''
        
stop
for i in range(0,len(pred)):
    plt.plot(pred[i],true[i],'r.')
    plt.show()

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
