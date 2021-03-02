import numpy as np
import torch
import sys,os,h5py
sys.path.append('../')
import data as data
import architecture
import matplotlib.pyplot as plt


#################################### INPUT ##########################################
# data parameters
fin  = 'my_fire2_data.h5'
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
pred = np.zeros((size,1), dtype=np.float32)
true = np.zeros((size,1), dtype=np.float32)

# get the parameters of the trained model
model = architecture.model_1hl(7, h1, 1, dropout_rate)

model.load_state_dict(torch.load(fmodel, map_location=torch.device(device)))
model.to(device=device)

# loop over the different batches and get the prediction
offset = 0
model.eval()
for x, y in test_loader:
    with torch.no_grad():
        x    = x.to(device)
        y    = y.to(device)
        y_NN = model(x)
        length = x.shape[0]
        pred[offset:offset+length] = y_NN.cpu().numpy()
        true[offset:offset+length] = y.cpu().numpy()
        offset += length

# get mean and std of S_850
f = h5py.File(fin, 'r')
S850 = f['A_all_skirt_850'][:]
f.close()

S850 = np.log10(1.0 + S850)
mean, std = np.mean(S850), np.std(S850)
print("std",std)

# compute the rmse; de-normalize
error_norm = ((pred - true))**2
pred  = pred*std + mean
true  = true*std + mean
error = (pred - true)**2

print('Error^2 norm      = %.3e'%np.mean(error_norm))
print('Error             = %.3e'%np.sqrt(np.mean(error)))

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
