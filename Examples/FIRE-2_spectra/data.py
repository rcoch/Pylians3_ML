import torch 
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import numpy as np
import sys, os, time, h5py

def normalize_data(data, labels):

    # normalize input
    data[:,0]  = (data[:,0] - np.mean(data[:,0]))/np.std(data[:,0])  #redshift
    data[:,1]  = (data[:,1] - np.mean(data[:,1]))/np.std(data[:,1])  #SFR
    data[:,2]  = (data[:,2] - np.mean(data[:,2]))/np.std(data[:,2])  #SFR_10
    data[:,3]  = (data[:,3] - np.mean(data[:,3]))/np.std(data[:,3])  #SFR_100
    data[:,4]  = (data[:,4] - np.mean(data[:,4]))/np.std(data[:,4])  #M_dust
    data[:,5]  = (data[:,5] - np.mean(data[:,5]))/np.std(data[:,5])  #M_gas
    data[:,6]  = (data[:,6] - np.mean(data[:,6]))/np.std(data[:,6])  #M_star

    # normalize labels
    print ("labels", labels)

    array = np.log10(1.0 + labels)
    print ("labels array", array)
    #labels = (array - np.mean(array))/np.std(array)
    #print ("shape of labels", np.shape(labels))
    labels = array
    
    return data, labels

# read data and get training, validation or testing sets
# fin ---------> file with the data
# seed --------> random seed used to split among different datasets
# mode --------> 'train', 'valid', 'test' or 'all'
# normalize ---> whether to normalize the data or not
def read_data(fin, seed, mode, normalize):

    # read data
    f     = h5py.File(fin, 'r')
    redshift    = f['A_all_z'][:];
    SFR    = f['A_all_sfr'][:];
    SFR_10    = f['A_all_sfr_10'][:];
    SFR_100   = f['A_all_sfr_100'][:];
    M_dust     = f['A_all_dust_mass'][:];
    M_gas  = f['A_all_gas_mass'][:];
    M_star  = f['A_all_stellar_mass'][:];
    #S850  = f['A_all_skirt_850'][:];
    wavelengths= f['A_all_wavelength'][:];
    fluxes = f['A_all_fluxes'][:];

    f.close()
    
    SFR = np.log10(1.0+SFR)
    SFR_10 = np.log10(1.0+SFR_10)
    SFR_100 = np.log10(1.0+SFR_100)
    M_dust = np.log10(1.0+M_dust)
    M_gas = np.log10(1.0+M_gas)
    M_star = np.log10(1.0+M_star)
    
    # get data, labels and number of elements
    data     = np.vstack([redshift, SFR, SFR_10, SFR_100, M_dust, M_gas, M_star]).T
    #labels   = S850.reshape((S850.shape[0],1))
    
    print ("shape of fluxes", np.shape(fluxes))
    labels = fluxes.reshape((fluxes.shape[0],90))
    #labels = fluxes
    elements = data.shape[0]

    # normalize data
    if normalize:  data, labels = normalize_data(data, labels)

    # get the size and offset depending on the type of dataset
    if   mode=='train':   size, offset = int(elements*0.70), int(elements*0.00)
    elif mode=='valid':   size, offset = int(elements*0.15), int(elements*0.70)
    elif mode=='test':    size, offset = int(elements*0.15), int(elements*0.85)
    elif mode=='all':     size, offset = int(elements*1.00), int(elements*0.00)
    else:                 raise Exception('Wrong name!')

    # randomly shuffle the cubes. Instead of 0 1 2 3...999 have a 
    # random permutation. E.g. 5 9 0 29...342
    np.random.seed(seed)
    indexes = np.arange(elements) 
    np.random.shuffle(indexes)
    indexes = indexes[offset:offset+size] #select indexes of mode

    return data[indexes], labels[indexes]


# This class creates the dataset 
class make_dataset():

    def __init__(self, mode, seed, fin):

        # get data
        inp, out = read_data(fin, seed, mode, normalize=True)

        # get the corresponding bottlenecks and parameters
        self.size   = inp.shape[0]
        self.input  = torch.tensor(inp, dtype=torch.float32)
        self.output = torch.tensor(out, dtype=torch.float32)
        
        print ("size of input and output", np.shape(self.input), np.shape(self.output))
        
    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.input[idx], self.output[idx]


# This routine creates a dataset loader
def create_dataset(mode, seed, fin, batch_size):
    data_set = make_dataset(mode, seed, fin)
    dataset_loader = DataLoader(dataset=data_set, batch_size=batch_size, shuffle=True)
    return dataset_loader
