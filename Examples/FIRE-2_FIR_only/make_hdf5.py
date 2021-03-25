import numpy as np
import h5py
import matplotlib.pyplot as plt

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx
    
data_path = 'FIRE_data/'

# load A1 data
A1_z = np.load(data_path+'full_z_arr_1.npy')
A1_sfr = np.load(data_path+'full_sfr_arr_A1.npy')
A1_sfr_10 = np.load(data_path+'A1_SFR_10Myr.npy')
A1_sfr_100 = np.load(data_path+'A1_SFR_100Myr.npy')
A1_dust_mass = np.load(data_path+'full_dustmass_arr_A1.npy')
A1_gas_mass = np.load(data_path+'full_gasmass_arr_A1.npy')
A1_stellar_mass = np.load(data_path+'full_stellarmass_arr_A1.npy')
A1_skirt_850 = np.load(data_path+'flux_arr_A1.npy')
A1_wavelengths = np.load(data_path+'wavelength_arr_A1.npy')
A1_fluxes = np.load(data_path+'flux_orien_0_A1.npy')

print ("A1_fluxes",np.shape(A1_fluxes))

# scale to solar mass units
A1_stellar_mass = A1_stellar_mass*1e10
A1_gas_mass = A1_gas_mass*1e10

# load A2 data
A2_z = np.load(data_path+'full_z_arr_2.npy')
A2_sfr = np.load(data_path+'full_sfr_arr_A2.npy')
A2_sfr_10 = np.load(data_path+'A2_SFR_10Myr.npy')
A2_sfr_100 = np.load(data_path+'A2_SFR_100Myr.npy')
A2_dust_mass = np.load(data_path+'full_dustmass_arr_A2.npy')
A2_gas_mass = np.load(data_path+'full_gasmass_arr_A2.npy')
A2_stellar_mass = np.load(data_path+'full_stellarmass_arr_A2.npy')
A2_skirt_850 = np.load(data_path+'flux_arr_A2.npy')
A2_wavelengths = np.load(data_path+'wavelength_arr_A2.npy')
A2_fluxes = np.load(data_path+'flux_orien_0_A2.npy')

# scale to solar mass units
A2_stellar_mass = A2_stellar_mass*1e10
A2_gas_mass = A2_gas_mass*1e10

# load A4 data
A4_z = np.load(data_path+'full_z_arr_4.npy')
A4_sfr = np.load(data_path+'full_sfr_arr_A4.npy')
A4_sfr_10 = np.load(data_path+'A4_SFR_10Myr.npy')
A4_sfr_100 = np.load(data_path+'A4_SFR_100Myr.npy')
A4_dust_mass = np.load(data_path+'full_dustmass_arr_A4.npy')
A4_gas_mass = np.load(data_path+'full_gasmass_arr_A4.npy')
A4_stellar_mass = np.load(data_path+'full_stellarmass_arr_A4.npy')
A4_skirt_850 = np.load(data_path+'flux_arr_A4.npy')
A4_wavelengths = np.load(data_path+'wavelength_arr_A4.npy')
A4_fluxes = np.load(data_path+'flux_orien_0_A4.npy')

# scale to solar mass units
A4_stellar_mass = A4_stellar_mass*1e10
A4_gas_mass = A4_gas_mass*1e10

# load A8 data
A8_z = np.load(data_path+'full_z_arr_8.npy')
A8_sfr = np.load(data_path+'full_sfr_arr_A8.npy')
A8_sfr_10 = np.load(data_path+'A8_SFR_10Myr.npy')
A8_sfr_100 = np.load(data_path+'A8_SFR_100Myr.npy')
A8_dust_mass = np.load(data_path+'full_dustmass_arr_A8.npy')
A8_gas_mass = np.load(data_path+'full_gasmass_arr_A8.npy')
A8_stellar_mass = np.load(data_path+'full_stellarmass_arr_A8.npy')
A8_skirt_850 = np.load(data_path+'flux_arr_A8.npy')
A8_wavelengths = np.load(data_path+'wavelength_arr_A8.npy')
A8_fluxes = np.load(data_path+'flux_orien_0_A8.npy')

# scale to solar mass units
A8_stellar_mass = A8_stellar_mass*1e10
A8_gas_mass = A8_gas_mass*1e10

# reshape SKIRT values
A1_skirt_850 = A1_skirt_850.astype(np.float)
A2_skirt_850 = A2_skirt_850.astype(np.float)
A4_skirt_850 = A4_skirt_850.astype(np.float)
A8_skirt_850 = A8_skirt_850.astype(np.float)

print ('redshifts', A1_z[34:])
# now join all the arrays
A_all_z_array = np.append(A1_z[34:],A2_z[34:])
A_all_z_array = np.append(A_all_z_array,A4_z[34:])
A_all_z_array = np.append(A_all_z_array,A8_z[34:])

A_all_sfr_array = np.append(A1_sfr[34:],A2_sfr[34:])
A_all_sfr_array = np.append(A_all_sfr_array,A4_sfr[34:])
A_all_sfr_array = np.append(A_all_sfr_array,A8_sfr[34:])

A_all_sfr_10_array = np.append(A1_sfr_10[34:],A2_sfr_10[34:])
A_all_sfr_10_array = np.append(A_all_sfr_10_array,A4_sfr_10[34:])
A_all_sfr_10_array = np.append(A_all_sfr_10_array,A8_sfr_10[34:])

A_all_sfr_100_array = np.append(A1_sfr_100[34:],A2_sfr_100[34:])
A_all_sfr_100_array = np.append(A_all_sfr_100_array,A4_sfr_100[34:])
A_all_sfr_100_array = np.append(A_all_sfr_100_array,A8_sfr_100[34:])

A_all_dust_mass_array = np.append(A1_dust_mass[34:],A2_dust_mass[34:])
A_all_dust_mass_array = np.append(A_all_dust_mass_array,A4_dust_mass[34:])
A_all_dust_mass_array = np.append(A_all_dust_mass_array,A8_dust_mass[34:])

A_all_gas_mass_array = np.append(A1_gas_mass[34:],A2_gas_mass[34:])
A_all_gas_mass_array = np.append(A_all_gas_mass_array,A4_gas_mass[34:])
A_all_gas_mass_array = np.append(A_all_gas_mass_array,A8_gas_mass[34:])

A_all_stellar_mass_array = np.append(A1_stellar_mass[34:],A2_stellar_mass[34:])
A_all_stellar_mass_array = np.append(A_all_stellar_mass_array,A4_stellar_mass[34:])
A_all_stellar_mass_array = np.append(A_all_stellar_mass_array,A8_stellar_mass[34:])

A_all_skirt_850_array = np.append(A1_skirt_850[34:],A2_skirt_850[34:])
A_all_skirt_850_array = np.append(A_all_skirt_850_array,A4_skirt_850[34:])
A_all_skirt_850_array = np.append(A_all_skirt_850_array,A8_skirt_850[34:])

# necessary to specify the axis as this the wavelength and fluxes are 2d arrays
A_all_wavelength_array = np.append(A1_wavelengths[34:],A2_wavelengths[34:],axis=0)
A_all_wavelength_array = np.append(A_all_wavelength_array,A4_wavelengths[34:],axis=0)
A_all_wavelength_array = np.append(A_all_wavelength_array,A8_wavelengths[34:],axis=0)

A_all_fluxes_array = np.append(A1_fluxes[34:],A2_fluxes[34:],axis=0)
A_all_fluxes_array = np.append(A_all_fluxes_array,A4_fluxes[34:],axis=0)
A_all_fluxes_array = np.append(A_all_fluxes_array,A8_fluxes[34:],axis=0)


for i in range(0,len(A_all_z_array)-3):
    if A_all_sfr_array[i]>-1 and A_all_sfr_10_array[i]>-1 and A_all_sfr_100_array[i]>-1 and A_all_dust_mass_array[i]>-1 and A_all_gas_mass_array[i]>-1 and A_all_stellar_mass_array[i]>-1 and A_all_skirt_850_array[i]>-1:
        pass
    else:
        print ("nan here", i)
        A_all_z_array = np.delete(A_all_z_array, i)
        A_all_sfr_array = np.delete(A_all_sfr_array, i)
        A_all_sfr_10_array = np.delete(A_all_sfr_10_array, i)
        A_all_sfr_100_array = np.delete(A_all_sfr_100_array, i)
        A_all_dust_mass_array = np.delete(A_all_dust_mass_array, i)
        A_all_gas_mass_array = np.delete(A_all_gas_mass_array, i)
        A_all_stellar_mass_array = np.delete(A_all_stellar_mass_array, i)
        A_all_skirt_850_array = np.delete(A_all_skirt_850_array, i)
        A_all_wavelength_array = np.delete(A_all_wavelength_array,i,axis=0)
        A_all_fluxes_array = np.delete(A_all_fluxes_array,i,axis=0)


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx
    
alma_freqs = np.array([100,150,185,230,345,460,650,870])
alma_wavelengths = 3*1e8*1./(alma_freqs*1e9)*1e6
print ("alma_wavelengths in microns", alma_wavelengths)

alma_fir_phot_all = []

for i in range(0,len(A_all_z_array)):
    old_wavelengths_arr=A_all_wavelength_array[i]
    old_fluxes_arr=A_all_fluxes_array[i]
    
    index_0 = find_nearest(old_wavelengths_arr*(1+A_all_z_array[i]),alma_wavelengths[0])
    alma_100 = old_fluxes_arr[index_0]*(1+A_all_z_array[i])

    index_1 = find_nearest(old_wavelengths_arr*(1+A_all_z_array[i]),alma_wavelengths[1])
    alma_150 = old_fluxes_arr[index_1]*(1+A_all_z_array[i])
    
    index_2 = find_nearest(old_wavelengths_arr*(1+A_all_z_array[i]),alma_wavelengths[2])
    alma_185 = old_fluxes_arr[index_2]*(1+A_all_z_array[i])
    
    index_3 = find_nearest(old_wavelengths_arr*(1+A_all_z_array[i]),alma_wavelengths[3])
    alma_230 = old_fluxes_arr[index_3]*(1+A_all_z_array[i])
    
    index_4 = find_nearest(old_wavelengths_arr*(1+A_all_z_array[i]),alma_wavelengths[4])
    alma_345 = old_fluxes_arr[index_4]*(1+A_all_z_array[i])
    
    index_5 = find_nearest(old_wavelengths_arr*(1+A_all_z_array[i]),alma_wavelengths[5])
    alma_460 = old_fluxes_arr[index_5]*(1+A_all_z_array[i])
    
    index_6 = find_nearest(old_wavelengths_arr*(1+A_all_z_array[i]),alma_wavelengths[6])
    alma_650 = old_fluxes_arr[index_6]*(1+A_all_z_array[i])
    
    index_7 = find_nearest(old_wavelengths_arr*(1+A_all_z_array[i]),alma_wavelengths[7])
    alma_870 = old_fluxes_arr[index_7]*(1+A_all_z_array[i])
    
    alma_fir_phot = [alma_100,alma_150,alma_185,alma_230,alma_345,alma_460,alma_650,alma_870]
    alma_fir_phot_all.append(alma_fir_phot)

# put these into a .h5 file
hf = h5py.File('my_fire2_data_fir_subset.h5', 'w')
hf.create_dataset('A_all_z', data=A_all_z_array)
hf.create_dataset('A_all_sfr', data=A_all_sfr_array)
hf.create_dataset('A_all_sfr_10', data=A_all_sfr_10_array)
hf.create_dataset('A_all_sfr_100', data=A_all_sfr_100_array)
hf.create_dataset('A_all_dust_mass', data=A_all_dust_mass_array)
hf.create_dataset('A_all_gas_mass', data=A_all_gas_mass_array)
hf.create_dataset('A_all_stellar_mass', data=A_all_stellar_mass_array)
hf.create_dataset('A_all_alma_fir_phot', data=alma_fir_phot_all)
hf.close()

my_fire2_data = "my_fire2_data_fir_subset.h5"
f = h5py.File(my_fire2_data,'r')
A_all_z = f['A_all_z']

A_all_alma_fir_phot = f['A_all_alma_fir_phot']
print ("A_all_alma_fir_phot - shape correct",np.shape(A_all_alma_fir_phot))


for i in range(0,100):
    spectrum = A_all_alma_fir_phot[i]
    plt.plot(np.log10(alma_wavelengths),np.log10(spectrum))

plt.show()
