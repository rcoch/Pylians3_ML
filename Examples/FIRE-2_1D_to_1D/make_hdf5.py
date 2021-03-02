import numpy as np
import h5py

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

# scale to solar mass units
A8_stellar_mass = A8_stellar_mass*1e10
A8_gas_mass = A8_gas_mass*1e10

# reshape SKIRT values
A1_skirt_850 = A1_skirt_850.astype(np.float)
A2_skirt_850 = A2_skirt_850.astype(np.float)
A4_skirt_850 = A4_skirt_850.astype(np.float)
A8_skirt_850 = A8_skirt_850.astype(np.float)

# now join all the arrays
A_all_z_array = np.append(A1_z,A2_z)
A_all_z_array = np.append(A_all_z_array,A4_z)
A_all_z_array = np.append(A_all_z_array,A8_z)

A_all_sfr_array = np.append(A1_sfr,A2_sfr)
A_all_sfr_array = np.append(A_all_sfr_array,A4_sfr)
A_all_sfr_array = np.append(A_all_sfr_array,A8_sfr)

A_all_sfr_10_array = np.append(A1_sfr_10,A2_sfr_10)
A_all_sfr_10_array = np.append(A_all_sfr_10_array,A4_sfr_10)
A_all_sfr_10_array = np.append(A_all_sfr_10_array,A8_sfr_10)

A_all_sfr_100_array = np.append(A1_sfr_100,A2_sfr_100)
A_all_sfr_100_array = np.append(A_all_sfr_100_array,A4_sfr_100)
A_all_sfr_100_array = np.append(A_all_sfr_100_array,A8_sfr_100)

A_all_dust_mass_array = np.append(A1_dust_mass,A2_dust_mass)
A_all_dust_mass_array = np.append(A_all_dust_mass_array,A4_dust_mass)
A_all_dust_mass_array = np.append(A_all_dust_mass_array,A8_dust_mass)

A_all_gas_mass_array = np.append(A1_gas_mass,A2_gas_mass)
A_all_gas_mass_array = np.append(A_all_gas_mass_array,A4_gas_mass)
A_all_gas_mass_array = np.append(A_all_gas_mass_array,A8_gas_mass)

A_all_stellar_mass_array = np.append(A1_stellar_mass,A2_stellar_mass)
A_all_stellar_mass_array = np.append(A_all_stellar_mass_array,A4_stellar_mass)
A_all_stellar_mass_array = np.append(A_all_stellar_mass_array,A8_stellar_mass)

A_all_skirt_850_array = np.append(A1_skirt_850,A2_skirt_850)
A_all_skirt_850_array = np.append(A_all_skirt_850_array,A4_skirt_850)
A_all_skirt_850_array = np.append(A_all_skirt_850_array,A8_skirt_850)


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

# put these into a .h5 file
hf = h5py.File('my_fire2_data.h5', 'w')
hf.create_dataset('A_all_z', data=A_all_z_array)
hf.create_dataset('A_all_sfr', data=A_all_sfr_array)
hf.create_dataset('A_all_sfr_10', data=A_all_sfr_10_array)
hf.create_dataset('A_all_sfr_100', data=A_all_sfr_100_array)
hf.create_dataset('A_all_dust_mass', data=A_all_dust_mass_array)
hf.create_dataset('A_all_gas_mass', data=A_all_gas_mass_array)
hf.create_dataset('A_all_stellar_mass', data=A_all_stellar_mass_array)
hf.create_dataset('A_all_skirt_850', data=A_all_skirt_850_array)
hf.close()

my_fire2_data = "my_fire2_data.h5"
f = h5py.File(my_fire2_data,'r')
A_all_z = f['A_all_z']
A_all_sfr = f['A_all_sfr']
A_all_sfr_10 = f['A_all_sfr_10']

print (np.shape(A_all_z))
print (np.shape(A_all_sfr))
print (np.shape(A_all_sfr_10))

print ("array", A_all_sfr_array)
