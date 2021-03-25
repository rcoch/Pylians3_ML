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


A1_wavelengths = np.load(data_path+'wavelength_arr_A1.npy')
A1_fluxes_0 = np.load(data_path+'flux_orien_0_A1.npy')
A1_fluxes_1 = np.load(data_path+'flux_orien_1_A1.npy')
A1_fluxes_2 = np.load(data_path+'flux_orien_2_A1.npy')
A1_fluxes_3 = np.load(data_path+'flux_orien_3_A1.npy')
A1_fluxes_4 = np.load(data_path+'flux_orien_4_A1.npy')
A1_fluxes_5 = np.load(data_path+'flux_orien_5_A1.npy')
A1_fluxes_6 = np.load(data_path+'flux_orien_6_A1.npy')

print ("A1_fluxes_0",np.shape(A1_fluxes_0))

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
A2_fluxes_0 = np.load(data_path+'flux_orien_0_A2.npy')
A2_fluxes_1 = np.load(data_path+'flux_orien_1_A2.npy')
A2_fluxes_2 = np.load(data_path+'flux_orien_2_A2.npy')
A2_fluxes_3 = np.load(data_path+'flux_orien_3_A2.npy')
A2_fluxes_4 = np.load(data_path+'flux_orien_4_A2.npy')
A2_fluxes_5 = np.load(data_path+'flux_orien_5_A2.npy')
A2_fluxes_6 = np.load(data_path+'flux_orien_6_A2.npy')

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
A4_fluxes_0 = np.load(data_path+'flux_orien_0_A4.npy')
A4_fluxes_1 = np.load(data_path+'flux_orien_1_A4.npy')
A4_fluxes_2 = np.load(data_path+'flux_orien_2_A4.npy')
A4_fluxes_3 = np.load(data_path+'flux_orien_3_A4.npy')
A4_fluxes_4 = np.load(data_path+'flux_orien_4_A4.npy')
A4_fluxes_5 = np.load(data_path+'flux_orien_5_A4.npy')
A4_fluxes_6 = np.load(data_path+'flux_orien_6_A4.npy')

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
A8_fluxes_0 = np.load(data_path+'flux_orien_0_A8.npy')
A8_fluxes_1 = np.load(data_path+'flux_orien_1_A8.npy')
A8_fluxes_2 = np.load(data_path+'flux_orien_2_A8.npy')
A8_fluxes_3 = np.load(data_path+'flux_orien_3_A8.npy')
A8_fluxes_4 = np.load(data_path+'flux_orien_4_A8.npy')
A8_fluxes_5 = np.load(data_path+'flux_orien_5_A8.npy')
A8_fluxes_6 = np.load(data_path+'flux_orien_6_A8.npy')

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

A_all_fluxes_0_array = np.append(A1_fluxes_0[34:],A2_fluxes_0[34:],axis=0)
A_all_fluxes_0_array = np.append(A_all_fluxes_0_array,A4_fluxes_0[34:],axis=0)
A_all_fluxes_0_array = np.append(A_all_fluxes_0_array,A8_fluxes_0[34:],axis=0)

A_all_fluxes_1_array = np.append(A1_fluxes_1[34:],A2_fluxes_1[34:],axis=0)
A_all_fluxes_1_array = np.append(A_all_fluxes_1_array,A4_fluxes_1[34:],axis=0)
A_all_fluxes_1_array = np.append(A_all_fluxes_1_array,A8_fluxes_1[34:],axis=0)

A_all_fluxes_2_array = np.append(A1_fluxes_2[34:],A2_fluxes_2[34:],axis=0)
A_all_fluxes_2_array = np.append(A_all_fluxes_2_array,A4_fluxes_2[34:],axis=0)
A_all_fluxes_2_array = np.append(A_all_fluxes_2_array,A8_fluxes_2[34:],axis=0)

A_all_fluxes_3_array = np.append(A1_fluxes_3[34:],A2_fluxes_3[34:],axis=0)
A_all_fluxes_3_array = np.append(A_all_fluxes_3_array,A4_fluxes_3[34:],axis=0)
A_all_fluxes_3_array = np.append(A_all_fluxes_3_array,A8_fluxes_3[34:],axis=0)

A_all_fluxes_4_array = np.append(A1_fluxes_4[34:],A2_fluxes_4[34:],axis=0)
A_all_fluxes_4_array = np.append(A_all_fluxes_4_array,A4_fluxes_4[34:],axis=0)
A_all_fluxes_4_array = np.append(A_all_fluxes_4_array,A8_fluxes_4[34:],axis=0)

A_all_fluxes_5_array = np.append(A1_fluxes_5[34:],A2_fluxes_5[34:],axis=0)
A_all_fluxes_5_array = np.append(A_all_fluxes_5_array,A4_fluxes_5[34:],axis=0)
A_all_fluxes_5_array = np.append(A_all_fluxes_5_array,A8_fluxes_5[34:],axis=0)

A_all_fluxes_6_array = np.append(A1_fluxes_6[34:],A2_fluxes_6[34:],axis=0)
A_all_fluxes_6_array = np.append(A_all_fluxes_6_array,A4_fluxes_6[34:],axis=0)
A_all_fluxes_6_array = np.append(A_all_fluxes_6_array,A8_fluxes_6[34:],axis=0)



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
        A_all_wavelength_array = np.delete(A_all_wavelength_array,i,axis=0)
        A_all_fluxes_0_array = np.delete(A_all_fluxes_0_array,i,axis=0)
        A_all_fluxes_1_array = np.delete(A_all_fluxes_1_array,i,axis=0)
        A_all_fluxes_2_array = np.delete(A_all_fluxes_2_array,i,axis=0)
        A_all_fluxes_3_array = np.delete(A_all_fluxes_3_array,i,axis=0)
        A_all_fluxes_4_array = np.delete(A_all_fluxes_4_array,i,axis=0)
        A_all_fluxes_5_array = np.delete(A_all_fluxes_5_array,i,axis=0)
        A_all_fluxes_6_array = np.delete(A_all_fluxes_6_array,i,axis=0)

print ("shape of wavelengths array - correct", np.shape(A_all_wavelength_array))
print ("shape of fluxes array - correct", np.shape(A_all_fluxes_0_array))


# remove the wavelengths that aren't present in every file from the wavelengths and fluxes array
wavelengths_file= "/Users/rachelcochrane/Documents/Pylians/prepping_FIRE_data/wave_orig.dat"
# load in these wavelengths as an array
wavelengths_correct = np.genfromtxt(wavelengths_file, dtype=None,delimiter=' ')

A_all_wavelength_array_const_wavelengths = []
A_all_fluxes_array_0_const_wavelengths = []
A_all_fluxes_array_1_const_wavelengths = []
A_all_fluxes_array_2_const_wavelengths = []
A_all_fluxes_array_3_const_wavelengths = []
A_all_fluxes_array_4_const_wavelengths = []
A_all_fluxes_array_5_const_wavelengths = []
A_all_fluxes_array_6_const_wavelengths = []

for i in range(0,len(A_all_z_array)):
    old_wavelengths_arr = A_all_wavelength_array[i]
    old_fluxes_arr_0 = A_all_fluxes_0_array[i]
    old_fluxes_arr_1 = A_all_fluxes_1_array[i]
    old_fluxes_arr_2 = A_all_fluxes_2_array[i]
    old_fluxes_arr_3 = A_all_fluxes_3_array[i]
    old_fluxes_arr_4 = A_all_fluxes_4_array[i]
    old_fluxes_arr_5 = A_all_fluxes_5_array[i]
    old_fluxes_arr_6 = A_all_fluxes_6_array[i]

    elements_to_delete = []
    for j in range(0,len(old_wavelengths_arr)):
        if old_wavelengths_arr[j] in wavelengths_correct:
            pass
        else:
            elements_to_delete.append(j)
    
    new_wavelengths_arr = np.delete(old_wavelengths_arr,elements_to_delete)
    new_fluxes_arr_0 = np.delete(old_fluxes_arr_0,elements_to_delete)
    new_fluxes_arr_1 = np.delete(old_fluxes_arr_1,elements_to_delete)
    new_fluxes_arr_2 = np.delete(old_fluxes_arr_2,elements_to_delete)
    new_fluxes_arr_3 = np.delete(old_fluxes_arr_3,elements_to_delete)
    new_fluxes_arr_4 = np.delete(old_fluxes_arr_4,elements_to_delete)
    new_fluxes_arr_5 = np.delete(old_fluxes_arr_5,elements_to_delete)
    new_fluxes_arr_6 = np.delete(old_fluxes_arr_6,elements_to_delete)

    
    A_all_wavelength_array_const_wavelengths.append(new_wavelengths_arr)
    
    A_all_fluxes_array_0_const_wavelengths.append(new_fluxes_arr_0)
    A_all_fluxes_array_1_const_wavelengths.append(new_fluxes_arr_1)
    A_all_fluxes_array_2_const_wavelengths.append(new_fluxes_arr_2)
    A_all_fluxes_array_3_const_wavelengths.append(new_fluxes_arr_3)
    A_all_fluxes_array_4_const_wavelengths.append(new_fluxes_arr_4)
    A_all_fluxes_array_5_const_wavelengths.append(new_fluxes_arr_5)
    A_all_fluxes_array_6_const_wavelengths.append(new_fluxes_arr_6)

print ("shapes of new wavelength vs flux arrays - correct")
print (np.shape(A_all_wavelength_array_const_wavelengths))
print (np.shape(A_all_fluxes_array_0_const_wavelengths))


# now need to duplicate the first arrays, and add all the different orientations in
# this is doubling each time! need to just add one array each time

final_z_array = []
final_sfr_array = []
final_sfr_10_array = []
final_sfr_100_array = []
final_dust_mass_array = []
final_gas_mass_array = []
final_stellar_mass_array = []

for i in range(0,7):
    
    final_z_array = np.append(final_z_array,A_all_z_array)
    final_sfr_array = np.append(final_sfr_array,A_all_sfr_array)
    final_sfr_10_array = np.append(final_sfr_10_array,A_all_sfr_10_array)
    final_sfr_100_array = np.append(final_sfr_100_array,A_all_sfr_100_array)
    final_dust_mass_array = np.append(final_dust_mass_array,A_all_dust_mass_array)
    final_gas_mass_array = np.append(final_gas_mass_array,A_all_gas_mass_array)
    final_stellar_mass_array = np.append(final_stellar_mass_array,A_all_stellar_mass_array)
    
    if i==0:
        final_wavelengths_array = A_all_wavelength_array_const_wavelengths
        A_all_orientations = np.ones(len(A_all_z_array))*i
    else:
        A_all_orientations = np.append(A_all_orientations,np.ones(len(A_all_z_array))*i)
        final_wavelengths_array = np.append(final_wavelengths_array,A_all_wavelength_array_const_wavelengths,axis=0)

print ("shape of arrays*7", np.shape(final_z_array))
print ("orientations",np.shape(A_all_orientations))

A_all_fluxes_array_const_wavelengths = np.append(A_all_fluxes_array_0_const_wavelengths,A_all_fluxes_array_1_const_wavelengths,axis=0)
A_all_fluxes_array_const_wavelengths = np.append(A_all_fluxes_array_const_wavelengths,A_all_fluxes_array_2_const_wavelengths,axis=0)
A_all_fluxes_array_const_wavelengths = np.append(A_all_fluxes_array_const_wavelengths,A_all_fluxes_array_3_const_wavelengths,axis=0)
A_all_fluxes_array_const_wavelengths = np.append(A_all_fluxes_array_const_wavelengths,A_all_fluxes_array_4_const_wavelengths,axis=0)
A_all_fluxes_array_const_wavelengths = np.append(A_all_fluxes_array_const_wavelengths,A_all_fluxes_array_5_const_wavelengths,axis=0)
A_all_fluxes_array_const_wavelengths = np.append(A_all_fluxes_array_const_wavelengths,A_all_fluxes_array_6_const_wavelengths,axis=0)


# put these into a .h5 file
hf = h5py.File('my_fire2_data_spectrum_subset_orientations.h5', 'w')
hf.create_dataset('A_all_z', data=final_z_array)
hf.create_dataset('A_all_sfr', data=final_sfr_array)
hf.create_dataset('A_all_sfr_10', data=final_sfr_10_array)
hf.create_dataset('A_all_sfr_100', data=final_sfr_100_array)
hf.create_dataset('A_all_dust_mass', data=final_dust_mass_array)
hf.create_dataset('A_all_gas_mass', data=final_gas_mass_array)
hf.create_dataset('A_all_stellar_mass', data=final_stellar_mass_array)
hf.create_dataset('A_all_wavelength', data=final_wavelengths_array)
hf.create_dataset('A_all_fluxes', data=A_all_fluxes_array_const_wavelengths)
hf.create_dataset('A_all_orientations', data=A_all_orientations)

hf.close()

my_fire2_data = "my_fire2_data_spectrum_subset_orientations.h5"
f = h5py.File(my_fire2_data,'r')
A_all_z = f['A_all_z']
A_all_sfr = f['A_all_sfr']
A_all_sfr_10 = f['A_all_sfr_10']
A_all_wavelength = f['A_all_wavelength']
A_all_fluxes = f['A_all_fluxes']
A_all_orientations = f['A_all_orientations']

print (np.shape(A_all_z))
print (np.shape(A_all_sfr))
print (np.shape(A_all_sfr_10))
print (np.shape(A_all_wavelength))
print (np.shape(A_all_fluxes))
print (np.shape(A_all_orientations))

