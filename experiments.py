from sifce import datatools
import numpy as np
import pandas as pd
import bilby
import pylab
import matplotlib.pyplot as plt
from bilby.core.prior import Uniform, Sine, Cosine, PowerLaw, Constraint
from pycbc.detector import Detector
from pycbc.psd import analytical 
from bilby.gw.conversion import bilby_to_lalsimulation_spins

def get_mass2(prior, n_samples):
    '''
    This function calculates mass_2 based on mass_ratio and mass_1
    '''
    mass_1=prior['mass_1'].sample(n_samples)
    mass_2=mass_1 * prior['mass_ratio'].sample(n_samples)
    return mass_1, mass_2

def compute_strain_df(row):
    return datatools.compute_strain_fd({'mass_1':row['mass_1'],'mass_2': row['mass_2']}, 'IMRPhenomXPHM')

def main():
    ifos=['H1', 'L1', 'V1']
    ifos_det = []

    for ifo in ifos:
        ifos_det.append(Detector(ifo))
    
    sample_rate=1024
    flow = 10.0
    delta_f = 1.0/16
    flen = int(sample_rate/delta_f)
    psd = analytical.aLIGOZeroDetHighPower(flen, delta_f, flow) 
    psd_dict = dict(H1= psd, L1= psd, V1=psd)

    prior_sky=dict(dec=Cosine(name='dec'),
               ra=Uniform(name='ra', minimum=0, maximum=2 * np.pi, boundary='periodic'),
               psi =  Uniform(name='psi', minimum=0, maximum=np.pi, boundary='periodic'))

    # create samples
    prior_waves=dict(mass_ratio=PowerLaw(alpha=2, name='mass_ratio', minimum=0.125, maximum=1),
                    mass_1= PowerLaw(alpha=-1, name='mass_1', minimum=10, maximum=80),
                    a_1 = Uniform(name='a_1', minimum=0, maximum=0.99),
                    a_2 = Uniform(name='a_2', minimum=0, maximum=0.99),
                    tilt_1 = Sine(name='tilt_1'),
                    tilt_2 = Sine(name='tilt_2'),
                    phi_12 = Uniform(name='phi_12', minimum=0, maximum=2 * np.pi, boundary='periodic'),
                    phi_jl = Uniform(name='phi_jl', minimum=0, maximum=2 * np.pi, boundary='periodic'),
                    luminosity_distance = PowerLaw(alpha=2, name='luminosity_distance', minimum=50, maximum=2000, unit='Mpc', latex_label='$d_L$'),
                    dec =  Cosine(name='dec'),
                    ra =  Uniform(name='ra', minimum=0, maximum=2 * np.pi, boundary='periodic'),
                    theta_jn =  Sine(name='theta_jn'),
                    psi =  Uniform(name='psi', minimum=0, maximum=np.pi, boundary='periodic'),
                    phase =  Uniform(name='phase', minimum=0, maximum=2 * np.pi, boundary='periodic'))

    n_samples = 10
    m1_list, m2_list = get_mass2(prior_waves, n_samples)
    theta_jn_list = prior_waves["theta_jn"].sample(n_samples)
    phase_list = prior_waves["phase"].sample(n_samples)
    a_1_list = prior_waves["a_1"].sample(n_samples)
    a_2_list = prior_waves["a_2"].sample(n_samples)
    tilt_1_list = prior_waves["tilt_1"].sample(n_samples)
    tilt_2_list = prior_waves["tilt_2"].sample(n_samples)
    phi_12_list = prior_waves["phi_12"].sample(n_samples)
    phi_jl_list = prior_waves["phi_jl"].sample(n_samples)
    iota_list = list()
    spin_1x_list = list()
    spin_1y_list = list()
    spin_1z_list = list()
    spin_2x_list = list()
    spin_2y_list = list()
    spin_2z_list = list()

    for i in range(n_samples):
        iota, spin_1x, spin_1y, spin_1z, spin_2x, spin_2y, spin_2z = bilby_to_lalsimulation_spins(theta_jn = theta_jn_list[i],
                                                                                                  a_1 = a_1_list[i],
                                                                                                  a_2 = a_2_list[i],
                                                                                                  tilt_1 = tilt_1_list[i],
                                                                                                  tilt_2 = tilt_2_list[i],
                                                                                                  phi_12 = phi_12_list[i],
                                                                                                  phi_jl = phi_jl_list[i],
                                                                                                  mass_1 = m1_list[i],
                                                                                                  mass_2 = m2_list[i], 
                                                                                                  reference_frequency = 20,
                                                                                                  phase = phase_list[i])
        iota_list.append(iota)
        spin_1x_list.append(spin_1x)
        spin_1y_list.append(spin_1y)
        spin_1z_list.append(spin_1z)
        spin_2x_list.append(spin_2x)
        spin_2y_list.append(spin_2y)
        spin_2z_list.append(spin_2z)

    # create waveforms
    hp_list = list()
    hc_list = list()
    # return two lists, hc and hp --> what is the best way to store it? 

    for m1, m2, s1x, s1y, s1z, s2x, s2y, s2z in zip(m1_list, m2_list, spin_1x_list, spin_1y_list, spin_1z_list, spin_2x_list, spin_2y_list, spin_2z_list):
        hp, hc = datatools.compute_strain_fd({'mass_1':m1,
                                              'mass_2': m2, 
                                              'spin_1x': s1x, 
                                              'spin_1y': s1y,
                                              'spin_1z': s1z, 
                                              'spin_2x': s2x, 
                                              'spin_2y': s2y,
                                              'spin_2z': s2z}, 
                                              'IMRPhenomXPHM', 
                                              delta_f=delta_f)
        hp_list.append(hp)
        hc_list.append(hc)

    # hp, hc = datatools.compute_strain_fd()

    #create sky loc
    ra_lst = prior_sky['ra'].sample(100)
    dec_lst = prior_sky['dec'].sample(100)
    psi_lst = prior_sky['psi'].sample(100)

    # how to store? list of lists? 
    # need two loops: can go ahead and compute snr so dont have to store list of list of lists
    # set up psd!! do i need different ones for different ifos? 
    snr_list_of_lists = list()
    for hp, hc in zip(hp_list, hc_list):
        snr_list = list()
        for ra, dec, psi in zip(ra_lst, dec_lst, psi_lst):
            strains = dict()
            for ifo, det in zip(ifos, ifos_det):
                strain = datatools.project_and_combine({'ra': ra, 'dec':dec, 'psi': psi}, 
                                                       {'plus': hp, 'cross': hc}, 
                                                       end_time=10, detector_obj=det)
                strains[ifo] = (strain,)     
            snr = datatools.compute_snr_fd(strains, psd_dict)
            snr_list.append(snr[0]['net'])
        snr_list_of_lists.append(snr_list)

    #snr_list_of_lists is a list of snr_lists
    #snr_list is composed of dictionaries, each with four keys: 'H1', 'L1', 'V1' and 'net' 

    # plt.scatter(np.arange(0, 100, 1), snr_list_of_lists[0])
    # plt.show()

    for i, ll in enumerate(snr_list_of_lists):
        print(f'''---------------------------------------------- 
        SNR waveform {i}:
            mass 1 = {m1_list[i]}, mass 2 = {m2_list[i]}, 
            spins 1 =({spin_1x_list[i], spin_1y_list[i], spin_1z_list[i]}), spins 2 = ({spin_2x_list[i], spin_2y_list[i], spin_2z_list[i]})
        5 iterations: 
                mean:   {np.mean(np.array(ll[0:5]))}
                max:    {np.max(np.array(ll[0:5]))}
                st dev: {np.std(np.array(ll[0:5]))}
        10 iterations: 
                mean:   {np.mean(np.array(ll[0:10]))}
                max:    {np.max(np.array(ll[0:10]))}
                st dev: {np.std(np.array(ll[0:10]))}
        100 iterations: 
                mean:   {np.mean(np.array(ll[0:100]))}
                max:    {np.max(np.array(ll[0:100]))}
                st dev: {np.std(np.array(ll[0:100]))}
        ''')
    
if __name__ == "__main__":
    main()