#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 15 10:25:44 2024

NapNest: Extract power and complexity from preproc segs (output from OA_eeg_preprocessing.py)

For codes involving complexity and connectivity computation, please refer to the codes available on GitHub, NICE package

@author: nico
"""

#%% POWER EXTRACTION  

import mne
import pandas as pd
import numpy as np
import os, glob
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict
from statsmodels.nonparametric.smoothers_lowess import lowess
from fooof import FOOOF
from fooof.sim.gen import gen_aperiodic
from fooof.bands import Bands
from complexity import compute_complexity
import nice
from mne_connectivity import spectral_connectivity_epochs

# %matplotlib inline

# Define paths
path_data = '/Volumes/disk-nico/napnest/data_ref_filt_BCinterp_30s/'
path_table = '/Users/nicolas.decat/Library/Mobile Documents/com~apple~CloudDocs/Documents/PhD/projects/NapNest/subjective_dimensions/gigatable.csv'
path_badsegs = '/Volumes/disk-nico/napnest/secured_BCBS/all_badsegs.csv' 

# Parameters
table = pd.read_csv(path_table) 
badsegs = pd.read_csv(path_badsegs, usecols=[0, 1, 2, 3]) 

# Get files
files = [f for f in os.listdir(path_data) if len(f) == 5]
base_path = Path(path_data)
files = list(base_path.rglob('*ref_filt_BCinterp.fif'))
files = [file.name for file in files]

# Output bigdic
columnNames = [
    "subject","channel", 
    'abs_delta','abs_theta','abs_alpha','abs_beta','abs_gamma',
    'rel_delta','rel_theta','rel_alpha','rel_beta','rel_gamma',
    'psd_data',
    'abs_delta_ff','abs_theta_ff','abs_alpha_ff','abs_beta_ff','abs_gamma_ff',
    'rel_delta_ff','rel_theta_ff','rel_alpha_ff','rel_beta_ff','rel_gamma_ff',
    'psd_data_ff', 'slope', 'offset',
    'Kolmogorov','Permutation_Entropy_theta','Sample_Entropy',
    'WSMI_delta','WSMI_theta','WSMI_alpha','WSMI_beta'    
    ]
cols_power = [
    'abs_delta','abs_theta','abs_alpha','abs_beta','abs_gamma',
    'rel_delta','rel_theta','rel_alpha','rel_beta','rel_gamma',
    'abs_delta_ff','abs_theta_ff','abs_alpha_ff','abs_beta_ff','abs_gamma_ff',
    'rel_delta_ff','rel_theta_ff','rel_alpha_ff','rel_beta_ff','rel_gamma_ff'
    ] 
cols_entropy = ['Kolmogorov','Permutation_Entropy_theta','Sample_Entropy']
cols_connectivity = ['WSMI_delta','WSMI_theta','WSMI_alpha','WSMI_beta' ]

bigdic = {f : [] for f in columnNames}

# Parameters
physio_chan = ['IO','EMG chin','EMG arm','ECG']
segments = {'10-0': (20, 30), '20-10': (10, 20), '30-20': (0, 10)} 
which_seg = '10-0'

# Define the right segment
which_seg = '10-0'

# Loop through files
for file in files:
        
    subjectname = file[0:11]
    if subjectname[-1:] == 'X':
        subjectname += 'T'
    elif subjectname[-1:] == 'O':
        subjectname += 'N'
    
    sub = subjectname[0:5]
    print(subjectname)
    path_file = path_data + sub + '/' + subjectname + '_ref_filt_BCinterp.fif'
    
    # Skip this file if the targeted window is a bad segment
    row_badseg = badsegs[badsegs['trial'].str.contains(subjectname, na=False)]
    value_badseg = which_seg[:2]
    
    # if row_badseg[value_badseg].iloc[0] == 1:
    #     continue
    
# Import file info ##########################################################

    trial_preproc = mne.io.read_raw_fif(path_file, preload=True)
    
    # Get metadata
    sr = trial_preproc.info['sfreq']
    chan = [name for name in trial_preproc.ch_names 
         if name not in physio_chan]
    channel_name = chan
    num_channels = len(channel_name)
    channel_type = ['eeg']*num_channels
    
# Crop to the right window ##################################################

    tmin, tmax = segments[which_seg]
    trial_preproc_win = trial_preproc.copy().crop(tmin=tmin, tmax=tmax)

# Extract power #############################################################

    # Define frequency bands
    freqs = np.linspace(0.5, 40, 80)
    freq_bands = {
          'delta': (.5, 4),
          'theta': (4, 8),
          'alpha': (8, 12),
          'beta': (12, 30),
          'gamma': (30, 40)} 
    
    # Compute the PSD (on 5s windows sliding by 0.5s)
    temp_list = []
    temp_power = trial_preproc_win.compute_psd(
                method = "welch",
                fmin = .5, 
                fmax = 40,
                n_fft = 1000,
                n_overlap = 250,
                n_per_seg = 500,
                window = "hamming",
                n_jobs = 4)
      
    # Extract the PSD for each band
    temp_power = temp_power.get_data()
    
# FOOOF: store periodic signal, slope and offset
    fooofed_data_chan = []
    slope = []
    offset = []
    for i_ch, chan in enumerate(channel_name):
        
        temp_power_chan = temp_power[i_ch]
  
        fm = FOOOF(peak_width_limits = [1, 4], aperiodic_mode="fixed")
        fm.add_data(freqs, temp_power_chan)
        fm.fit()
        
        init_ap_fit = gen_aperiodic(
            fm.freqs, 
            fm._robust_ap_fit(fm.freqs, fm.power_spectrum)
            )
        
        init_flat_spec = fm.power_spectrum - init_ap_fit 
        fooofed_data_chan.append(init_flat_spec)
        slope.append(fm.get_results().aperiodic_params[1])
        offset.append(fm.get_results().aperiodic_params[0])

    temp_power_ff = np.vstack(fooofed_data_chan)
    
    # Run aperiodic and periodic power
    for data_idx, power_data in [("", temp_power), ("_ff", temp_power_ff)]:

        # Loop through epochs
        abs_bandpower_ch_all_epoch = []
        rel_bandpower_ch_all_epoch = []
        
        # Store accumulated band power values across epochs for each band and electrode
        accumulated_abs_bandpower = defaultdict(list)
        accumulated_rel_bandpower = defaultdict(lambda: defaultdict(list))
    
        # Calculate absolute band power for each channel
        abs_bandpower_ch = {
            f"abs_{band}{data_idx}": np.nanmean(
                power_data[:, np.logical_and(freqs >= borders[0], freqs <= borders[1])],
                axis=1
            )
            for band, borders in freq_bands.items()
        }
        
        # Compute the total power for each channel across all bands
        total_abs_power_ch = {}
        for channel_idx in range(power_data.shape[0]):  # Loop through each channel
            total_abs_power_ch[channel_idx] = np.sum([
                np.nanmean(power_data[channel_idx, np.logical_and(freqs >= borders[0], freqs <= borders[1])])
                for borders in freq_bands.values()
            ])
        
        # Calculate relative band power for each channel
        rel_bandpower_ch = {
            f"rel_{band}{data_idx}": {
                channel_idx: abs_bandpower_ch[f"abs_{band}{data_idx}"][channel_idx] / total_abs_power_ch[channel_idx]
                for channel_idx in range(power_data.shape[0])
            }
            for band in freq_bands.keys()
        }
        
        # Gather absolute and relative power values for each band across channels
        for band, values in abs_bandpower_ch.items():
            accumulated_abs_bandpower[band].append(values)
        
        for band, channel_data in rel_bandpower_ch.items():
            for channel_idx, value in channel_data.items():
                accumulated_rel_bandpower[band][channel_idx].append(value)
        
        # Calculate the mean absolute power across the single segment for each band and channel
        average_abs_bandpower = {
            band: np.mean(values, axis=0)  # Averaging over channels
            for band, values in accumulated_abs_bandpower.items()
        }
        
        # Calculate the mean relative power across the single segment for each band and channel
        average_rel_bandpower = defaultdict(dict)
        for band, channel_data in accumulated_rel_bandpower.items():
            for channel_idx, values in channel_data.items():
                average_rel_bandpower[band][channel_idx] = np.mean(values)
          
# Extract complexity ########################################################

        # Extract the time series and reshape in 3D (add the epoch component)
        signal = trial_preproc_win.get_data()
        signal = np.expand_dims(signal, axis=0)
        
        # Create info object and Epoch array
        info = mne.create_info(ch_names=channel_name, sfreq=sr, ch_types=channel_type)
        epochs = mne.EpochsArray(signal, info, event_id=None)
        
        # Compute complexity (Esteban's script)
        runfile('/Users/nicolas.decat/Library/Mobile Documents/com~apple~CloudDocs/Documents/GitHub/mycodes/permutation_entropy.py', wdir='/Users/nicolas.decat/Library/Mobile Documents/com~apple~CloudDocs/Documents/GitHub/mycodes')
        complexity = compute_complexity(epochs,reduction=None,compute_gamma=False)
        

# Extract connectivity ########################################################
        
        runfile('/Users/nicolas.decat/Library/Mobile Documents/com~apple~CloudDocs/Documents/GitHub/nice/nice/algorithms/connectivity.py', wdir='/Users/nicolas.decat/Library/Mobile Documents/com~apple~CloudDocs/Documents/GitHub/nice/nice/algorithms')
        from connectivity import epochs_compute_wsmi

        # Compute wSMI
        print('Computing wSMI')
        markers= {}

        montage = mne.channels.make_standard_montage('standard_1020')  # Example for 10-20 system
        epochs.set_montage(montage)
        frequencies= [(0.5,4,32,'delta'),
                      (4,8,16,'theta'),
                      (8,12,8,'alpha'),
                      (12,30,4,'beta')]
        for fmin,fmax,tau, name in frequencies:
            wsmi,_,_,_= epochs_compute_wsmi(epochs,kernel=3,tau=tau,
                                        n_jobs=-1,backend='python',method_params= {'bypass_csd':False})
            wsmi= wsmi.transpose(2,0,1)
            print(wsmi.shape)
            markers['WSMI_{}'.format(name)]= wsmi
                 
    # Add to the bigdic ###########################################################
                
        if data_idx == "": which_cols_power = cols_power[:10]
        if data_idx == "_ff": which_cols_power = cols_power[10:]
        
        for i_ch, channel in enumerate(channel_name) :
            if data_idx == "": # only do it once
                bigdic['subject'].append(subjectname)
                bigdic['channel'].append(channel)
                for col in cols_entropy :
                    bigdic[col].append(np.squeeze(complexity[col])[i_ch])
                print('')
            elif data_idx == "_ff":
                bigdic['slope'].append(slope[:][i_ch])
                bigdic['offset'].append(offset[:][i_ch])
                for col in cols_connectivity:
                    if channel == 'Fp1': # only on first chan
                        marker = markers[col]
                        bigdic[col].append(np.squeeze(markers[col]))
                    else: bigdic[col].append(0)
            bigdic[f'psd_data{data_idx}'].append(power_data[:][i_ch])
            for col in which_cols_power :
                if col.startswith('abs'):
                    bigdic[col].append(abs_bandpower_ch[col][i_ch])
                if col.startswith('rel'):
                    bigdic[col].append(rel_bandpower_ch[col][i_ch])


# Save the bigdic
df = pd.DataFrame.from_dict(bigdic)


