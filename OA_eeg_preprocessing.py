#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  4 20:32:07 2024

@author: nicolasdecat

Script in 3 parts
1) Crop eeg segments (-30 to 0s before each trial), preproc (reref, filtering), save (sub_ref_filt.fif)
2) From sub_ref_filt -> manual flagging of BC and BS (csv saved)
3) from sub_ref_filt -> BC interpolation, save BS-free segs (sub_preproc.fif)
"""

import mne
import pandas as pd
import numpy as np
import os, glob
import matplotlib.pyplot as plt
from pathlib import Path
# %matplotlib inline


################################################################################################################################################
#%% CROPPING -30 0s + PREPROC + SAVE (Prep for BC visualisation)

# Define paths
path_table = '/Users/nicolas.decat/Library/Mobile Documents/com~apple~CloudDocs/Documents/PhD/projects/NapNest/subjective_dimensions/gigatable.csv'
path_data = '/Volumes/disk-nico/napnest/data/' 

# Load data
table = pd.read_csv(path_table)

# Parameters
len_seg_sec = 30
behavior = {"eyes-open", "trial-too-short","intentional","moving"}

# Get subs
subs = [f for f in os.listdir(path_data) if len(f) == 5]
# subs = subs[37]

#%% loop through sujectsi

for i, sub in enumerate(subs):
    
    print(f'Working on sub {sub} ({i+1}/{len(subs)})...')

    # Import .eeg
    path_vhdr = path_data + f'{sub}/{sub}.vhdr'
    raw = mne.io.read_raw_brainvision(vhdr_fname=path_vhdr)
    
    # Get metadata
    sr = raw.info['sfreq']
    physio_chan = ['IO','EMG chin','EMG arm','ECG']
    # physio_chan = ['EMG arm','ECG']

    chan_name = [name for name in raw.ch_names if name not in physio_chan] # Get EEG only
    
    # Get the onset and offset of the naps (drop if nan)
    table_sub = table[table["Subject"] == sub].dropna(subset=["Trial start", "Trial end","Probe"])
    
    # Drop if onset/offset is NaN, or if behavioral issues
    table_sub = table_sub.dropna(subset=["Trial start", "Trial end","Probe"])
    table_sub = table_sub[~table_sub["Behavior"].isin(behavior)]
    
    # If there are trials, create a folder to store report / .fif
    if not table_sub.empty:
        save_path = '/Volumes/disk-nico/napnest/data_ref_filt_30s/'
        report_path = save_path + f'{sub}/Report_ref_filt' 
        path_to_create=[report_path]
        for p in path_to_create:
            path = Path(p)
            path.mkdir(parents=True, exist_ok=True)
    else:
        continue

    # Loop through trials
    for i in range(len(table_sub)):
        
        # Get trial info
        trial = table_sub.iloc[[i]]
        probe = trial["Probe"].iloc[0]
        group = trial["Group"].iloc[0]
        trialend = trial["Trial end"].iloc[0]
        trial_label = sub + '_' + probe + '_' + group
        
        # Crop -30s locked to probe (with a +/-10s to prevent filtering issues on the epoch borders)
        margin = 10
        seg_offset = int(trialend) + margin
        seg_onset = int(trialend) - len_seg_sec - margin
        trial_raw = raw.copy().crop(tmin=seg_onset, tmax=seg_offset).pick(chan_name)
        
#%% Preprocessing
        
        # Add raw trial and psd to report 
        report = mne.Report(title='PREPROCESSING_' + trial_label)
        report.add_raw(trial_raw,title='Raw data '+ trial_label, psd=False)  
        fig=trial_raw.compute_psd(method='welch', fmin=0, fmax=100).plot(show=False)
        report.add_figure(fig,title='PSD raw '+ trial_label)
        
        # Montage 
        montage = mne.channels.make_standard_montage('standard_1020')
        trial_raw.set_montage(montage)
        trial_raw.info['description'] = 'standard_1020'
        
        # Average referencing
        trial_filt = trial_raw.copy().load_data()
        mne.set_eeg_reference(trial_filt, ref_channels='average', copy = False)
        
        # Filtering
        trial_filt.notch_filter(np.arange(50,125,50),method='fft',filter_length='auto',phase='zero',n_jobs=-1) #NICE way
 
        hpass=0.1
        iir_params=dict(ftype='butter', order=6)
        trial_filt.filter(l_freq=hpass,h_freq=None, n_jobs=-1, method='iir', iir_params=iir_params)
        
        lpass=35
        iir_params=dict(ftype='butter', order=8)
        trial_filt.filter(l_freq=None,h_freq=lpass, n_jobs=-1, method='iir', iir_params=iir_params)
        
        # Crop back to 30s
        sfreq = trial_filt.info['sfreq']
        samples_to_remove = int(margin * sfreq)
        data = trial_filt.get_data()
        data_sliced = data[:, samples_to_remove:-samples_to_remove]
        trial_filt = mne.io.RawArray(data_sliced, trial_filt.info)    
        
        # Add filtered trial to report
        report.add_raw(trial_filt,title='Reref avg + notch + 0.1-35Hz filtering '+ trial_label, psd=False)  
        fig=trial_filt.compute_psd(method='welch', fmin=0, fmax=100).plot(show=False)
        report.add_figure(fig,title='Welch after Reref avg + notch + 0.1-35Hz filtering '+ trial_label)
        
        # Save filtered trial
        trial_filt.save(save_path + f'{sub}/{trial_label}_ref_filt_2.fif', overwrite=True)
        # trial_filt.save(f'/Volumes/disk-nico/napnest/NN_fig/snapshot_C4_N2/{trial_label}_ref_filt_2.fif', overwrite=True)
        
        # Save report
        report.save(report_path +'/'+ trial_label + '_ref_filt_report.html', overwrite=True, open_browser=False)


 
################################################################################################################################################
#%% MANUAL FLAGGING OF BC (BS is reported manually on a csv)

import mne
import pandas as pd
import numpy as np
import os, glob
import matplotlib.pyplot as plt
import csv
from pathlib import Path
# %matplotlib auto

# Define paths and parameters
path_data = '/Volumes/disk-nico/napnest/data_ref_filt_30s/' 
path_save = '/Volumes/disk-nico/napnest/' 
all_badchans = []

# Get subs
base_path = Path(path_data)
files = list(base_path.rglob('*ref_filt.fif'))
files = [file.name for file in files]


#%% loop through sujects

for i, file in enumerate(files):
    
    sub = file[:5]
    # Import .fif
    path_fif = path_data + f'{sub}/{file}'
    trial_filt = mne.io.read_raw_fif(path_fif, preload=True)
    
    # Plot the data with block=True (will close only after you mark BCs and close the figure)
    fig = trial_filt.plot(duration=30,block=True)
    plt.show()
    
    # # Store bad channel info (marked manually)
    badchans = trial_filt.info['bads']
    print(f"bad channels: {badchans}")
    all_badchans.append((file[:11],[str(channel) for channel in badchans]))
            
    # Save the output table
    with open(path_save + 'all_badchans.csv', 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['trial', 'badchan'])
        for row in all_badchans:
            csv_writer.writerow(row)


################################################################################################################################################


#%% PREPROC with BC INTERPOLATION 

import mne
import pandas as pd
import numpy as np
import os, glob
import matplotlib.pyplot as plt
from pathlib import Path
# %matplotlib inline

# Define paths
path_table = '/Users/nicolas.decat/Library/Mobile Documents/com~apple~CloudDocs/Documents/PhD/projects/NapNest/subjective_dimensions/gigatable.csv'
path_data = '/Volumes/disk-nico/napnest/data/' 
path_badchans = '/Volumes/disk-nico/napnest/secured_BCBS/all_badchans.csv' 

# Load data
table = pd.read_csv(path_table)
table_badchans = pd.read_csv(path_badchans)

# Parameters
len_seg_sec = 30
behavior = {"eyes-open","eyes-blinks","trial-too-short","intentional","moving"}

# Get subs
subs = [f for f in os.listdir(path_data) if len(f) == 5]

#%% loop through sujects

for i, sub in enumerate(subs):
    
    print(f'Working on sub {sub} ({i+1}/{len(subs)})...')
    
    # Import .eeg
    path_vhdr = path_data + f'{sub}/{sub}.vhdr'
    raw = mne.io.read_raw_brainvision(vhdr_fname=path_vhdr)
    
    # Get metadata
    sr = raw.info['sfreq']
    physio_chan = ['IO','EMG chin','EMG arm','ECG']
    chan_name = [name for name in raw.ch_names if name not in physio_chan] # Get EEG only
    
    # Get the onset and offset of the naps (drop if nan)
    table_sub = table[table["Subject"] == sub].dropna(subset=["Trial start", "Trial end","Probe"])
    
    # Drop if onset/offset is NaN, or if behavioral issues
    table_sub = table_sub.dropna(subset=["Trial start", "Trial end","Probe"])
    table_sub = table_sub[~table_sub["Behavior"].isin(behavior)]
    
    # If there are trials, create a folder to store report / .fif
    if not table_sub.empty:
        save_path = '/Volumes/disk-nico/napnest/data_ref_filt_BCinterp_30s/'
        report_path = save_path + f'{sub}/Report_ref_filt_BCinterp' 
        path_to_create=[report_path]
        for p in path_to_create:
            path = Path(p)
            path.mkdir(parents=True, exist_ok=True)
    else:
        continue

    # Loop through trials
    for i in range(len(table_sub)):
        
        # Get trial info
        trial = table_sub.iloc[[i]]
        probe = trial["Probe"].iloc[0]
        group = trial["Group"].iloc[0]
        trialend = trial["Trial end"].iloc[0]
        trial_label = sub + '_' + probe + '_' + group
        
        # Crop -30s locked to probe (with a +/-10s to prevent filtering issues on the epoch borders)
        margin = 10
        seg_offset = int(trialend) + margin
        seg_onset = int(trialend) - len_seg_sec - margin
        trial_raw = raw.copy().crop(tmin=seg_onset, tmax=seg_offset).pick(chan_name)
        
        #%% Preprocessing
        
        # Add raw trial and psd to report 
        report = mne.Report(title='PREPROCESSING_' + trial_label)
        report.add_raw(trial_raw,title='Raw data '+ trial_label, psd=False)  
        fig=trial_raw.compute_psd(method='welch', fmin=0, fmax=100).plot(show=False)
        report.add_figure(fig,title='PSD raw '+ trial_label)
        
        # Montage 
        montage = mne.channels.make_standard_montage('standard_1020')
        trial_raw.set_montage(montage)
        trial_raw.info['description'] = 'standard_1020'
        
        # Average referencing
        trial_filt = trial_raw.copy().load_data()
        mne.set_eeg_reference(trial_filt, ref_channels='average', copy = False)
        
        # BC interpolation
        badchans = eval(table_badchans[table_badchans['trial'] == f"{trial_label}"[:11]]["badchan"].iloc[0])
        trial_filt.info['bads'].extend(badchans)
        trial_preproc = trial_filt.copy()
        trial_preproc.interpolate_bads(reset_bads=True)  # Interpolate bad channels 
        
        # Add badchans on Report
        report.add_html(badchans, title=f'{len(badchans)} bad channels interpolated')
        
        # Add trial and psd after reref and BC interp 
        report.add_raw(trial_preproc,title='Reref avg + BC interp  '+ trial_label, psd=False)  
        fig=trial_preproc.compute_psd(method='welch', fmin=0, fmax=100).plot(show=False)
        report.add_figure(fig,title='Welch after Reref avg + BC interp '+ trial_label)
        
        # Filtering
        trial_preproc.notch_filter(np.arange(50,125,50),method='fft',filter_length='auto',phase='zero',n_jobs=-1) #NICE way
 
        hpass=0.5
        iir_params=dict(ftype='butter', order=6)
        trial_preproc.filter(l_freq=hpass,h_freq=None, n_jobs=-1, method='iir', iir_params=iir_params)
        
        lpass=35
        iir_params=dict(ftype='butter', order=8)
        trial_preproc.filter(l_freq=None,h_freq=lpass, n_jobs=-1, method='iir', iir_params=iir_params)
        
        # Crop back to 30s
        sfreq = trial_preproc.info['sfreq']
        samples_to_remove = int(margin * sfreq)
        data = trial_preproc.get_data()
        data_sliced = data[:, samples_to_remove:-samples_to_remove]
        trial_preproc = mne.io.RawArray(data_sliced, trial_preproc.info)    
        
        # Add filtered trial to report
        report.add_raw(trial_preproc,title=f'Reref avg + BC interp + notch + {hpass}-{lpass}Hz filtering '+ trial_label, psd=False)  
        fig=trial_preproc.compute_psd(method='welch', fmin=0, fmax=100).plot(show=False)
        report.add_figure(fig,title=f'Welch after Reref avg + BC interp + notch + {hpass}-{lpass}Hz filtering '+ trial_label)
        
        # Save filtered trial
        trial_preproc.save(save_path + f'{sub}/{trial_label}_ref_filt_BCinterp.fif', overwrite=True)
        
        # Save report
        report.save(report_path +'/'+ trial_label + '_ref_filt_BCinterp_report.html', overwrite=True, open_browser=False)





