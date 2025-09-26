#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 17 16:49:35 2024

NapNest: plot wSMI

@author: nico
"""

import pandas as pd
import numpy as np
import os, mne
import matplotlib.pyplot as plt
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
# Define paths 

path_table = '/Users/nicolas.decat/Library/Mobile Documents/com~apple~CloudDocs/Documents/PhD/projects/NapNest/subjective_dimensions/gigatable_4c.csv'
path_eeg = "/Users/nicolas.decat/Downloads/eeg_features_10-0.csv"


# Import table + eeg, merge them
table = pd.read_csv(path_table, index_col=0, sep=None)
eeg_init = pd.read_csv(path_eeg, index_col=0)
eeg_init = eeg_init.drop(columns=['psd_data', 'psd_data_ff'])

# Merge table and eeg
eeg = eeg_init
eeg[['Subject', 'Probe', 'Group']] = eeg['subject'].str.split('_', expand=True)
eeg = eeg.merge(table, 
                left_on=['Subject', 'Probe', 'Group'], 
                right_on=['Subject', 'Probe', 'Group'], 
                how='left')

# Only keep Stage-at-probe = Wake
# eeg = eeg[eeg['Stage-at-probe'].isin(['W'])]  # ['1','MSE']

# Add eeg info to table_wsmi cols
table_wsmi = pd.read_pickle("/Users/nicolas.decat/Downloads/eeg_features_10-0_wsmi")

table_wsmi = table_wsmi.merge(
    eeg[['subject', 'Cluster', 'Stage-at-probe']].drop_duplicates(),
    on='subject',
    how='left'
)


# Get dimension and channel name
wsmi_cols = ['WSMI_delta','WSMI_theta','WSMI_alpha','WSMI_beta']
allchans = mne.io.read_raw_brainvision(vhdr_fname='/Users/nicolas.decat/Downloads/VD002/VD002.vhdr').info['ch_names']
physio_chan = ['IO','EMG chin','EMG arm','ECG']
chans = [name for name in allchans 
     if name not in physio_chan]

# Define ROI
frontal_channels = ['Fp1', 'Fp2', 'Fpz', 'AF3', 'AF4', 'AFz', 'F3', 'F4', 'Fz', 'F1', 'F2']
central_channels = ['FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'C1', 'C2', 'C3', 'C4', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4']
occipital_channels = ['P3', 'P1', 'Pz', 'P2', 'P4', 'PO3', 'PO1', 'POz', 'PO2', 'PO4', 'Oz', 'O1', 'O2']
roi_names = ['Frontal', 'Central', 'Occipital']
roi_channels = [frontal_channels, central_channels, occipital_channels]

# Find indices corresponding to ROI channels
roi_indices = {}
for roi_name, roi_chans in zip(roi_names, roi_channels):
    roi_indices[roi_name] = [i for i, ch in enumerate(chans) if ch in roi_chans]



##############################################################################################################
# BETWEEN-ROI CONNECTIVITY ####################################################################################
##############################################################################################################

# Define pairs of ROIs to compute between-ROI connectivity
roi_pairs = [
    ('Frontal', 'Central'),
    ('Central', 'Occipital'),
    ('Frontal', 'Occipital')
]

# We'll create a new list of results that includes both within- and between-ROI metrics
all_results = []

for _, row in table_wsmi.iterrows():
    subject = row['subject']
    wsmi_values = {'subject': subject}

    # Compute within-ROI connectivity (same as before)
    for band in ['WSMI_delta', 'WSMI_theta', 'WSMI_alpha', 'WSMI_beta']:
        matrix = row[band]

        # Between-ROI connectivity
        # Only pairs of channels from different ROIs are considered
        for roi1, roi2 in roi_pairs:
            ind1 = roi_indices[roi1]
            ind2 = roi_indices[roi2]
            # Extract the submatrix representing connections from ROI1 to ROI2
            submatrix = matrix[np.ix_(ind1, ind2)]
            # remove the 0s (remainings of the lower triangle)
            non_zero_values = submatrix[submatrix != 0]
            avg_between = np.mean(non_zero_values)
            median_between = np.median(non_zero_values)
            wsmi_values[f"{band}_{roi1}_{roi2}"] = median_between

    all_results.append(wsmi_values)

# Convert results to DataFrame
roi_avg_df = pd.DataFrame(all_results)

columns_to_plot = [
    "WSMI_delta_Frontal_Central", "WSMI_delta_Central_Occipital", "WSMI_delta_Frontal_Occipital",
    "WSMI_theta_Frontal_Central", "WSMI_theta_Central_Occipital", "WSMI_theta_Frontal_Occipital",
    "WSMI_alpha_Frontal_Central", "WSMI_alpha_Central_Occipital", "WSMI_alpha_Frontal_Occipital",
    "WSMI_beta_Frontal_Central",  "WSMI_beta_Central_Occipital",  "WSMI_beta_Frontal_Occipital"
]

# Add cluster info in roi_avg_df
roi_avg_df = roi_avg_df.merge(eeg[['subject', 'Cluster']], on='subject', how='left')
roi_avg_df = roi_avg_df.drop_duplicates(subset='subject')
roi_avg_df = roi_avg_df.dropna(subset=['Cluster'])
roi_avg_df['Cluster'] = roi_avg_df['Cluster'].astype(int)

cluster_colors = {'1': '#ff5733', '2': '#ffcc00', '3': '#3366ff', '4': '#cc33ff'}
roi_avg_df['color'] = roi_avg_df['Cluster'].map(cluster_colors)

# Organize based on frequencies (same logic as before)
freq_bins = {}
for col in columns_to_plot:
    parts = col.split('_')
    freq = parts[1]  # frequency is still the second element, e.g. 'delta', 'theta', etc.
    if freq not in freq_bins:
        freq_bins[freq] = []
    freq_bins[freq].append(col)

# Plotting each frequency in a separate figure
for freq, cols in freq_bins.items():
    n_cols = len(cols)
    fig, axes = plt.subplots(1, n_cols, figsize=(5*n_cols, 6))
    if n_cols == 1:
        axes = [axes]  # Ensure iterable even if only one subplot
    
    fig.suptitle(freq, fontsize=12)

    for ax, metric in zip(axes, cols):
        
        # Plot a boxplot with the custom color palette for clusters
        sns.boxplot(x="Cluster", y=metric, data=roi_avg_df, ax=ax, palette=cluster_colors)
        # Add individual data points
        sns.stripplot(x="Cluster", y=metric, data=roi_avg_df, ax=ax, color='black', size=4, jitter=True, alpha=0.7)
  
        # Make boxplots semi-transparent
        for patch in ax.artists:
            r, g, b, a = patch.get_facecolor()
            patch.set_facecolor((r, g, b, 0.5))  # half transparent
  
        ax.set_title(metric, fontsize=12)
        ax.set_xlabel("Cluster")
        ax.set_ylabel(metric)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()


#%% Stat analyses: CLUSTER 3 vs. each of the other clusters 

# MAKE SURE YOU RAN ABOVE SCRIPT BASED ON ALL DATA (NOT JUST W OR N1MSE)

import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.multitest import multipletests

# Load the dataset
df = roi_avg_df

# Add subject ID and Stage in the table
df['SubjectID'] = df['subject'].str[:5]
df['Stage'] = df['subject'].map(
   dict(zip(table_wsmi['subject'], table_wsmi['Stage-at-probe'])))

# Merge 1 and MSE values to make a N1MSE group
df['Stage'] = df['Stage'].replace(['1', 'MSE'], 'N1MSE')

# Reshape the dataset into long format for easier analysis
long_df = pd.melt(df, 
                  id_vars=["SubjectID", "Stage","Cluster"], 
                  value_vars=[col for col in df.columns if "WSMI_" in col],
                  var_name="Frequency_ROI",
                  value_name="WSMI")

# Extract Frequency and ROI from column names
long_df["Frequency"] = long_df["Frequency_ROI"].apply(lambda x: x.split("_")[1])
long_df["ROI"] = long_df["Frequency_ROI"].apply(lambda x: "_".join(x.split("_")[2:]))
long_df = long_df.drop(columns=["Frequency_ROI"])

# Run OLS
from statsmodels.formula.api import ols
summary_results = []
anova_detailed_wsmi = []

for freq in long_df["Frequency"].unique():
    for roi in long_df["ROI"].unique():
        sub_df = long_df[(long_df["Frequency"] == freq) & (long_df["ROI"] == roi)]
        
        # ANOVA model (exact same approach as your other metrics)
        model_an = ols("WSMI ~ C(Cluster) + C(Stage)", data=sub_df).fit()
        anova_table = sm.stats.anova_lm(model_an)
        
        # Extract p-values like in your existing code
        cluster_pval = anova_table.loc["C(Cluster)", "PR(>F)"]
        stage_pval = anova_table.loc["C(Stage)", "PR(>F)"]
        
        # Store the results
        result = {
            "Frequency": freq,
            "ROI": roi,
            "Cluster_effect": cluster_pval,
            "Stage_effect": stage_pval
        }
        summary_results.append(result)
        
        # Store detailed ANOVA results
        for effect in ["C(Cluster)"]:
            f_val = anova_table.loc[effect, "F"]
            p_val = anova_table.loc[effect, "PR(>F)"]
            df_num = anova_table.loc[effect, "df"]
            df_denom = anova_table.loc["Residual", "df"]
            
            anova_detailed_wsmi.append({
                "Frequency": freq,
                "ROI": roi,
                "Effect": effect.replace("C(", "").replace(")", ""),
                "F-value": f_val,
                "df (num)": df_num,
                "df (denom)": df_denom,
                "p-value (raw)": p_val
            })

# Convert to DataFrame
summary_df = pd.DataFrame(summary_results)
summary_df['Frequency_ROI'] = summary_df['Frequency'] + '_' + summary_df['ROI']
summary_df = summary_df[['Frequency_ROI', 'Cluster_effect', 'Stage_effect']]
anova_detailed_df = pd.DataFrame(anova_detailed_wsmi)

# summary_df.to_csv("/Users/nicolas.decat/Library/Mobile Documents/com~apple~CloudDocs/Documents/PhD/projects/NapNest/subjective_dimensions/wsmi_cluster_effect_2.csv", index=False)

'''
Ok so summary_df is  all the wsmi metrics that are significant, after OLS. Now, go back to clustering_anova script, see which ones are still significant after FDR
Then come back with those wsmi metrics that are significant after FDR (stored in corrected_results_df, remove manually the other metrics that are not wsmi), and do post hoc
''' 


# Post hoc (only on significant metrics)
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import pandas as pd

posthoc_results = []

# Filter significant metrics (adjust threshold as needed)
corrected = pd.read_csv("/Users/nicolas.decat/Library/Mobile Documents/com~apple~CloudDocs/Documents/PhD/projects/NapNest/subjective_dimensions/corrected_results_df.csv")
metrics = corrected['Metric'].dropna().astype(str)
metrics = metrics[metrics.str.contains("Occipital|Frontal|Central", case=False, regex=True)]

significant_tests = summary_df[summary_df["Cluster effect (corrected p)"] < 0.05]
significant_tests = summary_df[summary_df['Frequency_ROI'].astype(str).isin(metrics)].copy()


# Loop over each significant freq+ROI combination
# Prepare list to hold posthoc results
posthoc_results = []
mean_diff_summary = []  

# Loop over each significant freq+ROI combination
for _, row in significant_tests.iterrows():
    freq_roi = row["Frequency_ROI"]
    
    # Match from right in case ROI contains underscores
    split_index = freq_roi.find("_")
    freq = freq_roi[:split_index]
    roi = freq_roi[split_index + 1:]
    
    # Subset data
    sub_df = long_df[(long_df["Frequency"] == freq) & (long_df["ROI"] == roi)]

    # Perform Tukey HSD
    tukey = pairwise_tukeyhsd(endog=sub_df["WSMI"], groups=sub_df["Cluster"], alpha=0.05)

    # Convert results to DataFrame
    tukey_df = pd.DataFrame(data=tukey.summary().data[1:], columns=tukey.summary().data[0])

    # Add context columns
    tukey_df["Frequency"] = freq
    tukey_df["ROI"] = roi

    posthoc_results.append(tukey_df)
    
    # Add mean diff
    for _, row_tukey in tukey_df.iterrows():
        comparison = f"{row_tukey['group1']} vs {row_tukey['group2']}"
        mean_diff_summary.append({
            "Frequency": freq,
            "ROI": roi,
            "Comparison": comparison,
            "Mean difference": row_tukey["meandiff"]
        })

# Concatenate all posthoc results
all_posthoc_df = pd.concat(posthoc_results, ignore_index=True)
mean_diff_df = pd.DataFrame(mean_diff_summary)

# Reorder columns for readability
cols = ["Frequency", "ROI", "group1", "group2", "meandiff", "p-adj", "lower", "upper", "reject"]
all_posthoc_df = all_posthoc_df[cols]
significant_posthoc_df = all_posthoc_df[all_posthoc_df["reject"] == True]

# FDR
from statsmodels.stats.multitest import multipletests

# Apply FDR correction to p-values
pvals = all_posthoc_df["p-adj"].astype(float)
_, pvals_corrected, _, _ = multipletests(pvals, method='fdr_bh')

# Add corrected p-values and updated reject flag
all_posthoc_df["p_fdr"] = pvals_corrected
all_posthoc_df["reject_fdr"] = all_posthoc_df["p_fdr"] < 0.05

# Keep only significant ones
significant_posthoc_df_fdr = all_posthoc_df[all_posthoc_df["reject_fdr"] == True]


