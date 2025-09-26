#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 11 10:57:08 2024

NapNest: Ex&mining whether there's a main effect of the cluster on eeg

@author: nico
"""

#%% Prep everything for stat analyses 

import pandas as pd
import numpy as np
import os, mne
import matplotlib.pyplot as plt
from statsmodels.formula.api import ols
from scipy.stats import zscore
import statsmodels.api as sm

# Parameters data
len_seg_sec = 10
data = ''
condition = 'anova-avgchans' 

eeg_feat_col = ['rel_delta_ff','rel_theta_ff','rel_alpha_ff','rel_beta_ff','rel_gamma_ff','ratio_delta_alpha',
                'Kolmogorov','Permutation_Entropy_theta','Sample_Entropy',
                'slope','offset','total_power']

# Define paths 
path_table = '/Users/nicolas.decat/Library/Mobile Documents/com~apple~CloudDocs/Documents/PhD/projects/NapNest/subjective_dimensions/gigatable_4c.csv'
path_eeg = "/Volumes/disk-nico/napnest/napnest_eeg_feat/eeg_features_10-0.csv"

# Import table + eeg
table = pd.read_csv(path_table, index_col=0, sep=None)
eeg = pd.read_csv(path_eeg, index_col=0)

# Merge table and eeg
eeg[['Subject', 'Probe', 'Group']] = eeg['subject'].str.split('_', expand=True)
eeg = eeg.merge(table, 
                left_on=['Subject', 'Probe', 'Group'], 
                right_on=['Subject', 'Probe', 'Group'], 
                how='left')

# Add ratio alpha theta, total spectral power and inter-channel variance
eeg['ratio_delta_alpha'] = eeg['abs_delta'] / eeg['abs_alpha']
eeg['total_power'] = eeg[['abs_delta', 'abs_theta', 'abs_alpha', 'abs_beta', 'abs_gamma']].sum(axis=1)


# Import random raw data to get channel position
allchans = mne.io.read_raw_brainvision(vhdr_fname='/Volumes/disk-nico/napnest/data/VD002/VD002.vhdr').info['ch_names']
# allchans = mne.io.read_raw_brainvision(vhdr_fname='/Users/nicolas.decat/Downloads/VD002/VD002.vhdr').info['ch_names']
physio_chan = ['IO','EMG chin','EMG arm','ECG']
chans = [name for name in allchans 
     if name not in physio_chan]

# Replace 'N1' and 'MSE' with 'N1MSE' in the 'Stage-at-probe' column
eeg['Stage-at-probe'] = eeg['Stage-at-probe'].replace({'1': 'N1MSE', 'MSE': 'N1MSE'})

# Get dimension and EEG metric names
metrics_eeg = eeg_feat_col
dimension = 'Cluster'
metrics_eeg = eeg_feat_col
anova_results = pd.DataFrame(columns=["Metric", "Cluster effect", "Stage effect", "Interaction cluster-stage"])

cluster_effects_df = pd.DataFrame()
anova_results = pd.DataFrame()
all_subject_cook = pd.DataFrame()



#%% OLS | AVG CHAN | FDR

if condition == 'anova-avgchans':
        
    cluster_effects_df = pd.DataFrame()
    anova_detailed_results = []

    # Loop through each metric
    for metric in metrics_eeg:
        
        print(f"Processing metric: {metric}")
        
        # table with the current metric
        table_metric = eeg[
            ['subject', dimension, 'channel', metric,'Stage-at-probe']
            ].dropna()  
        table_metric.rename(columns={'Stage-at-probe': 'Stage'}, inplace=True)
        
        # Average across channels
        table_metric_avg = table_metric.groupby(
            ["subject", 
             "Cluster", 
             "Stage"
             ]
        ).agg({metric: 'mean'}).reset_index()
        table_metric = table_metric_avg
        
        # Z-score the metric column
        table_metric[metric] = table_metric[metric].apply(
            lambda x: (x - table_metric[metric].mean()) / table_metric[metric].std())
        
        table_metric['sub'] = table_metric['subject'].str[:5]

        # ANOVA: 
        model_an = ols(f'{metric} ~ C(Cluster) + C(Stage)', data=table_metric).fit()
        anova_table = sm.stats.anova_lm(model_an)
        anova_table
        
        # Compute Cook’s distance
        influence = model_an.get_influence()
        cooks_d = influence.cooks_distance[0]  # Extract Cook's distances (index 0)
        
        # Add it to the table
        table_metric['cooks_d'] = cooks_d
        
        # Compute subject-level influence by averaging Cook's distance
        subject_cook = table_metric.groupby('sub')['cooks_d'].mean().reset_index()
        subject_cook.columns = ['sub', 'avg_cooks_d']
        subject_cook['Metric'] = metric
        
        # Concatenate results to a global DataFrame (defined outside loop)
        if 'all_subject_cook' in locals():
            all_subject_cook = pd.concat([all_subject_cook, subject_cook], ignore_index=True)
        else:
            all_subject_cook = subject_cook.copy()
    
        # Extract p-values for the current metric
        cluster_pval = anova_table.loc["C(Cluster)", "PR(>F)"]
        stage_pval = anova_table.loc["C(Stage)", "PR(>F)"]
        # interaction_cluststage_pval = anova_table.loc["C(Cluster):C(Stage)", "PR(>F)"]
        
        # Append results to the DataFrame
        new_row = {
        "Metric": f"{metric}",  
        "Cluster effect": cluster_pval,
        "Stage effect": stage_pval,
        # "Interaction cluster-stage": interaction_cluststage_pval,
        }
        
        # Use pd.concat instead of append
        anova_results = pd.concat([anova_results, pd.DataFrame([new_row])], ignore_index=True)
        
        # Get the beta coefficients (effect sizes)
        coefficients = model_an.params
            
        effects_row = {'Metric': metric}
        
        # Add the intercept (baseline)
        baseline_cluster = coefficients.index[0]  # Usually "Intercept"
        effects_row['Baseline'] = coefficients[baseline_cluster]
        
        # Add each cluster coefficient
        for idx in coefficients.index:
            if idx != baseline_cluster:
                # Extract cluster name (removing the "C(Cluster)[T." prefix and "]" suffix)
                cluster_name = idx.replace("C(Cluster)[T.", "").replace("]", "")
                effects_row[f'Cluster_{cluster_name}'] = coefficients[idx]
            
        # Add p-value
        effects_row['p_value'] = cluster_pval
        
        # Add to the cluster effects DataFrame
        cluster_effects_df = pd.concat([cluster_effects_df, pd.DataFrame([effects_row])], ignore_index=True)

        print(all_subject_cook.sort_values(by='avg_cooks_d', ascending=False).head())
        
        
        # Add the detailed info of ANOVA (for table S)
        # Extract detailed ANOVA results for each effect (Cluster and Stage)
        for effect in ["C(Cluster)"]:
            f_value = anova_table.loc[effect, "F"]
            p_value = anova_table.loc[effect, "PR(>F)"]
            df_num = anova_table.loc[effect, "df"]
            df_denom = anova_table.loc["Residual", "df"]
    
            anova_detailed_results.append({
                'Metric': metric,
                'Effect': effect.replace("C(", "").replace(")", ""),
                'F-value': f_value,
                'df (num)': df_num,
                'df (denom)': df_denom,
                'p-value (raw)': p_value
            })


    # Convert final results to a DataFrame
    anova_results_df = pd.DataFrame(anova_results)
    anova_detailed_results_df = pd.DataFrame(anova_detailed_results)

    # Remove non significant results
    # anova_results_df = anova_results_df[anova_results_df['p_value'] <= 0.05]
        
    # Add wSMI before FDR correction (only to check the new fdr tests. But don't run further than fdr bc it'll not work)
    wsmi = pd.read_csv('/Users/nicolas.decat/Library/Mobile Documents/com~apple~CloudDocs/Documents/PhD/projects/NapNest/subjective_dimensions/wsmi_cluster_effect.csv')
    wsmi_renamed = wsmi.rename(columns={
        'Frequency_ROI': 'Metric',
        'Cluster_effect': 'Cluster effect',
        'Stage_effect': 'Stage effect'
    })
    wsmi_renamed['Interaction cluster-stage'] = np.nan
    anova_results_df = pd.concat([anova_results_df, wsmi_renamed], ignore_index=True)

    
    # Melt the DataFrame to have all p-values in a single column for correction
    melted_pvals = anova_results_df.melt(id_vars=["Metric"], value_vars=["Cluster effect", 
                                                                         "Stage effect", 
                                                                         # "Interaction cluster-stage"
                                                                         ],var_name="Effect", value_name="p_value").dropna()
    
    # Apply FDR correction    
    from statsmodels.stats.multitest import fdrcorrection
    melted_pvals = anova_results_df.melt(id_vars=["Metric"], value_vars=["Cluster effect"],var_name="Effect", value_name="p_value").dropna()
    rejected, pvals_corrected = fdrcorrection(melted_pvals['p_value'], alpha=0.05, method='indep', is_sorted=False)
    melted_pvals['p_value_corrected'] = pvals_corrected
    significant_pvals = melted_pvals[rejected]
    
    # Pivot back to wide format with only significant corrected p-values using pivot_table
    corrected_results_df = significant_pvals.pivot_table(index="Metric",columns="Effect",values="p_value_corrected",aggfunc='first').reset_index()
    corrected_results_df.columns.name = None  
    corrected_results_df = corrected_results_df.rename(columns={"Cluster effect": "Cluster effect (corrected p)"})
    
    # Post-hoc: pairwise cluster comparisons; what cluster drives the effects?
    from statsmodels.stats.multicomp import pairwise_tukeyhsd
    significant_cluster_metrics = corrected_results_df.dropna(subset=["Cluster effect (corrected p)"])["Metric"].tolist()
    
    pairwise_results = []
    tukey_detailed_results = []

    # Perform pairwise comparisons for each significant metric
    for metric in significant_cluster_metrics:
        
        # table with the current metric
        table_metric = eeg[
            ['subject', dimension, 'channel', metric,'Stage-at-probe']
            ].dropna()  
        table_metric.rename(columns={'Stage-at-probe': 'Stage'}, inplace=True)
        table_metric['sub'] = table_metric['subject'].str[:5]

        
        # Average across channels
        table_metric_avg = table_metric.groupby(
            ["subject", "Cluster", "Stage","sub"]
        ).agg({metric: 'mean'}).reset_index()
        table_metric = table_metric_avg

        # Average across channels
        table_metric_avg = table_metric.groupby(["subject", "Cluster", "Stage","sub"])[metric].mean().reset_index()
        
        # Z-score the metric column
        table_metric_avg[metric] = (table_metric_avg[metric] - table_metric_avg[metric].mean()) / table_metric_avg[metric].std()

        # Perform Tukey's HSD
        tukey = pairwise_tukeyhsd(endog=table_metric_avg[metric],
                                  groups=table_metric_avg['Cluster'],
                                  alpha=0.05)
        
        # Convert Tukey results to a DataFrame
        tukey_df = pd.DataFrame(data=tukey.summary().data[1:], columns=tukey.summary().data[0])
        
        # Rename columns for consistency
        tukey_df.rename(columns={
            'p-adj': 'p_value',
            'meandiff': 'mean_diff',
            'lower': 'ci_lower',
            'upper': 'ci_upper'
        }, inplace=True)
                
        # Extract relevant columns and rename
        tukey_df = tukey_df[['group1', 'group2', 'p_value', 'mean_diff','reject']]
        
        # Add metric information
        tukey_df['Metric'] = metric
        # Create a Comparison column
        tukey_df['group1'] = tukey_df['group1'].astype(str)
        tukey_df['group2'] = tukey_df['group2'].astype(str)
        tukey_df['Comparison'] = tukey_df['group1'] + " vs " + tukey_df['group2']        
        pairwise_results.append(tukey_df[['Metric', 'Comparison', 'p_value','mean_diff','reject']])
             
        
        # Store compact and detailed results
        pairwise_results.append(tukey_df[['Metric', 'Comparison', 'p_value']])
        tukey_detailed_results.append(tukey_df[['Metric', 'Comparison', 'mean_diff','p_value', 'reject']])
        
    pairwise_results_df = pd.concat(pairwise_results, ignore_index=True)
    tukey_detailed_df = pd.concat(tukey_detailed_results, ignore_index=True)
    
    # Filter significant results
    pairwise_significant_df = tukey_detailed_df[tukey_detailed_df['reject'] == True].reset_index(drop=True)
    
    # # Benjamini-Hochberg FDR correction 
    # from statsmodels.stats.multitest import multipletests
    # pairwise_BH_results_df = pd.concat(pairwise_results, ignore_index=True)
    # rejected_BH_pw, pvals_BH_corrected_pw, _, _ = multipletests(
    #     pairwise_BH_results_df['p_value'],
    #     alpha=0.05,
    #     method='fdr_bh'
    # )
    # pairwise_BH_results_df['p_value_corrected'] = pvals_BH_corrected_pw
    
    
    
    


#%% Check which model is best (with or without sub as fixed effect)

anova_results = pd.DataFrame(columns=["Metric", "Cluster effect", "Stage effect", "sub effect", 
                                      "AIC_with_sub", "BIC_with_sub", "AIC_no_sub", "BIC_no_sub"])

for metric in metrics_eeg:
    print(f"Processing metric: {metric}")
    
    table_metric = eeg[['subject', dimension, 'channel', metric, 'Stage-at-probe']].dropna()
    table_metric.rename(columns={'Stage-at-probe': 'Stage'}, inplace=True)
    
    # Average across channels
    table_metric_avg = table_metric.groupby(["subject", "Cluster", "Stage"]).agg({metric: 'mean'}).reset_index()
    table_metric = table_metric_avg
    
    # Z-score
    table_metric[metric] = (table_metric[metric] - table_metric[metric].mean()) / table_metric[metric].std()
    table_metric['sub'] = table_metric['subject'].str[:5]

    # Model WITH subject
    model_with_sub = ols(f'{metric} ~ C(Cluster) + C(Stage) + C(sub)', data=table_metric).fit()
    anova_table = sm.stats.anova_lm(model_with_sub)

    cluster_pval = anova_table.loc["C(Cluster)", "PR(>F)"]
    stage_pval = anova_table.loc["C(Stage)", "PR(>F)"]
    sub_pval = anova_table.loc["C(sub)", "PR(>F)"]

    aic_with_sub = model_with_sub.aic
    bic_with_sub = model_with_sub.bic

    # Model WITHOUT subject
    model_no_sub = ols(f'{metric} ~ C(Cluster) + C(Stage)', data=table_metric).fit()
    aic_no_sub = model_no_sub.aic
    bic_no_sub = model_no_sub.bic

    new_row = {
        "Metric": metric,
        "Cluster effect": cluster_pval,
        "Stage effect": stage_pval,
        "sub effect": sub_pval,
        "AIC_with_sub": aic_with_sub,
        "BIC_with_sub": bic_with_sub,
        "AIC_no_sub": aic_no_sub,
        "BIC_no_sub": bic_no_sub
    }

    anova_results = pd.concat([anova_results, pd.DataFrame([new_row])], ignore_index=True)

# sort metrics by AIC_with_sub for clarity
anova_results_sorted = anova_results.sort_values(by="AIC_with_sub", ascending=True)

import matplotlib.pyplot as plt
import seaborn as sns
# Plot AIC comparison
plt.figure(figsize=(12, 6))
sns.scatterplot(x="Metric", y="AIC_with_sub", data=anova_results_sorted, label="AIC with sub", marker='o')
sns.scatterplot(x="Metric", y="AIC_no_sub", data=anova_results_sorted, label="AIC without sub", marker='X')
plt.xticks(rotation=90)
plt.ylabel("AIC")
plt.title("AIC Comparison: With vs Without Subject Effect")
plt.legend()
plt.tight_layout()
plt.show()

# Plot BIC comparison
plt.figure(figsize=(12, 6))
sns.scatterplot(x="Metric", y="BIC_with_sub", data=anova_results_sorted, label="BIC with sub", marker='o')
sns.scatterplot(x="Metric", y="BIC_no_sub", data=anova_results_sorted, label="BIC without sub", marker='X')
plt.xticks(rotation=90)
plt.ylabel("BIC")
plt.title("BIC Comparison: With vs Without Subject Effect")
plt.legend()
plt.tight_layout()
plt.show()


