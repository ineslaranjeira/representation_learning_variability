"""
Plots respondents as a fraction of total number of people per role
"""
#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.io
import os
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches
from datetime import datetime

current_path="/Volumes/ctessereau/Backup_USB_Charline/PostdocSwap/IBL/IDEA/"
current_path='/home/ines/repositories/representation_learning_variability/IBL_IDEA/'
current_path='/Users/ineslaranjeira/Documents/Repositories/representation_learning_variability/IBL_IDEA/'

# current_path="C:/proj/int-brain-lab/survey-analysis/"
df_1 = pd.read_csv(current_path+"Anonymous IBL Survey 2022.csv")
df_2 = pd.read_csv(current_path+"Anonymous IBL Survey 2023.csv")
df_3 = pd.read_csv(current_path+"Anonymous IBL Survey 2024.csv")
df_4 = pd.read_csv(current_path+"Anonymous IBL Survey 2025.csv")
df_demo = pd.read_csv(current_path+"Consortium Member List - Full List.csv")
dfs = [df_1, df_2, df_3, df_4]
offset = 0.05
markersize = 8
fs = 20

#%%
# Fix weird value and remove uninformative NaNs
df_demo.loc[df_demo['Join Date*']=='***']= np.nan 
reduced_df = df_demo[['Membership Type', 'Join Date*', 'Leave Date']]
reduced_df['Leave Date'] = reduced_df['Leave Date'].replace([np.nan], ['12/01/2025']) # if person didn't leave, put leave date in the future
reduced_df['Membership Type'] = reduced_df['Membership Type'].replace(['Graduate Student'], ['Student'])
reduced_df = reduced_df.dropna()

join_dates = reduced_df['Join Date*'].apply(lambda x: datetime.strptime(x, '%m/%d/%Y'))
leave_dates = reduced_df['Leave Date'].apply(lambda x: datetime.strptime(x, '%m/%d/%Y'))

threshold_2025 = '05/01/2025'
threshold_2024 = '05/01/2024'
threshold_2023 = '05/01/2023'
threshold_2022 = '05/01/2022'
keep_2025 = (join_dates < datetime.strptime(threshold_2025, '%m/%d/%Y')) & (
             leave_dates >= datetime.strptime(threshold_2025, '%m/%d/%Y')) 
keep_2024 = (join_dates < datetime.strptime(threshold_2024, '%m/%d/%Y')) & (
             leave_dates >= datetime.strptime(threshold_2024, '%m/%d/%Y')) 
keep_2023 = (join_dates < datetime.strptime(threshold_2023, '%m/%d/%Y')) & (
             leave_dates >= datetime.strptime(threshold_2023, '%m/%d/%Y')) 
keep_2022 = (join_dates < datetime.strptime(threshold_2022, '%m/%d/%Y')) & (
             leave_dates >= datetime.strptime(threshold_2022, '%m/%d/%Y')) 
#%%
position_2025 = reduced_df['Membership Type'][keep_2025]
position_2024 = reduced_df['Membership Type'][keep_2024]
position_2023 = reduced_df['Membership Type'][keep_2023]
position_2022 = reduced_df['Membership Type'][keep_2022]

# Position of the respondents 
respondent_position_2025 = df_4['What role best describes your position in IBL?']
respondent_position_2024 = df_3['What role best describes your position in IBL?']
respondent_position_2023 = df_2['What role best describes your position in IBL?']
respondent_position_2022 = df_1['What role best describes your position in IBL?']

positions = ['PI', 'Postdoc', 'Staff', 'Student']

# Calculate fraction of respondents
position_count_2025 = position_2025.value_counts()
respondent_position_2025_count = respondent_position_2025.value_counts()
fraction_respondents_per_position_2025 = respondent_position_2025_count[positions] / position_count_2025[positions]

position_count_2024 = position_2024.value_counts()
respondent_position_2024_count = respondent_position_2024.value_counts()
fraction_respondents_per_position_2024 = respondent_position_2024_count[positions] / position_count_2024[positions]

position_count_2023 = position_2023.value_counts()
respondent_position_2023_count = respondent_position_2023.value_counts()
fraction_respondents_per_position_2023 = respondent_position_2023_count[positions] / position_count_2023[positions]

positions = ['PI', 'Postdoc', 'Staff', 'Technician']
position_count_2022 = position_2022.value_counts()
respondent_position_2022_count = respondent_position_2022.value_counts()
fraction_respondents_per_position_2022 = respondent_position_2022_count[positions] / position_count_2022[positions]
#%%
# Merge
fraction_respondents_per_position_2022.name = '2022 - N='+ str(len(respondent_position_2022))
fraction_respondents_per_position_2023.name = '2023 - N='+ str(len(respondent_position_2023))
fraction_respondents_per_position_2024.name = '2024 - N='+ str(len(respondent_position_2024))
fraction_respondents_per_position_2025.name = '2025 - N='+ str(len(respondent_position_2025))

merged_1 = pd.merge(fraction_respondents_per_position_2022, 
                     fraction_respondents_per_position_2023, left_index=True, right_index=True, how='outer')
merged_2 = pd.merge(merged_1, 
                     fraction_respondents_per_position_2024, left_index=True, right_index=True, how='outer')
merged = pd.merge(merged_2, 
                     fraction_respondents_per_position_2025, left_index=True, right_index=True, how='outer')
index = 'What role best describes your position in IBL?'
melted = pd.melt(merged.reset_index(), id_vars=[index], value_vars=['2022 - N='+ str(len(respondent_position_2022)), 
                                                                      '2023 - N='+ str(len(respondent_position_2023)), 
                                                                      '2024 - N='+ str(len(respondent_position_2024)),
                                                                      '2025 - N='+ str(len(respondent_position_2025))])

#%%
# Plot
melted = melted.rename(columns={'variable': 'Year'})

# melted.loc[(melted['Year'] =='2022 - N='+ str(len(respondent_position_2022))), 'value'] = np.nan
# melted.loc[(melted['Year'] =='2023 - N='+ str(len(respondent_position_2023))), 'value'] = np.nan
# melted.loc[(melted['Year'] =='2024 - N='+ str(len(respondent_position_2024))), 'value'] = np.nan

color_palette = sns.dark_palette("#69d", 4)

plt.figure(figsize=(8, 5))
sns.barplot(x=index, y='value', hue='Year', data=melted, palette=color_palette)
sns.despine()
plt.ylim([0, 1])
plt.xticks(fontsize=fs-5)
plt.yticks(fontsize=fs-5)
plt.xlabel('Role in the IBL', fontsize=fs)
plt.ylabel('Fraction of respondents', fontsize=fs)
plt.title('Responsiveness to the survey', fontsize=fs)
# plt.show()
plt.tight_layout()
plt.savefig(current_path+'Responsiveness to survey')
plt.close()
# %%
