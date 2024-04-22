"""
    Written by Sebastian, probably ruined by Andrew.
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
# current_path="C:/proj/int-brain-lab/survey-analysis/"
df_1 = pd.read_csv(current_path+"Anonymous IBL Survey 2022.csv")
df_2 = pd.read_csv(current_path+"Anonymous IBL Survey 2023.csv")
df_3 = pd.read_csv(current_path+"Anonymous IBL Survey 2024.csv")
df_demo = pd.read_csv(current_path+"Consortium Member List - Full List.csv")
dfs = [df_1, df_2, df_3]
offset = 0.05
markersize = 8
fs = 20

#%%

# Fix weird value and remove uninformative NaNs
df_demo.loc[df_demo['Join Date*']=='***']= np.nan 
reduced_df = df_demo[['Membership Type', 'Join Date*', 'Leave Date']]
reduced_df['Leave Date'] = reduced_df['Leave Date'].replace([np.nan], ['12/01/2024'])
reduced_df['Membership Type'] = reduced_df['Membership Type'].replace(['Graduate Student'], ['Student'])
reduced_df = reduced_df.dropna()

join_dates = reduced_df['Join Date*'].apply(lambda x: datetime.strptime(x, '%m/%d/%Y'))
leave_dates = reduced_df['Leave Date'].apply(lambda x: datetime.strptime(x, '%m/%d/%Y'))

threshold_2024 = '05/01/2024'
threshold_2023 = '05/01/2023'
threshold_2022 = '05/01/2022'
keep_2024 = (join_dates < datetime.strptime(threshold_2024, '%m/%d/%Y')) & (
             leave_dates >= datetime.strptime(threshold_2024, '%m/%d/%Y')) 
keep_2023 = (join_dates < datetime.strptime(threshold_2023, '%m/%d/%Y')) & (
             leave_dates >= datetime.strptime(threshold_2023, '%m/%d/%Y')) 
keep_2022 = (join_dates < datetime.strptime(threshold_2022, '%m/%d/%Y')) & (
             leave_dates >= datetime.strptime(threshold_2022, '%m/%d/%Y')) 
#%%

position_2024 = reduced_df['Membership Type'][keep_2024]
position_2023 = reduced_df['Membership Type'][keep_2023]
position_2022 = reduced_df['Membership Type'][keep_2022]

# Position of the respondents 
# respondent_position_2024 = df_3.loc[df_3['What role best describes your position in IBL?'
#                                     ]!='Prefer not to say', 'What role best describes your position in IBL?']
# respondent_position_2023 = df_2.loc[df_2['What role best describes your position in IBL?'
#                                     ]!='Prefer not to say', 'What role best describes your position in IBL?']
# respondent_position_2022 = df_1.loc[df_1['What role best describes your position in IBL?'
#                                     ]!='Prefer not to say', 'What role best describes your position in IBL?']

# Position of the respondents 
respondent_position_2024 = df_3['What role best describes your position in IBL?']
respondent_position_2023 = df_2['What role best describes your position in IBL?']
respondent_position_2022 = df_1['What role best describes your position in IBL?']

positions = ['PI', 'Postdoc', 'Staff', 'Student']

# Calculate fraction of respondents
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

merged_ts = pd.merge(fraction_respondents_per_position_2022, 
                     fraction_respondents_per_position_2023, left_index=True, right_index=True, how='outer')
merged = pd.merge(merged_ts, 
                     fraction_respondents_per_position_2024, left_index=True, right_index=True, how='outer')

melted = pd.melt(merged.reset_index(), id_vars=['index'], value_vars=['2022 - N='+ str(len(respondent_position_2022)), 
                                                                      '2023 - N='+ str(len(respondent_position_2023)), 
                                                                      '2024 - N='+ str(len(respondent_position_2024))])
#%%
# Plot
color_palette = sns.color_palette("hls", 8)
melted = melted.rename(columns={'variable': 'Year'})
sns.barplot(x='index', y='value', hue='Year', data=melted, palette=color_palette)
sns.despine()
plt.ylim([0, 1])
plt.xticks(fontsize=fs-5)
plt.yticks(fontsize=fs-5)
plt.xlabel('Role in the IBL', fontsize=fs)
plt.ylabel('Fraction of respondents', fontsize=fs)
plt.title('Responsiveness to the survey', fontsize=fs)
# %%

def plot_help(df, question, color,base):
    values = df[question].values
    ps, ns = np.unique(values, return_counts=True)
    # Initialize DataFrame to store x-offsets
    ps, ns = np.unique(values, return_counts=True)
    counts={ps:0 for ps in ps }
    for index, (idx, row) in enumerate(df.iterrows()):
        value=row[question]
        if not np.isnan(value):  # Skip NaN values
            num_points_for_value = np.sum(values == value)  # Count the number of points with the same value
            x_offset = np.linspace(-offset * (num_points_for_value - 1) / 2, offset * (num_points_for_value - 1) / 2, num_points_for_value)
            plt.plot(base + x_offset[counts[value]], value, 'o', color=color[idx], markersize=markersize)
            counts[value]+=1
    temp_mean = np.nanmean(values)
    temp_median = np.nanmedian(values)
    plt.plot([base - 0.25, base + 0.25], [temp_mean, temp_mean], 'r')
    plt.plot([base - 0.25, base + 0.25], [temp_median, temp_median], 'k')
    return temp_mean, temp_median

def plot(df_1, df_2, df_3,col_1,col_2,col_3,color_dict, question, min, max, current_path,short_title):
    means, medians = [], []
    q = question if type(question) == str else question[0]
    a, b = plot_help(df_1, q,col_1, base=2022)
    means.append(a)
    medians.append(b)
    q = question if type(question) == str else question[1]
    a, b = plot_help(df_2, q,col_2, base=2023)
    means.append(a)
    medians.append(b)
    q = question if type(question) == str else question[2]
    a, b = plot_help(df_3, q,col_3, base=2024)
    means.append(a)
    medians.append(b)
    plt.plot([2022, 2023,2024], means, 'm')
    plt.plot([2022, 2023,2024], medians, 'k')
    plt.xticks([2022, 2023,2024], fontsize=fs-4)
    plt.yticks(range(min, 1 + max), fontsize=fs-4)
    plt.title(short_title, fontsize=fs)
    plt.ylabel("Rating", fontsize=fs)
    plt.xlabel("Year", fontsize=fs)
    legend_patches = [mpatches.Patch(color=color, label=response) for response, color in color_dict.items()]
    plt.legend(handles=legend_patches, loc='upper left', bbox_to_anchor=(0, -0.2), ncol=4)
    sns.despine()
    plt.tight_layout()
    plt.savefig(current_path+short_title)
    plt.close()

# extract all response to 'What role best describes your position in IBL?' 
def extract_color_schemes_per_role_in_IBL(df1,df2,df3):
    df_demo
    question = 'What role best describes your position in IBL?'
    responses_df1 = df1[question]
    responses_df2 = df2[question]
    responses_df3 = df3[question]
    # Concatenate the responses from all three DataFrames
    all_responses = pd.concat([responses_df1, responses_df2, responses_df3], ignore_index=True)
    # Extract unique responses and assign a unique color to each response
    unique_responses = all_responses.dropna().unique()
    colors = plt.cm.tab10(range(len(unique_responses)))
    color_dict = {response: color for response, color in zip(unique_responses, colors)}
    color_dict[np.nan] = 'lightgray'  # Assign a color for 'nan' values
    # Associate each respondent in each DataFrame with a color based on their response
    color_map_df1 = responses_df1.map(color_dict)
    color_map_df2 = responses_df2.map(color_dict)
    color_map_df3 = responses_df3.map(color_dict)
    return color_map_df1,color_map_df2, color_map_df3,color_dict

col_1,col_2,col_3,color_dict=extract_color_schemes_per_role_in_IBL(df_1,df_2,df_3)

plot(*dfs,col_1,col_2,col_3,color_dict, 'How happy are you as a member of IBL?', 1, 5,current_path, "Happiness level")
plot(*dfs,col_1,col_2,col_3,color_dict, 'How integrated do you feel as a member of IBL?', 1, 5, current_path,"Integrated IBL")
plot(*dfs,col_1,col_2,col_3,color_dict, 'How integrated do you feel as a member of IBL?', 1, 5, current_path,"Integrated home lab")

plot(*dfs,col_1,col_2,col_3,color_dict, 'What percentage of your time do you spend working for IBL?', 1, 10, current_path,"Percentage worked on IBL")

plot(*dfs,col_1,col_2,col_3,color_dict, 'Do you feel explicit or implicit pressure to work more hours than you feel is healthy?', 1, 5, current_path,"Work time pressure")
plot(*dfs,col_1,col_2,col_3,color_dict, 'Do you feel explicit or implicit pressure to work more hours on IBL projects than you would have liked?	', 1, 5, current_path,"Pressur IBL projects")

plot(*dfs,col_1,col_2,col_3,color_dict, 'Do you feel explicit or implicit pressure to avoid taking vacations?', 1, 5, current_path,"Vacation pressure")

qs = ['Have you experienced or witnessed a hostile work environment in IBL? (bullying, gender harassment, sexual harassment)','Have you experienced or witnessed a hostile work environment in IBL? (bullying, gender harassment, sexual harassment)', 'Have you experienced or witnessed a hostile work environment in IBL? (bullying, gender harassment, sexual harassment)']
plot(*dfs,col_1,col_2,col_3,color_dict, qs, 1, 10, current_path,"Hostile work environment")

plot(*dfs,col_1,col_2,col_3,color_dict, 'How well does IBL communicate as a whole?', 1, 10, current_path,"Lab communication")




plot(*dfs,col_1,col_2,col_3,color_dict, 'Do you feel it is easy to get information from your colleagues in the IBL?', 1,5, current_path,"Getting info from colleagues")
plot(*dfs,col_1,col_2,col_3,color_dict, 'Do you feel it is easy to get information from IBL task force leader(s) or working group chairs you work with?', 1, 5, current_path,"info from leaders")
plot(*dfs,col_1,col_2,col_3,color_dict, 'How useful do you find the various IBL working group and task force meetings?', 1, 5, current_path,"useful WGTF meetings")
# plot(*dfs,col_1,col_2,col_3,color_dict, 'How comfortable do you feel sharing personal concerns with larger parts of the lab (e.g. during catch-up or social FIKA)', 1, 10, "Lab concern sharing")



plot(*dfs,col_1,col_2,col_3,color_dict, 'How useful do you find the Wednesday IBL lab meeting?', 1, 5, current_path,"Wednesday Lab meetings useful")



plot(*dfs,col_1,col_2,col_3,color_dict, 'How easy is it for you to locate resources in the IBL?	', 1, 5, current_path,"locate ressources")


plot(*dfs,col_1,col_2,col_3,color_dict, 'How often do you encounter issues that could have been averted if things were better organized?  (e.g., documentation is missing/poor quality, equipment problems, servers down.)', 1, 5, current_path,"frequency disorganised errors")

plot(*dfs,col_1,col_2,col_3,color_dict, 'How supported do you feel by IBL PIs?', 1, 5, current_path,"IBL PIs support")

plot(*dfs,col_1,col_2,col_3,color_dict, 'Do you feel you are getting adequate mentoring, career advice, and general guidance to succeed from within the consortium?', 1, 5, current_path,"Guidance to succeed from consortim")

plot(*dfs,col_1,col_2,col_3,color_dict, 'Do you feel the IBL adequately provides you with the tools and technologies you need?', 1, 5, current_path,"Tools and technology provide")


plot(*dfs,col_1,col_2,col_3,color_dict, 'What is your opinion of the level of technical support in the IBL?', 1, 5, current_path,"Level technical support")
plot(*dfs,col_1,col_2,col_3,color_dict, 'What is your opinion of the level of administrative support in the IBL?', 1, 5, current_path,"Level administrative support")



questions=['Do you feel you have adequate support to take on tasks in your task forces or working groups?',	'Do you feel that you have adequate opportunities to learn new skills in your task forces or working groups?',	'Do you feel that you have adequate opportunities to learn new skills on your personal project?',	'How do you feel about the level of feedback on your work in task forces or working groups? ', 	'How do you feel about the level of feedback on your work on your personal project? ', 	'How do you feel about the level of guidance from your IBL task force leader(s) or working group chairs?	',		'How do you feel about the level of guidance from your IBL home PI?',	'How do you feel about the level of guidance from your PIs on your personal project?', 'How do you feel about the level of formal feedback you receive on your progress?','What is your impression of the funding of the IBL?', 'What is your impression of the funding of the IBL?', "How familiar are you with the IBL's authorship policies?"]


titles=['adequate support tasks TF WG',	'learn new skills TF WG',	'learn new skills on your personal project',	'level of feedback on your work in TF WG', 	'level of feedback on your personal project', 	'level of guidance from your IBL TF WG chairs',		'guidance from your IBL home PI',	'level of guidance from your PIs personal project', 'level of formal feedback you receive on your progress','funding of the IBL', 'funding of the IBL', 'IBLs authorship policies']

idx=0
for q in questions:
    plot(*dfs,col_1,col_2,col_3,color_dict, q, 1, 5, current_path,titles[idx])
    idx+=1
#%%