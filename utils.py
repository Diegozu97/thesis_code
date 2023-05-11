import os
import pandas as pd
import datetime
from pytrends.request import TrendReq
import tqdm
from tqdm import tqdm
tqdm.pandas(desc="progress bar")
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


# Getting the correct thesis folder 
def search_folder(thesis_folder: str, search_name: str):
    # recursively search for the folder in the directory tree
    for dirpath, dirnames, filenames in os.walk(thesis_folder):
        if search_name in dirnames:
            folder_path = os.path.join(dirpath, search_name)
            return folder_path
        for dirname in dirnames:
            sub_folder_path = os.path.join(dirpath, dirname)
            if os.path.isdir(sub_folder_path):
                search_folder(sub_folder_path, search_name)


def plot_variables(check, column_name1, column_name2):

    # create a figure and an axis object
    fig, ax1 = plt.subplots(figsize=(12, 6))
    # plot the sentiment column on the primary y-axis
    ax1.plot(check.index, check[column_name1], color='blue')
    ax1.set_ylabel(column_name1)

    # create a secondary y-axis
    ax2 = ax1.twinx()

    # plot the Close column on the secondary y-axis
    ax2.plot(check.index, check[column_name2], color='red')
    ax2.set_ylabel(column_name2)

    # add a legend
    ax1.legend([column_name1], loc='upper left')
    ax2.legend([column_name2], loc='upper right')

    # show the plot
    plt.show()
    
    
def read_modelling(path, name):
    df = pd.read_csv(path+name)
    df["datetime"] = pd.to_datetime(df["datetime"])
    df.sort_values(by = "datetime", ascending = True)
    return df


def model_evaluate(y_test, y_pred):
    
    from sklearn.metrics import classification_report
    from sklearn.metrics import confusion_matrix
    
    # classification report
    print(classification_report(y_test, y_pred))
    
    # confusion report
    cf_matrix = confusion_matrix(y_test, y_pred)
    
    categories = ['Negative', 'Positive']
    
    group_names = ['True Neg', 'False Pos','False Neg','True Pos']
    
    group_percentages = ['{0:.2%}'.format(value) for value in cf_matrix.flatten() / np.sum(cf_matrix)]

    labels = [f'{v1}\n{v2}' for v1, v2 in zip(group_names,group_percentages)]
    labels = np.asarray(labels).reshape(2,2)

    sns.heatmap(cf_matrix, annot = labels, cmap = 'Blues',fmt = '',
                xticklabels = categories, yticklabels = categories)

    plt.xlabel("Predicted values", fontdict = {'size':14}, labelpad = 10)
    plt.ylabel("Actual values"   , fontdict = {'size':14}, labelpad = 10)
    plt.title ("Confusion Matrix", fontdict = {'size':18}, pad = 20)
    
    
def fix_data(df, date_column): 
    for idx in tqdm(range(len(df))):
        
        current_date = pd.to_datetime(df[date_column].iloc[idx], errors='coerce')
        
        if current_date.weekday() == 6:
            df.at[idx, date_column] = current_date + pd.Timedelta(days=2)
        
        elif current_date.weekday() == 7: 
            
            df.at[idx, date_column] = current_date + pd.Timedelta(days=1) 
        else:
            continue
    
    return df


def plot_three_line_chart(df, x_col, y_col1, y_col2, y_col3):
    # create a figure and an axis object
    fig, ax1 = plt.subplots(figsize=(12, 6))
    
    # plot the first column on the primary y-axis
    ax1.plot(df[x_col], df[y_col1], color='blue')
    ax1.set_ylabel(y_col1)

    # plot the third column on the primary y-axis
    ax1.plot(df[x_col], df[y_col3], color='lightblue')
    ax1.set_ylabel(y_col3)

    # create a secondary y-axis
    ax2 = ax1.twinx()

    # plot the second column on the secondary y-axis
    ax2.plot(df[x_col], df[y_col2], color='gray')
    ax2.set_ylabel(y_col2)

    # add a legend
    ax1.legend([y_col1, y_col3], loc='upper left')
    ax2.legend([y_col2], loc='upper right')

    # show the plot
    plt.show()