import io
import os

import tkinter as tk
from tkinter import ttk
from tkinter import *
from tkinter import scrolledtext
from tkinter import Tk
from tkinter.filedialog import askopenfilename

import seaborn as sns
import statsmodels.regression.linear_model as sm
from statsmodels.tools.tools import add_constant

import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

Tk().withdraw()
filename = askopenfilename()
df1 = pd.read_csv(filename)

root = tk.Tk()
root.title("Python GUI - Multiple Linear Regression Analysis")
root.geometry("800x650+10+10")
root.resizable(0, 0)

scr = scrolledtext.ScrolledText(root, width=75, height=20, wrap=tk.WORD)
scr.grid(column=0, columnspan=3)

pd.set_option('display.max_columns', 500)


def create_window():
    window = tk.Toplevel(root)


def corr():
    '''
    Compute Correlation between all variables
    '''
    scr.insert(tk.INSERT, "Correlation between all variables : ")
    scr.insert(tk.INSERT, '\n\n')
    scr.insert(tk.INSERT, df1.corr())
    scr.insert(tk.INSERT, '\n\n')

def RegAna():
    '''
    Compute Multiple Regression Model
    User select one Depedended Varaible and one or more Independent Variable(s)
    '''
    val1 = combo1.get()
    dep_var = val1.split(" ")
    values = [combo2.get(idx) for idx in combo2.curselection()]
    val2 = ','.join(values)
    ind_var = val2.split(",")
    if(val2 != ""):
        scr.insert(tk.INSERT, "Regression between " + val1 + " ~ " + val2 +
                   " : ")
        scr.insert(tk.INSERT, '\n\n')
        result = sm.OLS(df1[dep_var], add_constant(df1[ind_var])).fit()
        scr.insert(tk.INSERT, result.summary())
        scr.insert(tk.INSERT, '\n\n')
    else:
        scr.insert(tk.INSERT, 'Please Select Independent Variable(s)')
        scr.insert(tk.INSERT, '\n\n')

def resPlot():
    '''
    Draw Residual Plot
    '''
    val1 = combo1.get()
    dep_var = val1.split(" ")
    values = [combo2.get(idx) for idx in combo2.curselection()]
    val2 = ','.join(values)
    ind_var = val2.split(",")
    if(val2 != ""):
        scr.insert(tk.INSERT, '\n\n')
        result = sm.OLS(df1[dep_var], add_constant(df1[ind_var])).fit()
        pred_val = result.fittedvalues.copy()
        true_val = df1[val1]
        residual = true_val - pred_val
        fig, ax = plt.subplots(1, 1)
        ax.scatter(pred_val, residual)
        plt.title("Residual Plot - Residual V/S Fitted")
        plt.show()
        scr.insert(tk.INSERT, '\n\n')
    else:
        scr.insert(tk.INSERT, 'Alert!, Select Independent Variable(s)')
        scr.insert(tk.INSERT, '\n\n')


def probPlot():
    '''
    Draw Probability Plot
    '''
    val1 = combo1.get()
    dep_var = val1.split(" ")
    values = [combo2.get(idx) for idx in combo2.curselection()]
    val2 = ','.join(values)
    ind_var = val2.split(",")
    if(val2 != ""):
        scr.insert(tk.INSERT, '\n\n')
        result = sm.OLS(df1[dep_var], add_constant(df1[ind_var])).fit()
        pred_val = result.fittedvalues.copy()
        true_val = df1[val1]
        residual = true_val - pred_val
        fig, ax = plt.subplots(1, 1)
        stats.probplot(residual, plot=ax, fit=True)
        plt.title("Probability Plot")
        plt.show()
        scr.insert(tk.INSERT, '\n\n')
    else:
        scr.insert(tk.INSERT, 'Alert!, Select Independent Variable(s)')
        scr.insert(tk.INSERT, '\n\n')


def scatterPlot():
    '''
    Draw Scatter Plot
    '''
    val1 = combo1.get()
    dep_var = val1.split(" ")
    values = [combo2.get(idx) for idx in combo2.curselection()]
    val2 = ','.join(values)
    ind_var = val2.split(",")
    if(len(combo2.curselection()) > 0 and len(combo2.curselection()) < 4):
        scr.insert(tk.INSERT, '\n\n')
        sns.pairplot(df1, x_vars=ind_var, y_vars=dep_var, size=7, aspect=0.7,
                     kind='reg')
        plt.show()
        scr.insert(tk.INSERT, '\n\n')
    else:
        scr.insert(tk.INSERT, 'Maximum Three Independent Variables')
        scr.insert(tk.INSERT, '\n\n')


# Combo Box - 1
lbl_sel1 = ttk.Label(root,
                     text="Select Dependent Variable").grid(row=1, column=0)
ch1 = tk.StringVar()
combo1 = ttk.Combobox(root, width=12, textvariable=ch1)
combo1.grid(row=1, column=1)
combo1['values'] = list(df1)[1:]
combo1.current(0)

# Combo Box - 2
lbl_sel2 = ttk.Label(root,
                     text="Select Dependent Variable").grid(row=2, column=0)
ch2 = tk.StringVar()
combo2 = Listbox(root, width=15, height=10, selectmode=tk.MULTIPLE)
combo2.grid(row=2, column=1)
for i in list(df1)[1:]:
    combo2.insert(tk.END, i)

corr_btn = ttk.Button(root, text="Correlation between all variables", command=corr)
corr_btn.grid(row=3, column=1)

reg_btn = ttk.Button(root, text="Regression", command=RegAna)
reg_btn.grid(row=3, column=2)

scatter_btn = ttk.Button(root, text="Scatter Plot (Max. Three Pairs)",
                         command=scatterPlot)
scatter_btn.grid(row=4, column=0)

resPlot_btn = ttk.Button(root, text="Residual Plot-Linearity & Equal Variance",
                         command=resPlot)
resPlot_btn.grid(row=4, column=1)

probPlot_btn = ttk.Button(root, text="Probability Plot", command=probPlot)
probPlot_btn.grid(row=4, column=2)

exit_btn = ttk.Button(root, text="Exit", command=root.destroy)
exit_btn.grid(row=5, column=0)

root.mainloop()

win = tk.Tk()
win.title("Python GUI - Multiple Linear Regression Model")
win.overrideredirect(False)
win.geometry("{0}x{1}+10+10".format(win.winfo_screenwidth() - 100,
             win.winfo_screenheight()-100))
win.resizable(0, 0)
scr = scrolledtext.ScrolledText(win, width=200, height=30, wrap=tk.WORD)
scr.grid(column=0, columnspan=3)


def readBtn():
    '''
    Read data csv file
    '''
    global df
    Tk().withdraw()
    filename = askopenfilename()
    df = pd.read_csv(filename)
    scr.insert(tk.INSERT, '\n Data File ... \n\n')
    pd.set_option('display.max_columns', 500)
    df1 = df.drop(df.columns[0], axis=1)
    scr.insert(tk.INSERT, df1)


def missValue():
    '''
    Identify Missing Values
    '''
    scr.insert(tk.INSERT, '\n\n Missing Value Anaysis ... \n\n')
    pd.set_option('display.max_columns', 500)
    buf = io.StringIO()
    df.info(buf=buf)
    s = buf.getvalue()
    scr.insert(tk.INSERT, s)


def completeCase():
    '''
    It is not always feasible.
    Drop missing values
    '''
    scr.insert(tk.INSERT, '\n\n Complete Case only ... \n\n')
    df1 = df.drop(df.columns[0], axis=1)
    df1 = df1.dropna()
    if not os.path.isdir("imputedData"):
        os.mkdir("imputedData")
    path = os.getcwd() + "\\imputedData\\completeCases.csv"
    df1.to_csv(path)
    scr.insert(tk.INSERT, df1.describe())


def cleanData():
    '''
    Data Pre-Processing with Descriptive Statistics (using Median)
    It is suitable for numeric data
    '''
    df1 = df.drop(df.columns[0], axis=1)
    df1.fillna(df1.median(), inplace=True)
    if not os.path.isdir("imputedData"):
        os.mkdir("imputedData")
    path = os.getcwd() + "\\imputedData\\imputeMissingVal.csv"
    df1.to_csv(path)
    scr.insert(tk.INSERT, '\n\n Imputing Missing Value using Median ... \n\n')
    scr.insert(tk.INSERT, df1.describe())


#def summaryInfo():
#    '''
#    Explore Missing Value using Correlation
#    '''
#    x = df.drop(df.columns[0], axis=1)
#    x.fillna(1, inplace=True)
#    x[x != 1] = 0
#    scr.insert(tk.INSERT, '\n\n Explore Missing Value using Correlation \n\n')
#    scr.insert(tk.INSERT, x.corr())


def regAna():
    '''
    Multiple Regression Analysis
    '''
    from stat_study import create_window
    create_window()


read_btn = ttk.Button(win, text="Read Data - CSV File", command=readBtn)
read_btn.grid(row=1, column=0)

miss_btn = ttk.Button(win, text="is Missing Value?", command=missValue)
miss_btn.grid(row=3, column=0)

complete_btn = ttk.Button(win, text="Complete Cases - Descriptive Statistics",
                          command=completeCase)
complete_btn.grid(row=1, column=1)

cleanStr_btn = ttk.Button(win, text="Data Pre-Preparation - Impute Missing",
                          command=cleanData)
cleanStr_btn.grid(row=3, column=1)

exit_btn = ttk.Button(win, text="Exit", command=win.destroy)
exit_btn.grid(row=3, column=2)

reg_btn = ttk.Button(win, text="Regression Model", command=regAna)
reg_btn.grid(row=11, column=0, padx=50, pady=50)

win.mainloop()