# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
""" seaborn leanring 
https://www.geeksforgeeks.org/seaborn-kdeplot-a-comprehensive-guide/?ref=lbp
https://seaborn.pydata.org/tutorial.html

"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Selecting style as white,
# dark, whitegrid, darkgrid 
# or ticks
# Context: plotting context parameters
# Style: defines style
# Palette: sets a color palette
# Font
# Font_scale: sets font size
# Color_codes: If set True then palette is activated, short hand notations for colors can be remapped from the palette.
sns.set_theme(style="whitegrid")
#sns.set_style(style='ticks')

# The possible value of the palette are:
# https://www.geeksforgeeks.org/seaborn-color-palette/?ref=lbp

"""### Dot Plot ###"""
""" stripplot 1 """ 
# x axis values
x =['sun', 'mon', 'fri', 'sat', 'tue', 'wed', 'thu']
# y axis values
y =[5, 6.7, 4, 6, 2, 4.9, 1.8]
# plotting strip plot with seaborn
ax = sns.stripplot(x, y);
# giving labels to x-axis and y-axis
ax.set(xlabel ='Days', ylabel ='Amount_spend')
# giving title to the plot
plt.title('stripplot 1');
# function to show plot
plt.show()

""" stripplot 2 """ 
# reading the dataset
df = sns.load_dataset('tips')
sns.stripplot(x='day', y='total_bill', data=df,
              jitter=True, hue='smoker', dodge=True)
plt.title('stripplot 2')
plt.show()  # Show plot


""" swarmplot """ 
# loading data-set
iris = sns.load_dataset('iris')
 
# plotting strip plot with seaborn
# deciding the attributes of dataset on
# which plot should be made
# ax = sns.swarmplot(x='species', y='sepal_length', data=iris)
ax = sns.swarmplot(x='sepal_length', y='species', data=iris)
# giving title to the plot
plt.title('swarmplot 1')
 
# function to show plot
plt.show()


"""### Leanear Relationship Plot ###"""
# set grid style
sns.set(style ="darkgrid")
# import dataset
tips = sns.load_dataset("tips")

sns.relplot(data=tips, x="total_bill", y="tip", hue="day")
plt.title('Leanear Relation Plot 1')
plt.show()  # Show plot

sns.relplot(data=tips, x="total_bill", y="tip", hue="day", col="time")
plt.title('Leanear Relation Plot 2')
plt.show()  # Show plot

sns.relplot(data=tips, x="total_bill", y="tip", hue="day", col="time", row="sex")
plt.title('Leanear Relation Plot 3')
plt.show()  # Show plot

sns.relplot(data=tips, x="total_bill", y="tip", size="size")
plt.title('Leanear Relation Plot 4')
plt.show()  # Show plot

sns.relplot(data=tips, x="total_bill", y="tip", hue="day", size="size")
plt.title('Leanear Relation Plot 5')
plt.show()  # Show plot

sns.relplot(data=tips, x="total_bill", y="tip", hue="day", size="size", style="sex")
plt.title('Leanear Relation Plot 6')
plt.show()  # Show plot


# load data
sns.set(style ="white")
tips = sns.load_dataset("tips")
sns.lmplot(x="total_bill", y="tip", hue="smoker", data=tips, markers=["o", "x"])
plt.title('Leanear Regration Plot')
plt.show()  # Show plot



"""### jointplot (with distribution at side of diagram) ###"""
sns.set(style ="darkgrid")
tips = sns.load_dataset("tips")
  
# here "*" is used as a marker for scatterplot
sns.jointplot(data=tips, x="total_bill", y="tip", kind="reg", marker="*")
plt.title('jointplot 1')
plt.show()  # Show plot


tips = sns.load_dataset("tips")
sns.jointplot(data=tips, x="total_bill", y="tip", hue="time")
plt.title('jointplot 2')
plt.show()  # Show plot


"""### line chart ###"""
flights = sns.load_dataset("flights")
flights_ = flights.loc[flights['month'].isin(['May', 'Apr']),:]
sns.lineplot(data=flights_, x="year", y="passengers", hue="month", err_style="band")
plt.title('lineplot 1')
plt.show()  # Show plot

fmri = sns.load_dataset("fmri")
sns.lineplot(
    data=fmri,
    x="timepoint", y="signal", hue="event", style="event",
    markers=True, dashes=False
)
plt.title('lineplot 2')
plt.show()  # Show plot

import numpy as np
x, y = np.random.normal(size=(2, 5000)).cumsum(axis=1)
sns.lineplot(x=x, y=y, sort=False, lw=1)
plt.title('tracking plot')
plt.show()  # Show plot


# 2 columns combine into one line chart
"""
sns.lineplot( x = 'Date',
             y = 'Births',
             data = data,
             label = 'DailyBirths')
  
sns.lineplot( x = 'Date',
             y = '7day_rolling_avg',
             data = data,
             label = 'Rollingavg')
plt.xlabel('Months of the year 1959')
plt.title('lineplot 3')
plt.show()  # Show plot
"""

"""### barplot ###"""
df = sns.load_dataset('titanic')
# class v / s fare barplot
import numpy as np
sns.barplot(x = 'class', y = 'fare', data = df, hue = 'adult_male',
            estimator=np.mean, order=df.sort_values('fare').loc[:,'class'].unique())
plt.title('barplot 1')
plt.show()

sns.barplot(x = 'class', y = 'fare', data = df, hue = 'adult_male',estimator=np.median, palette ='plasma')
plt.title('barplot 2')
plt.show()
"""countplot"""
df = sns.load_dataset('tips')
# count plot on two categorical variable
sns.countplot(x ='sex', hue = "smoker", data = df)
plt.title('countplot 1')
plt.show()



""" factorplot """ 
# Load data
titanic = sns.load_dataset("titanic")
# Set up a factorplot
g = sns.factorplot("class", "survived", "sex", data=titanic, kind="bar", 
                   palette="muted", legend=True)
plt.show()



"""### statistical plot ###"""

"""Histograms and Distributions""" 
penguins = sns.load_dataset("penguins")
sns.histplot(data = penguins,  x="flipper_length_mm", kde = True)
plt.title('Histograms 1')
plt.show()

sns.histplot(data = penguins,  x="flipper_length_mm", hue="species", kde = True)
plt.title('Histograms 2')
plt.show()


sns.displot(penguins, x="flipper_length_mm",hue="species", kde=True)
plt.title('Histograms 3')
plt.show()

sns.displot(penguins, x="flipper_length_mm", hue="species", kind="kde", fill=True)
plt.title('Histograms 4')
plt.show()

sns.displot(penguins, x="bill_length_mm", y="bill_depth_mm")
plt.title('Histograms 5')
plt.show()

sns.displot(penguins, x="bill_length_mm", y="bill_depth_mm", kind="kde")
plt.title('Histograms 6')
plt.show()

sns.displot(penguins, x="bill_length_mm", y="bill_depth_mm", hue="species")
plt.title('Histograms 7')
plt.show()

sns.displot(penguins, x="bill_length_mm", y="bill_depth_mm", hue="species", kind="kde")
plt.title('Histograms 8')
plt.show()

""" violinplot """ 
# Load data
tips = sns.load_dataset("tips")
# Set up a violinplot
ax = sns.violinplot(x="day", y="total_bill", hue="smoker",
                    data=tips, palette="muted", split=True)
plt.title('violinplot')
plt.show()  # Show plot


"""### FacetGrid ###"""

# loading of a dataframe from seaborn
df = sns.load_dataset('tips')
# Form a facetgrid using columns with a hue
# col is the column of diagrams
graph = sns.FacetGrid(df, col ="sex",  hue ="day")
# map the above form facetgrid with some attributes
graph.map(plt.scatter, "total_bill", "tip", edgecolor ="w").add_legend()
plt.show()


df = sns.load_dataset('tips')
# Form a facetgrid using columns with a hue
graph = sns.FacetGrid(df, row ='smoker', col ='time')
# map the above form facetgrid with some attributes
graph.map(plt.hist, 'total_bill', bins = 15, color ='orange')
plt.show()


df = sns.load_dataset('tips')
# Form a facetgrid using columns with a hue
graph = sns.FacetGrid(df, col ='time', hue ='smoker')
# map the above form facetgrid with some attributes
graph.map(sns.regplot, "total_bill", "tip").add_legend()
plt.show()


"""### PairGrid ###"""
"""###  对角线图阵 ###"""
# all number column pairs are taken into account
# loading dataset 
df = sns.load_dataset('tips') 
  
# PairGrid object with hue 
graph = sns.PairGrid(df, hue ='day') 
# type of graph for diagonal 
graph = graph.map_diag(plt.hist) 
# type of graph for non-diagonal 
graph = graph.map_offdiag(plt.scatter) 
# to add legends 
graph = graph.add_legend() 
# to show 
plt.show() 
# This code is contributed by Deepanshu Rusatgi.
  
# loading dataset 
df = sns.load_dataset('tips') 
# PairGrid object with hue 
graph = sns.PairGrid(df) 
# type of graph for diagonal 
graph = graph.map_diag(sns.kdeplot, lw = 2) 
# type of graph for non-diagonal(upper part) 
graph = graph.map_upper(sns.scatterplot) 
# type of graph for non-diagonal(lower part) 
graph = graph.map_lower(sns.kdeplot) 
# to show 
plt.show() 
# This code is contributed by Deepanshu Rusatgi.
