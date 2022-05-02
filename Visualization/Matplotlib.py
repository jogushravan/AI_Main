import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

a = np.arange(0,10)
b = 2*a*np.sin(a)
plt.plot ( a, b, 'ro--') #--> Basic plot, Can set Line style 'ro--'
#plt.fill_between(x,y,color='blue', alpha=0.2) --> fill teh gradient color
#fig, axes = plt.subplots(nrows=4, ncols=4)# plt.subplots(3,4) creates 3 rows & 4 columns grid
#fig, axes = plt.subplots() # returns Figure & Axes

fig = plt.figure()
names = ['A', 'B', 'C']
values = [19, 50, 29]
values_2 = [48, 19, 41]
plt.axes(0,0,1,1)
ax = fig.add_subplot(121)#1 row 2 column. subplot on position 1
ax2 = fig.add_subplot(122)#1 row 2 column. subplot on position 2
                        #add_subplot(232) means 2 rows 3 columns. subplot on position 2
                        
fig.suptitle('Plot Title')# set fugure level title
ax.set_title('Axis1 Title')# set Axis title
ax.set_xlabel('x_label')# set xlabel
ax.set_ylabel('y_label')# set ylabel
ax.set_xticks([0,1,2])# Set X ticks
ax.set_xticklabels(['zero','One','Two']) # Set X ticks lables
ax.set_yticks([-1,0,1])# Set y ticks

fig.legend(['Data'], loc="upper right") # set Legend

ax.bar(names, values,color='goldenrod') # chnages color
ax2.bar(names, values_2)
fig.tight_layout()
#plt.subplots_adjust(0.25, 0.35, 0.90, 0.8) # chnages layout

values = [15, 35, 5, 45]
labels = 'Oranges', 'Apples', 'Pears', 'Strawberries'
colors = {'r', 'g', 'b', 'r'}
explode = [0, 0, 0.2, 0]
plt.pie(values, labels=labels, colors=colors, explode=explode)# explode a piece of pie


#Histogram --  3 different types plot
 #barstacked
 #step
 #stepfilled
plt.hist(values, histtype='step') #histtype='stepfilled' ,Default Bar
plt.plot(values, color='r', linewitdh=2.0)# chnages Line color and line width


#Scatter Plot.  relationship between a categorical and a continuous variable
plt.scatter(x=values, y=values_2, color="darkslategrey", edgecolors="white", linewidths=0.1, alpha=0.7);

#3D Plots. o import the Axes3D module from mpl_tookits.mplot3d.
# for 3d plot add X, Y and Z values

from mpl_toolkits.mplot3d import Axes3D
file = pd.read_csv("vgsales.csv")
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
x = file['NA_Sales']
y = file['EU_Sales']
z = file['Other_Sales']
ax.scatter(x, y, z, c='r', s=20)
plt.xticks(rotation=60)
plt.show()















plt.show()
