import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D

df = pd.read_csv("Visualization\dm_office_sales.csv")
df.head()


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







































