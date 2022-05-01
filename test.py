import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("Visualization\dm_office_sales.csv")
df.head()




plt.figure(figsize=(10,4),dpi=200)
# Paired would be a good choice if there was a distinct jump from 0 and 1 to 2 and 3
#sns.countplot(x='level of education',data=df,hue='training level',palette='Paired')
sns.barplot(x='level of education',data=df,y='salary',hue='training level',palette='Paired', estimator=np.mean)

sns.countplot(x='level of education',data=df,hue='training level',palette='Set1')



plt.show()







































