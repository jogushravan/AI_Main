import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()
df = pd.read_csv(r'C:\Users\Diya\Downloads\ML by JosePortilla\05-Seaborn\dm_office_sales.csv')
print(df.head())

plt.figure(figsize=(12,8),dpi=40)
#plt.hist(df['salary'])
#plt.hist(df['sales'])
#sns.histplot(df['sales'])
sns.kdeplot(df['sales'],shade=True)
plt.show()

































