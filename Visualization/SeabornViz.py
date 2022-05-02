import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("Visualization\dm_office_sales.csv")
#df.head()
#df.info()
#df.describe()

######### ---------Scatter Plots -------------- #########
# Continuous feaures. show how different features are related to one another
#display how features are interconnected to each other
plt.figure(figsize=(12,8),dpi=40) # dpi makes words clear
sns.scatterplot(x='salary',y='sales',data=df,
                hue='level of education',   #
                palette='Dark2',
                alpha=0.8, # transparency
                s=200, #size
                linewidth=0,
                style='level of education');#size='work experience'

######### ---- Distributions Plots -->#rugplot , displot (histplot),  kdeplot--------- #########
# 3 Distributions plot Types  --> 
#Continios feature to visualize its Deviation or Mean in BINS

 #rugplot # The y axis doesn't really represent anything
# X axis is just a stick per data point
#rugplot itself is not very informative for larger data sets distribution
sns.set(style='darkgrid')
sns.rugplot(x='salary',data=df,height=0.5)
sns.displot(data=df,x='salary',
            bins=20,
            kde=True,
            color='red',
            edgecolor='black',
            lw=4,ls='--'  )
sns.histplot(data=df,x='salary',bins=10, kde=True)
sns.kdeplot(data=df,x='salary',shade=True,color='red',bw_adjust=0.5)# adjust bandwidth(bw_adjust), how smooth kde curve

######### --------- Group by aggregation metric visualization -->#Countplot ot Barplot---------------- #########
sns.barplot(x='level of education',   ## shows UP  tics on top bars
            y='salary',data=df,
            estimator=np.mean,
            ci='sd', # standard deviation for above tick 
            hue='division')

sns.countplot(x='level of education', # shows the Value_count  for each Category of per column data
              data=df,
              hue='training level',
              palette='Set1')

######### --- Distribution per category --># Boxplt , Violinplot , Swarmplot, Boxenplot---------------- #########
#Boxplot (Wisker plot) .. Distribution of Continiuos data. shows based on Quartiles 25%,50% & 75%
sns.boxplot(x='math score',y='parental level of education',data=df,orient='h') #orient='h' makes horizontal
#Violinplot ... Add 2 sided KDS plot makes Violin + Boxplot in center
sns.violinplot(x='parental level of education',y='math score',data=df,hue='gender',split=True) #setting split to True will draw half of a violin for each level
#Swarmplot .. shows distribution of all data points per category
sns.swarmplot(x='race/ethnicity',y='math score',data=df,hue='gender')
### boxenplot (letter-value plot)
sns.boxenplot(x='race/ethnicity',y='math score',data=df,hue='gender')
#plt.savefig('example_scatter.jpg')

######### --------- Comparison Plots --> pairplot() and jointplot() ---------------- #########
# jointplot--> Map Histogram to each feature of a Scatterplot to clarify the distribution. Scatterplt can be made Hex or 2d KDE plot
sns.jointplot(x='math score',y='reading score',data=df,kind='hex') #kind='kde' makes 2d KDE or Hex plt
# pairplot-->CPU & RAM intensive(Filter before using). Compare all numarical column.Creates Histograms & Scatterplot for all combination columns
sns.pairplot(df,hue='gender',palette='viridis') #corner=True-> romoves duplicate graphs(compare itself),diag_kind='hist' makes Histograms

######### --------- Grids --># Catplot() & Pairgrid() & FacetGrid()---------------- #########
# catplot()--> makes rows & column by 'row' & 'Col' attrubutes.Rows & columns are made denepends of number of category
sns.catplot(x='gender',y='math score',data=df,kind='box',row='lunch',col='test preparation course')  #chnage kind to make any type of plot
#Pairgrid() --> Add Upper, Diag, Lower diffrent types of plots
g = sns.PairGrid(df, hue="gender", palette="viridis",hue_kws={"marker": ["o", "+"]})
g = g.map_upper(sns.scatterplot, linewidths=1, edgecolor="w", s=40)
g = g.map_diag(sns.distplot)
g = g.map_lower(sns.kdeplot)
g = g.add_legend();
# FacetGrid
g = sns.FacetGrid(data=df,col='gender',row='lunch')
g = g.map(plt.scatter, "math score", "reading score", edgecolor="w")
g.add_legend()

######### ---------  Matrix Plots -->#  Heatmap,Clustermap  ---------------- #########
## Heatmap .comapre co relation Column vs rows
sns.heatmap(rates,linewidth=0.5,annot=True,cmap='viridis',center=40)
#Clustermap . Cluster togethere rows and column .. dundgram
sns.clustermap(rates)#,col_cluster=False,figsize=(12,8),cbar_pos=(-0.1, .2, .03, .4))



























