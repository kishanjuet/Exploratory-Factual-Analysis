#!/usr/bin/env python
# coding: utf-8

# # Exploratory Factual Analysis

# By Kishan Gupta

# # Intern Dataset

# In[32]:


import os
import pandas as pd
import seaborn as sns
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np


# In[20]:


df=pd.read_csv('intern_dataset.csv')


# In[21]:


df.head()


# In[22]:


print(df.shape)


# In[23]:


print(df.columns)


# In[24]:


df['Label'].value_counts()


# # 2-D Scatter Plot

# In[25]:


df.plot(kind='scatter',x='Signal1',y='Signal2')
plt.show()


# In[26]:


sns.set_style("whitegrid");
sns.FacetGrid(df, hue="Label",height=4)    .map(plt.scatter, "Signal1", "Signal2")    .add_legend();
plt.show();


# Obervations:
# 
# 1. Using Signal1 and Signal2 feature,we can distinguish Label 'C' from others.
# 2. Seperating Label 'A' and 'B' is much harder as they have considerable overlap.
# 

# # Pair-Plot

# In[50]:


plt.close();
sns.set_style("whitegrid");
sns.pairplot(df,hue='Label',height=3);
plt.show()


# Observations:
# 1. Signal2 is most useful feature to identify various Label types.
# 2. While Label 'C' can be easily separable, Labels 'A' and 'B' have some overlap
# 3. We can find "lines" and "if-else" conditions to build a simple model to classify the Label Type

# # Histogram plot

# In[58]:


sns.FacetGrid(df,hue='Label',height=5)    .map(sns.histplot,"Signal2")    .add_legend();
plt.show()


# In[60]:


sns.FacetGrid(df,hue='Label',height=5)    .map(sns.histplot,"Signal1")    .add_legend();
plt.show()


# # Cumulative Distribution Function(CDF)

# In[63]:


df_A=df.loc[df["Label"]=="A"]
df_B=df.loc[df["Label"]=="B"]
df_C=df.loc[df["Label"]=="C"]


# PDF and CDF for Label A

# In[75]:


counts, bin_edges= np.histogram(df_A["Signal2"], bins=10,
                              density=True)
#compute PDF
pdf=counts/(sum(counts))
print(pdf)
print(bin_edges)

#compute CDF
cdf=np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf)
plt.plot(bin_edges[1:],cdf)

plt.show()


# PDF and CDF for Label C

# In[76]:


counts, bin_edges= np.histogram(df_C["Signal2"], bins=10,
                                density=True)
#compute PDF
pdf=counts/(sum(counts))
print(pdf)
print(bin_edges)

#compute CDF
cdf=np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf)
plt.plot(bin_edges[1:],cdf)

plt.show()


# PDF and CDF for Label B

# In[77]:


counts, bin_edges= np.histogram(df_B["Signal2"], bins=10,
                                density=True)
#compute PDF
pdf=counts/(sum(counts))
print(pdf)
print(bin_edges)

#compute CDF
cdf=np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf)
plt.plot(bin_edges[1:],cdf)

plt.show()


# In[78]:


counts, bin_edges= np.histogram(df_A["Signal2"], bins=10,
                                density=True)
#compute PDF
pdf=counts/(sum(counts))
print(pdf)
print(bin_edges)

#compute CDF
cdf=np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf)
plt.plot(bin_edges[1:],cdf)


counts, bin_edges= np.histogram(df_C["Signal2"], bins=10,
                              density=True)
#compute PDF
pdf=counts/(sum(counts))
print(pdf)
print(bin_edges)

#compute CDF
cdf=np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf)
plt.plot(bin_edges[1:],cdf)


counts, bin_edges= np.histogram(df_B["Signal2"], bins=10,
                                density=True)
#compute PDF
pdf=counts/(sum(counts))
print(pdf)
print(bin_edges)

#compute CDF
cdf=np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf)
plt.plot(bin_edges[1:],cdf)

plt.show()


# # Mean , Variance and Std-Dev

# In[79]:


print("Mean:")
print(np.mean(df_A["Signal2"]))
#mean with an outlier.
print(np.mean(np.append(df_A["Signal2"],912000)))
print(np.mean(df_C["Signal2"]))
print(np.mean(df_B["Signal2"]))

print("\nStd-dev:")
print(np.std(df_A["Signal2"]))
print(np.std(df_C["Signal2"]))
print(np.std(df_B["Signal2"]))


# # Median, Percentile, Quantile, IQR, MAD

# In[90]:


print("\nMedians:")
print(np.median(df_A["Signal2"]),"of A")
#Median with an outlier
print("Median with an Outlier",np.median(np.append(df_A["Signal2"],912000)),"of A")
print(np.median(df_C["Signal2"]),"of C")
print(np.median(df_B["Signal2"]),"of B")

print("\nQuantiles:")
print(np.percentile(df_A["Signal2"],np.arange(0,100,25)),"of A")
print(np.percentile(df_C["Signal2"],np.arange(0,100,25)),"of C")
print(np.percentile(df_B["Signal2"],np.arange(0,100,25)),"of B")

print("\n90th Percentiles:")
print(np.percentile(df_A["Signal2"],90),"of A")
print(np.percentile(df_C["Signal2"],90),"of C")
print(np.percentile(df_B["Signal2"],90),"of B")

from statsmodels import robust
print("\nMedian Absolute Deviation:")
print(robust.mad(df_A["Signal2"]),"of A")
print(robust.mad(df_C["Signal2"]),"of C")
print(robust.mad(df_B["Signal2"]),"of B")

print("\nIQR:")
print(np.percentile(df_A["Signal2"],75)-np.percentile(df_A["Signal2"],25),"of A")
print(np.percentile(df_C["Signal2"],75)-np.percentile(df_C["Signal2"],25),"of C")
print(np.percentile(df_B["Signal2"],75)-np.percentile(df_B["Signal2"],25),"of B")


# # Box Plot and Whiskers

# In[92]:


sns.boxplot(x="Label",y="Signal1",data=df)
plt.show()


# In[93]:


sns.boxplot(x="Label",y="Signal2",data=df)
plt.show()


# # Violin Plots

# In[94]:


sns.violinplot(x="Label",y="Signal1",data=df,size=8)
plt.show()


# In[95]:


sns.violinplot(x="Label",y="Signal2",data=df,size=8)
plt.show()


# # Multivariate probability density and contour plot

# In[96]:


sns.jointplot(x="Signal1", y="Signal2", data=df_A, kind="kde")
plt.show()


# Observations:
# 1. In this 2d plot lower layer indicates more points and light layers or hills is called less points.
# 2. These lighter to denser lines is called contours. This graph is called Contours probability density plot.

# # Summarizing EDA

# # UniVarient Analysis:

# PDF/Histograms
# 
# CDF
# 
# Box Plot
# 
# Violin Plot
# 

# # BiVarient Analysis:

# Scatter Plot
# 
# Pair Plot

# # MultiVariant Analysis:

# We can use 3D plot.

# But since there are two feature Signal1 and Signal2 we can't plot 3-D Diagram, to make 3D plot we required one more dimension or feature
