#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


path = r'C:\Users\dato_\OneDrive\Desktop\PROJECT\individual_stocks_5yr-20211010T153422Z-001\individual_stocks_5yr'
company_list = ['AAPL_data.csv','GOOG_data.csv','MSFT_data.csv','AMZN_data.csv']
all_data = pd.DataFrame()
for file in company_list:
    current_df = pd.read_csv(path + '/' + file)
    all_data = pd.concat([all_data,current_df])
all_data.shape


# In[3]:


all_data.head()


# In[4]:


all_data.dtypes


# In[5]:


all_data['date'] = pd.to_datetime(all_data['date'])


# In[6]:


tech_list = all_data['Name'].unique()


# In[7]:


plt.figure(figsize=(20,12))
for i,company in enumerate (tech_list,1):
    plt.subplot(2,2,i)
    df = all_data[all_data['Name']==company]
    plt.plot(df['date'],df['close'])
    plt.xticks(rotation='vertical')
    plt.title(company)


# In[ ]:





# In[8]:


import plotly.express as px


# In[9]:


for company in tech_list:
    df = all_data[all_data['Name'] == company]
    fig = px.line(df,x = 'date', y = 'volume', title = company)
    fig.show()


# In[ ]:





# In[10]:


df = pd.read_csv(r'C:\Users\dato_\OneDrive\Desktop\PROJECT\individual_stocks_5yr-20211010T153422Z-001\individual_stocks_5yr\AAPL_data.csv')
df.head()


# In[11]:


df['Daily_Price_change'] = df['close'] - df['open']
df.head()


# In[12]:


df['1day % return'] = ((df['close']-df['open'])/df['close']) * 100


# In[13]:


df.head()


# In[14]:


fig = px.line(df,x = 'date', y = '1day % return', title = company)
fig.show()


# In[15]:


df2 = df.copy()


# In[16]:


df2.dtypes


# In[17]:


df2['date'] = pd.to_datetime(df2['date'])


# In[18]:


df2.set_index('date',inplace = True)


# In[19]:


df2.head()


# In[20]:


df2['2013-02-08':'2013-02-14']


# In[21]:


df2['close'].resample('M').mean().plot()


# In[22]:


df2['close'].resample('Y').mean().plot(kind = 'bar')


# In[23]:


aapl = pd.read_csv(r'C:\Users\dato_\OneDrive\Desktop\PROJECT\individual_stocks_5yr-20211010T153422Z-001\individual_stocks_5yr\AAPL_data.csv')
aapl.head()


# In[24]:


amzn = pd.read_csv(r'C:\Users\dato_\OneDrive\Desktop\PROJECT\individual_stocks_5yr-20211010T153422Z-001\individual_stocks_5yr\AMZN_data.csv')
amzn.head()


# In[37]:


msft = pd.read_csv(r'C:\Users\dato_\OneDrive\Desktop\PROJECT\individual_stocks_5yr-20211010T153422Z-001\individual_stocks_5yr\MSFT_data.csv')
msft.head()


# In[38]:


goog = pd.read_csv(r'C:\Users\dato_\OneDrive\Desktop\PROJECT\individual_stocks_5yr-20211010T153422Z-001\individual_stocks_5yr\GOOG_data.csv')
goog.head()


# In[ ]:





# In[39]:


close = pd.DataFrame()


# In[40]:


close['aapl']=aapl['close']
close['goog']=goog['close']
close['amzn']=amzn['close']
close['msft']=msft['close']


# In[41]:


close.head()


# In[42]:


sns.pairplot(data=close)


# In[43]:


sns.heatmap(close.corr(), annot=True)


# In[ ]:





# In[44]:


aapl.head()


# In[45]:


data = pd.DataFrame()


# In[51]:


data['aapl_change'] = ((aapl['close'] - aapl['open']) / aapl['close']) * 100
data['goog_change'] = ((goog['close'] - goog['open']) / goog['close']) * 100
data['amzn_change'] = ((amzn['close'] - amzn['open']) / amzn['close']) * 100
data['msft_change'] = ((msft['close'] - msft['open']) / msft['close']) * 100


# In[52]:


data.head()


# In[53]:


sns.pairplot(data = data)


# In[54]:


sns.heatmap(data.corr(), annot = True)


# In[56]:


sns.distplot(data['aapl_change'])


# In[57]:


data['aapl_change'].std()


# In[58]:


data['aapl_change'].std() * 2


# In[59]:


data['aapl_change'].std() * 3


# In[60]:


data['aapl_change'].quantile(0.1)


# In[61]:


data.describe().T


# In[ ]:




