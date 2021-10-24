#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

sns.set()


# In[2]:


files = [file for file in os.listdir(r'C:\Users\dato_\OneDrive\Desktop\PROJECT\Sales_Data')]
for file in files:
         print(file)


# In[3]:


path = (r'C:\Users\dato_\OneDrive\Desktop\PROJECT\Sales_Data')
all_data = pd.DataFrame()

for file in files:
    current_df = pd.read_csv(path + "/"+file)
    all_data = pd.concat([all_data, current_df])
    
all_data.shape


# In[4]:


all_data.to_csv(r'C:\Users\dato_\OneDrive\Desktop\PROJECT\Sales_Data/all_data.csv', index = False)


# In[5]:


all_data.head()


# In[6]:


all_data.isnull().sum()


# In[7]:


all_data = all_data.dropna(how = 'all')
all_data.shape


# In[8]:


'04/19/19 08:46'.split('/')[0]


# In[9]:


def month(x):
    return x.split('/')[0]


# In[10]:


all_data['month']=all_data['Order Date'].apply(month)


# In[11]:


all_data.dtypes


# In[12]:


all_data.head()


# In[13]:


all_data.dtypes


# In[14]:


#all_data['month'] = all_data['month'].astype(int)


# In[15]:


all_data['month'].unique()


# In[16]:


filter = all_data['month'] == 'Order Date'
all_data = all_data[~filter]
all_data.head()


# In[17]:


all_data['month'] = all_data['month'].astype(int)


# In[18]:


all_data.dtypes


# In[19]:


all_data['Quantity Ordered'] = all_data['Quantity Ordered'].astype(int)
all_data['Price Each'] = all_data['Price Each'].astype(float)


# In[20]:


all_data.dtypes


# In[21]:


all_data['sales'] = all_data['Quantity Ordered'] * all_data['Price Each']


# In[22]:


all_data.head()


# In[23]:


all_data.groupby('month')['sales'].sum()


# In[24]:


months = range(1, 13)
plt.bar(months,all_data.groupby('month')['sales'].sum())
plt.xticks(months)
plt.xlabel('month')
plt.ylabel('Sales in USD')


# In[25]:


all_data.head()


# In[26]:


'917 1st St, Dallas, TX 75001'.split(',')[1]


# In[27]:


def city(x):
    return x.split(',')[1]


# In[28]:


all_data['city'] = all_data['Purchase Address'].apply(city)


# In[29]:


all_data.head()


# In[30]:


all_data.groupby('city')['city'].count().plot.bar()


# In[31]:


all_data['Order Date'][0].dtype


# In[32]:


all_data['Hour'] = pd.to_datetime(all_data['Order Date']).dt.hour


# In[33]:


all_data.head()


# In[34]:


keys=[]
hour=[]
for key,hour_df in all_data.groupby('Hour'):
    keys.append(key)
    hour.append(len(hour_df))


# In[35]:


keys


# In[36]:


hour


# In[40]:


plt.plot(keys,hour)


# In[43]:


all_data.groupby('Product')['Quantity Ordered'].sum().plot(kind = 'bar')


# In[44]:


all_data.groupby('Product')['Price Each'].mean()


# In[46]:


products = all_data.groupby('Product')['Quantity Ordered'].sum().index
quantity = all_data.groupby('Product')['Quantity Ordered'].sum()
price = all_data.groupby('Product')['Price Each'].mean()


# In[50]:


fig, ax1 = plt.subplots()
ax2 = ax1.twinx()
ax1.bar(products, quantity, color = 'g')
ax2.plot(products, price)
ax1.set_xticklabels(products, rotation = 'vertical', size = 8)


# In[51]:


all_data.head()


# In[61]:


df = all_data['Order ID'].duplicated(keep = False)
df2 = all_data[df]
df2.head()


# In[64]:


df2['Grouped'] = df2.groupby('Order ID')['Product'].transform(lambda x: ','.join(x))


# In[65]:


df2.head()


# In[66]:


df2 = df2.drop_duplicates(subset = ['Order ID'])
df2.head()


# In[70]:


df2['Grouped'].value_counts()[0:5].plot.pie()


# In[ ]:




