#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()


# In[2]:


df = pd.read_csv(r'C:\Users\dato_\OneDrive\Desktop\PROJECT\zomato-bangalore-restourants/zomato.csv')
df.head()


# In[3]:


df.columns


# In[4]:


df.dtypes


# In[5]:


df.shape


# In[6]:


df.isnull().sum()


# In[7]:


feature_na = [feature for feature in df.columns if df[feature].isnull().sum() > 0]
feature_na


# In[8]:


for feature in feature_na:
    print(' {} has {} missing values'.format(feature,np.round(df[feature].isnull().sum()/len(df)*100,4)))


# In[ ]:





# In[9]:


df['rate'].unique()


# In[10]:


df.dropna(axis = 'index', subset = ['rate'],inplace = True)


# In[11]:


df.shape


# In[ ]:





# In[12]:


def split(x):
    return x.split('/')[0]


# In[13]:


df['rate'] = df['rate'].apply(split)


# In[14]:


df.head()


# In[15]:


df['rate'].unique()


# In[16]:


df.replace('NEW',0,inplace = True)


# In[17]:


df.replace('-',0,inplace = True)


# In[18]:


df['rate'].dtype


# In[19]:


df['rate'] = df['rate'].astype(float)


# In[20]:


df['rate'].dtype


# In[21]:


df.head()


# In[22]:


df_rate = df.groupby('name')['rate'].mean().to_frame().reset_index()
df_rate.columns = ['restaurant', 'avg_rating']
df_rate.head(20)


# In[23]:


sns.distplot(df_rate['avg_rating'])


# In[24]:


df_rate.shape


# In[25]:


chains = df['name'].value_counts()[0:20]
sns.barplot(x = chains, y = chains.index)
plt.title('Most famous Rest chains in Begalore')
plt.xlabel('Number of outlets')


# In[26]:


x = df['online_order'].value_counts()
x


# In[27]:


labels = ['accepted', 'Not accepted']


# In[28]:


import plotly.express as px


# In[29]:


px.pie(df, values = x, labels = labels, title = 'Pie Chart' )


# In[30]:


x = df['book_table'].value_counts()
x


# In[31]:


lables = ['not book', 'book']


# In[32]:


import plotly.graph_objs as go
from plotly.offline import iplot


# In[33]:


trace = go.Pie(labels = labels, values = x, hoverinfo = 'label + percent', textinfo = 'value')
iplot([trace])


# In[34]:


df['rest_type'].isna().sum()


# In[35]:


df['rest_type'].dropna(inplace=True)


# In[36]:


df['rest_type'].isna().sum()


# In[37]:


len(df['rest_type'].unique())


# In[38]:


trace1 = go.Bar(x = df['rest_type'].value_counts().nlargest(20).index,
y = df['rest_type'].value_counts().nlargest(20)
)


# In[39]:


iplot([trace1])


# In[40]:


df.groupby('name')['votes'].sum().nlargest(20).plot.bar()


# In[54]:


go.Bar(x = df.groupby('name')['votes'].sum().nlargest(20).index,
       y = df.groupby('name')['votes'].sum().nlargest(20))
iplot([trace1])


# In[55]:


restaurant = []
location = []
for key, location_df in df.groupby('location'):
    location.append(key)
    restaurant.append(len(location_df['name'].unique()))


# In[56]:


df_total = pd.DataFrame(zip(location,restaurant))
df_total.columns = ['location', 'restaurant']
df_total.head()


# In[57]:


df_total.set_index('location', inplace = True)
df_total.head()


# In[58]:


df_total.sort_values(by = 'restaurant').tail(10).plot.bar()


# In[59]:


cuisines = df['cuisines'].value_counts()[0:10]
cuisines


# In[60]:


trace1 = go.Bar(
x = df['cuisines'].value_counts()[0:10].index,
y = df['cuisines'].value_counts()[0:10]
)


# In[61]:


iplot([trace1])


# In[66]:


df.columns


# In[67]:


df['approx_cost(for two people)'].isna().sum()


# In[68]:


df.dropna(axis = 'index', subset = ['approx_cost(for two people)'],inplace = True)


# In[69]:


df['approx_cost(for two people)'].isna().sum()


# In[53]:


sns.distplot(df['approx_cost(for two people)'])


# In[80]:


df['approx_cost(for two people)'].dtype


# In[81]:


df['approx_cost(for two people)'].unique()


# In[82]:


df['approx_cost(for two people)'] = df['approx_cost(for two people)'].apply(lambda x: x.replace(',',''))


# In[83]:


df['approx_cost(for two people)'].unique()


# In[84]:


df['approx_cost(for two people)'] = df['approx_cost(for two people)'].astype(int)


# In[85]:


df['approx_cost(for two people)'].dtype


# In[86]:


sns.distplot(df['approx_cost(for two people)'])


# In[88]:


sns.scatterplot(x = 'rate', y = 'approx_cost(for two people)',hue = 'online_order',data = df)


# In[89]:


sns.boxplot(x = 'online_order', y = 'votes', data= df)


# In[91]:


px.box(df,x = 'online_order', y = 'votes')


# In[92]:


df.columns


# In[93]:


px.box(df,x = 'online_order', y = 'approx_cost(for two people)')


# In[94]:


df['approx_cost(for two people)'].min()


# In[95]:


df['approx_cost(for two people)'].max()


# In[96]:


df[df['approx_cost(for two people)'] == 6000]['name']


# In[97]:


data = df.copy()


# In[98]:


data.set_index('name',inplace= True)


# In[99]:


data.head()


# In[100]:


data['approx_cost(for two people)'].nlargest(10).plot.bar()


# In[101]:


data['approx_cost(for two people)'].nsmallest(10).plot.bar()


# In[102]:


data[data['approx_cost(for two people)'] <= 500]


# In[103]:


df_budget = data[data['approx_cost(for two people)'] <= 500].loc[:,('approx_cost(for two people)')]
df_budget = df_budget.reset_index()
df_budget.head()


# In[104]:


df[(df['rate'] > 4) & (df['approx_cost(for two people)'] <= 500)].shape


# In[105]:


len(df[(df['rate'] > 4) & (df['approx_cost(for two people)'] <= 500)]['name'].unique())


# In[106]:


df_new = df[(df['rate'] > 4) & (df['approx_cost(for two people)'] <= 500)]
df_new.head()


# In[107]:


location = []
total = []

for loc,location_df in df_new.groupby('location'):
    location.append(loc)
    total.append(len(location_df['name'].unique()))
    


# In[108]:


location_df = pd.DataFrame(zip(location,total))
location_df.head()


# In[109]:


location_df.columns = ['location','restaurant']
location_df.head()


# In[ ]:





# In[110]:


def return_budget(location,restaurant):
    budget=df[(df['approx_cost(for two people)']<=400) & (df['location']==location) & 
                     (df['rate']>4) & (df['rest_type']==restaurant)]
    return(budget['name'].unique())


# In[111]:


return_budget('BTM','Quick Bites')


# In[ ]:





# In[112]:


restaurant_location = df['location'].value_counts()[0:20]
sns.barplot(restaurant_location,restaurant_location.index)


# In[ ]:





# In[113]:


locations = pd.DataFrame({'Name':df['location'].unique()})
locations.head()


# In[114]:


from geopy.geocoders import Nominatim


# In[115]:


geolocator = Nominatim(user_agent = 'app')


# In[116]:


lat_lon = []

for location in locations['Name']:
    location=geolocator.geocode(location)
    if location is None:
        lat_lon.append(np.nan)
    else:
        geo = (location.latitude,location.longitude)
        lat_lon.append(geo)


# In[117]:


locations['geo_loc'] = lat_lon


# In[118]:


locations.head()


# In[119]:


locations.shape


# In[120]:


Rest_locations = pd.DataFrame(df['location'].value_counts().reset_index())
Rest_locations.head()


# In[121]:


Rest_locations.columns = ['Name','count']
Rest_locations.head()


# In[122]:


Restaurant_locations = Rest_locations.merge(locations,on='Name',how='left').dropna()
Restaurant_locations.head()


# In[123]:


np.array(Restaurant_locations['geo_loc'])


# In[124]:


lat,lon = zip(*np.array(Restaurant_locations['geo_loc']))


# In[125]:


Restaurant_locations['lat']=lat
Restaurant_locations['lon']=lon


# In[126]:


Restaurant_locations.head()


# In[127]:


Restaurant_locations.drop('geo_loc',axis=1,inplace=True)


# In[128]:


Restaurant_locations.head()


# In[129]:


import folium
from folium.plugins import HeatMap


# In[130]:


def generatebasemap(default_location = [12.97,77.59],default_zoom_start = 12):
    basemap = folium.Map(location = default_location, zoom_start = default_zoom_start)
    return basemap


# In[131]:


basemap = generatebasemap()


# In[132]:


basemap


# In[133]:


HeatMap(Restaurant_locations[['lat','lon','count']].values.tolist(),zoom = 20, radius = 15).add_to(basemap)


# In[134]:


basemap


# In[ ]:





# In[135]:


df.head()


# In[136]:


df2 = df[df['cuisines'] == 'North Indian']
df2.head()


# In[137]:


north_india = df2.groupby(['location'],as_index = False)['url'].agg('count')


# In[138]:


north_india.head()


# In[139]:


north_india.columns = ['Name','count']


# In[140]:


north_india.head()


# In[141]:


north_india = north_india.merge(locations,on='Name',how='left').dropna()
north_india.head(10)


# In[142]:


north_india['lat'],north_india['lon']=zip(*north_india['geo_loc'].values)
north_india.head()


# In[143]:


north_india.drop('geo_loc',axis=1,inplace = True)
north_india.head()


# In[144]:


basemap = generatebasemap()
HeatMap(north_india[['lat','lon','count']].values.tolist(),zoom = 20, radius = 15).add_to(basemap)
basemap


# In[ ]:





# In[146]:


df_1 = df.groupby(['rest_type','name']).agg('count')


# In[147]:


df_1


# In[148]:


df_1.sort_values(['url'],ascending = False)


# In[150]:


df_1.sort_values(['url'],ascending = False).groupby(['rest_type'],as_index = False).apply(lambda x: x.sort_values(by = 'url', ascending = False))


# In[153]:


df_1.sort_values(['url'],ascending = False).groupby(['rest_type'],as_index = False).apply(lambda x: x.sort_values(by = 'url', ascending = False))['url'].reset_index()


# In[155]:


dataset = df_1.sort_values(['url'],ascending = False).groupby(['rest_type'],as_index = False).apply(lambda x: x.sort_values(by = 'url', ascending = False))['url'].reset_index().rename(columns = {'url':'count'})
dataset


# In[157]:


casual = dataset[dataset['rest_type'] == 'Casual Dining']
casual


# In[ ]:




