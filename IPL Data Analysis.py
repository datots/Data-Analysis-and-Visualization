#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


df = pd.read_csv(r'C:\Users\dato_\OneDrive\Desktop\PROJECT\IPL data Analysis/deliveries.csv')
df.head()


# In[3]:


filt = df['batsman'] == 'DA Warner'
df_warner = df[filt]
df_warner.shape


# In[4]:


df_warner['dismissal_kind'].value_counts().plot.pie()


# In[5]:


len(df_warner[df_warner['batsman_runs'] == 4])


# In[6]:


401 * 4


# In[7]:


len(df_warner[df_warner['batsman_runs'] == 6])


# In[8]:


def count(df, runs):
    return len(df_warner[df_warner['batsman_runs'] == runs]) * runs
    


# In[9]:


count(df_warner, 1)


# In[10]:


count(df_warner, 2)


# In[11]:


count(df_warner, 3)


# In[12]:


count(df_warner, 4)


# In[13]:


count(df_warner, 6)


# In[14]:


slices = [997, 414, 39, 1604, 960]
labels = [1,2,3,4,6]
explode = [0,0,0,0.1,0]
plt.pie(slices,labels = labels, autopct = '%1.1f%%', explode = explode)


# In[15]:


df.head()


# In[16]:


df['bowling_team'].unique()


# In[17]:


Teams = {
      'Royal Challengers Bangalore':'RCB',
      'Sunrisers Hyderabad':'SRH',
      'Rising Pune Supergiant':'RPS',
      'Mumbai Indians':'MI',
      'Kolkata Knight Riders':'KKR',
      'Gujarat Lions':'GL',
      'Kings XI Punjab':'KXIP',
      'Delhi Daredevils':'DD', 
      'Chennai Super Kings':'CSK', 
      'Rajasthan Royals':'RR',
      'Deccan Chargers':'DC',
      'Kochi Tuskers Kerala':'KTK',
      'Pune Warriors':'PW',
      'Rising Pune Supergiants':'RPS'
}


# In[18]:


df['batting_team']= df['batting_team'].map(Teams)
df['bowling_team'] = df['bowling_team'].map(Teams)


# In[19]:


df.head()


# In[20]:


df.columns


# In[21]:


runs = df.groupby(['match_id', 'inning', 'batting_team'])['total_runs'].sum().reset_index()
runs.drop('match_id', axis = 1, inplace = True)
runs


# In[22]:


inning1 = runs[runs['inning'] == 1]
inning2 = runs[runs['inning'] == 2]


# In[23]:


sns.boxplot(x = 'batting_team', y = 'total_runs', data = inning1)


# In[24]:


sns.boxplot(x = 'batting_team', y = 'total_runs', data = inning2)


# In[ ]:





# In[25]:


high_scores = df.groupby(['match_id','inning','batting_team','bowling_team'])['total_runs'].sum().reset_index()
high_scores


# In[26]:


score_200 = high_scores[high_scores['total_runs'] >= 200]
score_200


# In[27]:


sns.countplot(score_200['batting_team'])


# In[28]:


sns.countplot(score_200['bowling_team'])


# In[29]:


balls = df.groupby('batsman')['ball'].count().reset_index()
balls


# In[30]:


runs = df.groupby('batsman')['batsman_runs'].sum().reset_index()
runs


# In[31]:


four = df[df['batsman_runs'] == 4]
four


# In[32]:


runs_4 = four.groupby('batsman')['batsman_runs'].count().reset_index()
runs_4.columns = ['batsman', '4s']
runs_4


# In[33]:


six = df[df['batsman_runs'] == 6]
six


# In[34]:


six = four.groupby('batsman')['batsman_runs'].count().reset_index()
six.columns = ['batsman', '6s']
six


# In[35]:


player = pd.concat([runs,balls.iloc[:,1],runs_4.iloc[:,1],six.iloc[:,1]], axis = 1)
player


# In[36]:


player.fillna(0, inplace = True)


# In[37]:


player


# In[38]:


player['strike _rate'] = (player['batsman_runs'] / player['ball']) * 100


# In[39]:


player


# In[40]:


grp = df.groupby(['match_id', 'batsman', 'batting_team'])['batsman_runs'].sum().reset_index()
grp


# In[41]:


max = grp.groupby('batsman')['batsman_runs'].max().reset_index()
max.columns = ['batsman','max_runs']
max


# In[42]:


player2 = pd.concat([player,max.iloc[:,1]], axis = 1)
player2


# In[43]:


max_runs = df.groupby('batsman')['batsman_runs'].sum()
max_runs.sort_values(ascending = False)[:10].plot(kind = 'bar')


# In[44]:


df.groupby(['match_id', 'batsman', 'batting_team'])['batsman_runs'].sum().reset_index().sort_values(by = 'batsman_runs', ascending = False).head(10)


# In[45]:


df['dismissal_kind'].unique()


# In[46]:


dismissal_kinds = ['caught', 'bowled', 'run out', 'lbw', 'caught and bowled',
       'stumped','hit wicket']


# In[47]:


hwt = df[df['dismissal_kind'].isin(dismissal_kinds)]
hwt.head()


# In[48]:


hwt['bowler'].value_counts()[:10].plot(kind = 'bar')


# In[49]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()


# In[50]:


df = pd.read_csv(r'C:\Users\dato_\OneDrive\Desktop\PROJECT\IPL data Analysis/matches.csv')
df


# In[51]:


df.isnull().sum()


# In[52]:


df.shape


# In[53]:


df.drop(['umpire3'], axis = 1, inplace = True)


# In[54]:


df.columns


# In[55]:


Teams = {
      'Royal Challengers Bangalore':'RCB',
      'Sunrisers Hyderabad':'SRH',
      'Rising Pune Supergiant':'RPS',
      'Mumbai Indians':'MI',
      'Kolkata Knight Riders':'KKR',
      'Gujarat Lions':'GL',
      'Kings XI Punjab':'KXIP',
      'Delhi Daredevils':'DD', 
      'Chennai Super Kings':'CSK', 
      'Rajasthan Royals':'RR',
      'Deccan Chargers':'DC',
      'Kochi Tuskers Kerala':'KTK',
      'Pune Warriors':'PW',
      'Rising Pune Supergiants':'RPS'
}


# In[56]:


df['team1'] = df['team1'].map(Teams)
df['team2'] = df['team2'].map(Teams)


# In[57]:


df.head()


# In[58]:


df.shape[0]


# In[59]:


len(df['city'].unique())


# In[60]:


len(df['team1'].unique())


# In[61]:


df['player_of_match'].value_counts()


# In[62]:


filter = df['win_by_runs'].max()
df[df['win_by_runs'] == filter]


# In[63]:


filter = df['win_by_wickets'].max()
df[df['win_by_wickets'] == filter]


# In[64]:


sns.countplot(x = 'season', hue = 'toss_decision', data = df)


# In[65]:


df['toss_winner'].value_counts().plot(kind = 'bar')


# In[66]:


df.head()


# In[67]:


df['team1'].value_counts()


# In[68]:


teams = (df['team1'].value_counts() + df['team2'].value_counts()).reset_index()
teams.columns = ['team_name','matches_played']
teams


# In[69]:


df['winner'] = df['winner'].map(Teams)


# In[70]:


wins = df['winner'].value_counts().reset_index()
wins.columns = ['team_name', 'wins']
wins


# In[71]:


player = teams.merge(wins, left_on = 'team_name',right_on = 'team_name', how = 'inner')
player


# In[72]:


player['%win'] = (player['wins'] / player['matches_played']) * 100
player


# In[73]:


import plotly.offline as py
import plotly.graph_objs as go


# In[74]:


trace1 = go.Bar(
x = player['team_name'],
y = player['matches_played'],
name = 'Total_Matches'
)

trace2 = go.Bar(
x = player['team_name'],
y = player['wins'],
name = 'Total_Matches'
)


# In[75]:


data = [trace1,trace2]
py.iplot(data)


# In[76]:


sns.countplot(df['season'])


# In[77]:


df.head()


# In[88]:


df2 = pd.read_csv(r'C:\Users\dato_\OneDrive\Desktop\PROJECT\IPL data Analysis/deliveries.csv')
df2.head()


# In[98]:


season = df[['id','season']].merge(df2,left_on='id',right_on='match_id',how='left').drop('id',axis=1)
season


# In[99]:


season = season.groupby('season')['total_runs'].sum().reset_index()
season.set_index('season',inplace=True)


# In[101]:


season.plot()


# In[104]:


avg_runs = df.groupby('season')['id'].count().reset_index().rename(columns={'id':'matches'})
avg_runs


# In[105]:


season


# In[108]:


season.reset_index(inplace=True)


# In[109]:


season=pd.concat([avg_runs,season.iloc[:,1]],axis=1)
season


# In[112]:


season['per_match_runs'] = season['total_runs']/season['matches']


# In[114]:


season.set_index('season')['per_match_runs'].plot()


# In[122]:


def lucky(df,team_name):
    return df[df['winner']==team_name]['venue'].value_counts().nlargest(5)


# In[124]:


lucky(df,'MI').plot(kind = 'bar')


# In[137]:


def comparison(team1,team2):
    compare = df[((df['team1'] == team1)|(df['team1'] == team2)|(df['team1']==team2))]
    sns.countplot(x = 'season', hue = 'winner', data = compare)


# In[138]:


comparison('MI', 'CSK')


# In[ ]:




