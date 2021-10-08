#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()


# In[2]:


comments = pd.read_csv('GBcomments.csv', error_bad_lines=False)


# In[3]:


comments.head()


# In[4]:


from textblob import TextBlob


# In[5]:


TextBlob('Its more accurate to call it the M+ (1000) be..').sentiment.polarity


# In[6]:


comments.isna().sum()


# In[7]:


comments.dropna(inplace=True)


# In[8]:


polarity=[]
for i in comments['comment_text']:
    polarity.append(TextBlob(i).sentiment.polarity)


# In[9]:


comments['polarity']=polarity


# In[10]:


comments.head(20)


# In[11]:


comments['polarity'] = polarity


# In[12]:


comments.head(20)


# In[ ]:





# In[13]:


comments_positive = comments[comments['polarity'] == 1]


# In[14]:


comments_positive.shape


# In[15]:


comments_positive.head()


# In[ ]:





# In[16]:


from wordcloud import WordCloud,STOPWORDS


# In[17]:


stopwords = set(STOPWORDS)


# In[18]:


total_comments = ' '.join(comments_positive['comment_text'])


# In[19]:


wordcloud = WordCloud(width = 1000, height = 500, stopwords = stopwords).generate(total_comments)


# In[20]:


plt.figure(figsize = (15,5))
plt.imshow(wordcloud)
plt.axis('off')


# In[21]:


comments_negative = comments[comments['polarity'] == -1]


# In[22]:


total_comments = ' '.join(comments_negative['comment_text'])


# In[23]:


wordcloud = WordCloud(width = 1000, height = 500, stopwords = stopwords).generate(total_comments)


# In[24]:


plt.figure(figsize = (15,5))
plt.imshow(wordcloud)
plt.axis('off')


# In[ ]:





# In[25]:


videos = pd.read_csv('USvideos.csv', error_bad_lines = False)


# In[26]:


videos.head()


# In[27]:


tags_complete = ' '.join(videos['tags'])


# In[28]:


tags_complete


# In[29]:


import re


# In[30]:


tags = re.sub('[^a-zA-Z]',' ',tags_complete)


# In[31]:


tags


# In[32]:


tags = re.sub(' +',' ',tags)


# In[33]:


wordcloud = WordCloud(width = 1000, height = 500,stopwords=set(STOPWORDS)).generate(tags)


# In[34]:


plt.figure(figsize=(15,5))
plt.imshow(wordcloud)
plt.axis('off')


# In[ ]:





# In[35]:


sns.regplot(data = videos, x = 'views', y = 'likes')
plt.title('Regression plot for views and likes')


# In[36]:


sns.regplot(data = videos, x = 'views', y = 'dislikes')
plt.title('Regression plot for views and dislikes')


# In[ ]:





# In[37]:


df_corr = videos[['views','likes','dislikes']]


# In[38]:


df_corr.corr()


# In[39]:


sns.heatmap(df_corr.corr(), annot = True)


# In[ ]:





# In[40]:


comments.head()


# In[41]:


comments['comment_text'][1]


# In[42]:


print('\U0001F600')


# In[43]:


from emoji import emojize


# In[44]:


len(comments)


# In[45]:


comment = comments['comment_text'][1]


# In[47]:


import emoji


# In[48]:


[c for c in comment if c in emoji.UNICODE_EMOJI_ENGLISH]


# In[49]:


str = ' '
for i in comments['comment_text']:
    list = [c for c in comment if c in emoji.UNICODE_EMOJI_ENGLISH]
    for ele in list:
        str = str + ele


# In[50]:


len(str)


# In[51]:


str


# In[ ]:





# In[52]:


result = {}
for i in set(str):
    result[i] = str.count(i)
    


# In[53]:


result


# In[54]:


result.items()


# In[55]:


final = {}
for key, value in sorted(result.items(),key = lambda item:item[1]):
    final[key] = value


# In[56]:


final


# In[57]:


keys = [*final.keys()]


# In[58]:


keys


# In[59]:


values = [*final.values()]


# In[60]:


values


# In[ ]:





# In[61]:


df=pd.DataFrame({'chars':keys[-20:],'num':values[-20:]})


# In[62]:


df


# In[63]:


import plotly.graph_objs as go
from plotly.offline import iplot


# In[64]:


trace = go.Bar(
x = df['chars'],
y = df['num']
)
iplot([trace])


# In[ ]:




