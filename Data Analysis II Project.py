#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np


# In[2]:


np.set_printoptions(suppress = True, linewidth = 100, precision = 2)


# In[3]:


raw_data_np = np.genfromtxt("loan-data.csv", delimiter = ';',
                             skip_header = 1,
                             autostrip = True)
raw_data_np


# In[4]:


np.isnan(raw_data_np).sum()


# In[5]:


temporary_fill = np.nanmax(raw_data_np) + 1
temporary_mean = np.nanmean(raw_data_np, axis = 0)


# In[6]:


temporary_mean


# In[7]:


temporary_stats = np.array([np.nanmin(raw_data_np, axis = 0),
                            temporary_mean,
                            np.nanmax(raw_data_np, axis = 0)])


# In[8]:


temporary_stats


# In[9]:


columns_strings = np.argwhere(np.isnan(temporary_mean)).squeeze()
columns_strings


# In[10]:


columns_numeric = np.argwhere(np.isnan(temporary_mean) == False).squeeze()
columns_numeric


# In[11]:


loan_data_strings = np.genfromtxt("loan-data.csv",
                                   delimiter = ';',
                                   skip_header = 1,
                                   autostrip = True,
                                   usecols = columns_strings,
                                   dtype = np.str)
loan_data_strings


# In[12]:


loan_data_numeric = np.genfromtxt("loan-data.csv",
                                   delimiter = ';',
                                   skip_header = 1,
                                   autostrip = True,
                                   usecols = columns_numeric,
                                   filling_values = temporary_fill)
loan_data_numeric


# In[13]:


header_full = np.genfromtxt("loan-data.csv",
                             delimiter = ';',
                             autostrip = True,
                             skip_footer = raw_data_np.shape[0],
                             dtype = np.str)

header_full        


# In[14]:


header_strings, header_numeric = header_full[columns_strings], header_full[columns_numeric]


# In[15]:


header_strings


# In[16]:


header_numeric


# In[17]:


def checkpoint(file_name, checkpoint_header, checkpoint_data):
    np.savez(file_name, header = checkpoint_header, data = checkpoint_data)
    checkpoint_variable = np.load(file_name + ".npz")
    return(checkpoint_variable)


# In[18]:


checkpoint_test = checkpoint("checkpoint-test", header_strings, loan_data_strings)


# In[19]:


checkpoint_test['data']


# In[20]:


np.array_equal(checkpoint_test['data'], loan_data_strings)


# In[21]:


header_strings


# In[22]:


header_strings[0] = "issue_date"


# In[23]:


loan_data_strings


# In[24]:


np.unique(loan_data_strings[:,0])


# In[25]:


loan_data_strings[:,0] = np.chararray.strip(loan_data_strings[:,0], "-15")


# In[26]:


loan_data_strings[:,0]


# In[27]:


months = np.array(['','Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])


# In[28]:


for i in range(13):
    loan_data_strings[:,0] = np.where(loan_data_strings[:,0] == months[i],
                                      i,
                                      loan_data_strings[:,0])


# In[29]:


np.unique(loan_data_strings[:,0])


# In[30]:


header_strings


# In[31]:


np.unique(loan_data_strings[:,1])


# In[32]:


np.unique(loan_data_strings[:,1]).size


# In[33]:


status_bad = np.array(['','Charged Off', 'Default', 'Late (31-120 days)'])


# In[34]:


loan_data_strings[:,1] = np.where(np.isin(loan_data_strings[:,1], status_bad),0,1)


# In[35]:


np.unique(loan_data_strings[:,1])


# In[36]:


header_strings


# In[37]:


np.unique(loan_data_strings[:,2])


# In[38]:


loan_data_strings[:,2] = np.chararray.strip(loan_data_strings[:,2], "months")
loan_data_strings[:,2]


# In[39]:


header_strings[2] = "term_months"


# In[40]:


loan_data_strings[:,2] = np.where(loan_data_strings[:,2] == '',
                                  '60',
                                  loan_data_strings[:,2])
loan_data_strings[:,2]


# In[41]:


np.unique(loan_data_strings[:,2])


# In[42]:


header_strings


# In[43]:


np.unique(loan_data_strings[:,3])


# In[44]:


np.unique(loan_data_strings[:,4])


# In[45]:


for i in np.unique(loan_data_strings[:,3])[1:]:
    loan_data_strings[:,4] = np.where((loan_data_strings[:,4] =='') & (loan_data_strings[:,3] == i),
    i + '5',
    loan_data_strings[:,4])


# In[46]:


np.unique(loan_data_strings[:,4], return_counts = True)


# In[47]:


loan_data_strings[:,4] = np.where(loan_data_strings[:,4] == '',
                                   'H1',
                                    loan_data_strings[:,4])


# In[48]:


np.unique(loan_data_strings[:,4])


# In[49]:


loan_data_strings = np.delete(loan_data_strings, 3, axis = 1)


# In[50]:


loan_data_strings[:,3]


# In[51]:


header_strings = np.delete(header_strings, 3)


# In[52]:


header_strings[3]


# In[53]:


np.unique(loan_data_strings[:,3])


# In[54]:


keys = list(np.unique(loan_data_strings[:,3]))
values = list(range(1, np.unique(loan_data_strings[:,3]).shape[0] + 1))
dict_sub_grade = dict(zip(keys, values))


# In[55]:


dict_sub_grade


# In[56]:


for i in np.unique(loan_data_strings[:,3]):
    loan_data_strings[:,3] = np.where(loan_data_strings[:,3] == i,
                                      dict_sub_grade[i],
                                      loan_data_strings[:,3])


# In[57]:


np.unique(loan_data_strings[:,3])


# In[58]:


header_strings


# In[59]:


np.unique(loan_data_strings[:,4])


# In[60]:


loan_data_strings[:,4] = np.where((loan_data_strings[:,4] == '') | (loan_data_strings[:,4] == 'Not Verified'),0,1)


# In[61]:


np.unique(loan_data_strings[:,4])


# In[62]:


loan_data_strings[:,5]


# In[63]:


np.chararray.strip(loan_data_strings[:,5], "https://www.lendingclub.com/browse/loanDetail.action?loan_id")


# In[64]:


loan_data_strings[:,5] = np.chararray.strip(loan_data_strings[:,5], "https://www.lendingclub.com/browse/loanDetail.action?loan_id=")


# In[65]:


header_full


# In[66]:


loan_data_numeric[:,0].astype(dtype = np.int32)


# In[67]:


loan_data_strings[:,5].astype(dtype = np.int32)


# In[68]:


np.array_equal(loan_data_numeric[:,0].astype(dtype = np.int32), loan_data_strings[:,5].astype(dtype = np.int32))


# In[69]:


loan_data_strings = np.delete(loan_data_strings, 5, axis = 1)
header_strings = np.delete(header_strings, 5)


# In[70]:


loan_data_strings[:,5]


# In[71]:


header_strings


# In[72]:


loan_data_numeric[:,0]


# In[73]:


header_numeric


# In[74]:


header_strings


# In[75]:


header_strings[5] = "state_address"


# In[76]:


states_names, states_count = np.unique(loan_data_strings[:,5], return_counts = True)
states_count_sorted = np.argsort(-states_count)
states_names[states_count_sorted], states_count[states_count_sorted]


# In[77]:


loan_data_strings[:,5] = np.where(loan_data_strings[:,5] =='',
                                  0,
                                  loan_data_strings[:,5])


# In[78]:


states_west = np.array(['WA', 'OR','CA','NV','ID','MT', 'WY','UT','CO', 'AZ','NM','HI','AK'])
states_south = np.array(['TX','OK','AR','LA','MS','AL','TN','KY','FL','GA','SC','NC','VA','WV','MD','DE','DC'])
states_midwest = np.array(['ND','SD','NE','KS','MN','IA','MO','WI','IL','IN','MI','OH'])
states_east = np.array(['PA','NY','NJ','CT','MA','VT','NH','ME','RI'])


# In[79]:


loan_data_strings[:,5] = np.where(np.isin(loan_data_strings[:,5], states_west), 1, loan_data_strings[:,5])
loan_data_strings[:,5] = np.where(np.isin(loan_data_strings[:,5], states_south), 2, loan_data_strings[:,5])
loan_data_strings[:,5] = np.where(np.isin(loan_data_strings[:,5], states_midwest), 3, loan_data_strings[:,5])
loan_data_strings[:,5] = np.where(np.isin(loan_data_strings[:,5], states_east), 4, loan_data_strings[:,5])


# In[80]:


np.unique(loan_data_strings[:,5])


# In[81]:


loan_data_strings


# In[82]:


loan_data_strings = loan_data_strings.astype(np.int)


# In[83]:


loan_data_strings


# In[84]:


checkpoint_strings = checkpoint("Checkpoint-Strings", header_strings, loan_data_strings)


# In[85]:


checkpoint_strings["header"]


# In[86]:


checkpoint_strings["data"]


# In[87]:


np.array_equal(checkpoint_strings["data"], loan_data_strings)


# In[88]:


loan_data_numeric


# In[89]:


np.isnan(loan_data_numeric).sum()


# In[90]:


header_numeric


# In[91]:


temporary_fill


# In[92]:


np.isin(loan_data_numeric[:,0], temporary_fill)


# In[93]:


np.isin(loan_data_numeric[:,0], temporary_fill).sum()


# In[94]:


header_numeric


# In[95]:


temporary_stats[:, columns_numeric]


# In[96]:


loan_data_numeric[:,2]


# In[97]:


loan_data_numeric[:,2] = np.where(loan_data_numeric[:,2] == temporary_fill,
                                  temporary_stats[0, columns_numeric[2]],
                                  loan_data_numeric[:,2])
loan_data_numeric[:,2]


# In[98]:


temporary_stats[0,columns_numeric[3]]


# In[99]:


header_numeric


# In[100]:


for i in [1,3,4,5]:
    loan_data_numeric[:,i] = np.where(loan_data_numeric[:,i] == temporary_fill,
                                      temporary_stats[2, columns_numeric[i]],
                                      loan_data_numeric[:,i])
                                                 


# In[101]:


loan_data_numeric


# In[102]:


EUR_USD = np.genfromtxt("EUR-USD.csv", delimiter = ',', autostrip = True, skip_header = 1, usecols = 3)
EUR_USD


# In[103]:


loan_data_strings[:,0]


# In[104]:


exchange_rate = loan_data_strings[:,0]

for i in range(1,13):
    exchange_rate = np.where(exchange_rate == i,
                             EUR_USD[i-1],
                             exchange_rate)
    
exchange_rate = np.where(exchange_rate == 0,
                         np.mean(EUR_USD),
                         exchange_rate)

exchange_rate


# In[105]:


exchange_rate.shape


# In[106]:


loan_data_numeric.shape


# In[107]:


exchange_rate = np.reshape(exchange_rate, (10000,1))


# In[108]:


loan_data_numeric = np.hstack((loan_data_numeric, exchange_rate))


# In[109]:


header_numeric = np.concatenate((header_numeric,np.array(['exchange_rate'])))
header_numeric


# In[110]:


header_numeric


# In[111]:


columns_dollar = np.array([1,2,4,5])


# In[112]:


loan_data_numeric[:,[columns_dollar]]


# In[113]:


loan_data_numeric[:,6]


# In[114]:


for i in columns_dollar:
    loan_data_numeric = np.hstack((loan_data_numeric, np.reshape(loan_data_numeric[:,i] / loan_data_numeric[:,6], (10000,1))))


# In[115]:


loan_data_numeric.shape


# In[116]:


loan_data_numeric


# In[117]:


header_additional = np.array([column_name + '_EUR' for column_name in header_numeric[columns_dollar]])


# In[118]:


header_additional


# In[119]:


header_numeric = np.concatenate((header_numeric, header_additional))


# In[120]:


header_numeric


# In[121]:


header_numeric[columns_dollar] = np.array([column_name + '_USD' for column_name in header_numeric[columns_dollar]])


# In[122]:


header_numeric


# In[123]:


columns_index_oreder = [0,1,7,2,8,3,4,9,5,10,6]


# In[124]:


header_numeric[columns_index_oreder]


# In[125]:


loan_data_numeric


# In[126]:


loan_data_numeric = loan_data_numeric[:,columns_index_oreder]


# In[127]:


header_numeric

loan_data_numeric[:,5]
# In[128]:


loan_data_numeric[:,5] = loan_data_numeric[:,5]/100


# In[129]:


loan_data_numeric[:,5]


# In[130]:


checkpoint_numeric = checkpoint("Checkpoint-Numeric",header_numeric,loan_data_numeric)


# In[131]:


checkpoint_numeric['header'], checkpoint_numeric['data']


# In[132]:


checkpoint_strings['data'].shape


# In[133]:


checkpoint_numeric['data'].shape


# In[134]:


loan_data = np.hstack((checkpoint_numeric['data'], checkpoint_strings['data']))


# In[135]:


loan_data


# In[136]:


np.isnan(loan_data).sum()


# In[137]:


header_full = np.concatenate((checkpoint_numeric['header'],checkpoint_strings['header']))


# In[138]:


loan_data = loan_data[np.argsort(loan_data[:,0])]


# In[139]:


loan_data


# In[140]:


np.argsort(loan_data[:,0])


# In[141]:


loan_data = np.vstack((header_full, loan_data))


# In[142]:


np.savetxt("loan-data-preprocessed.csv",
            loan_data,
            fmt = "%s",
            delimiter = ',')


# In[143]:


import pandas  as pd 


# In[144]:


raw_csv_data = pd.read_csv("Absenteeism-data.csv")


# In[145]:


type(raw_csv_data)


# In[146]:


raw_csv_data


# In[147]:


df = raw_csv_data.copy()


# In[148]:


df


# In[149]:


pd.options.display.max_columns = None
pd.options.display.max_rows = None


# In[150]:


display(df)


# In[151]:


df.info()


# In[152]:


df.drop(['ID'], axis = 1)


# In[153]:


df


# In[154]:


df = df.drop(['ID'], axis = 1)


# In[155]:


df


# In[156]:


raw_csv_data


# In[157]:


df['Reason for Absence']


# In[158]:


df['Reason for Absence'].min()


# In[159]:


df['Reason for Absence'].max()


# In[160]:


pd.unique(df['Reason for Absence'])


# In[161]:


df['Reason for Absence'].unique()


# In[162]:


len(df['Reason for Absence'].unique())


# In[163]:


sorted(df['Reason for Absence'].unique())


# In[164]:


reason_columns = pd.get_dummies(df['Reason for Absence'])
reason_columns


# In[165]:


reason_columns['check'] = reason_columns.sum(axis=1)
reason_columns


# In[166]:


reason_columns['check'].sum(axis=0)


# In[167]:


reason_columns['check'].unique()


# In[168]:


reason_columns = reason_columns.drop(['check'], axis = 1)
reason_columns


# In[169]:


reason_columns = pd.get_dummies(df['Reason for Absence'],drop_first = True)
reason_columns


# In[170]:


df.columns.values


# In[171]:


reason_columns.columns.values


# In[172]:


df = df.drop(['Reason for Absence'], axis = 1)
df


# In[173]:


reason_columns.loc[:, 1:14].max(axis=1)


# In[174]:


reason_type1 = reason_columns.loc[:, 1:14].max(axis=1)
reason_type2 = reason_columns.loc[:, 15:17].max(axis=1)
reason_type3 = reason_columns.loc[:, 18:21].max(axis=1)
reason_type4 = reason_columns.loc[:, 22:].max(axis=1)


# In[175]:


reason_type1


# In[176]:


df


# In[177]:


df = pd.concat([df, reason_type1,reason_type2,reason_type3,reason_type4], axis = 1)
df


# In[178]:


df.columns.values


# In[179]:


column_names = ['Date', 'Transportation Expense', 'Distance to Work', 'Age', 'Daily Work Load Average',
       'Body Mass Index', 'Education', 'Children', 'Pets', 'Absenteeism Time in Hours', 'reason_1','reason_2','reason_3','reason_4']
column_names


# In[180]:


df.columns = column_names


# In[181]:


df.head()


# In[182]:


columns_names_reorder = ['reason_1','reason_2','reason_3','reason_4',
                         'Date', 'Transportation Expense', 'Distance to Work', 'Age',
       'Daily Work Load Average',
       'Body Mass Index', 'Education', 'Children', 'Pets', 'Absenteeism Time in Hours']


# In[183]:


df = df[columns_names_reorder]
df.head()


# In[184]:


df_reason_mod = df.copy()
df_reason_mod


# In[185]:


df_reason_mod['Date']


# In[186]:


type(df_reason_mod['Date'][0])


# In[187]:


df_reason_mod['Date'] = pd.to_datetime(df_reason_mod['Date'])


# In[188]:


df_reason_mod['Date']


# In[189]:


df_reason_mod['Date'] = pd.to_datetime(df_reason_mod['Date'], format = '%d/%m/%Y')
df_reason_mod['Date']


# In[190]:


type(df_reason_mod['Date'][0])


# In[191]:


type(df_reason_mod['Date'][0])


# In[192]:


df_reason_mod.info()


# In[193]:


df_reason_mod['Date'][0]


# In[194]:


df_reason_mod['Date'][0].month


# In[195]:


list_months = []
list_months


# In[196]:


df_reason_mod.shape


# In[197]:


for i in range(df_reason_mod.shape[0]):
    list_months.append(df_reason_mod['Date'][i].month)


# In[198]:


list_months


# In[199]:


len(list_months)


# In[200]:


df_reason_mod['Month Value'] = list_months


# In[201]:


df_reason_mod.head(20)


# In[202]:


################## Day of the Week


# In[203]:


df_reason_mod['Date'][699].weekday()


# In[204]:


df_reason_mod['Date'][699]


# In[205]:


def date_to_weekday(date_value):
    return date_value.weekday()


# In[206]:


df_reason_mod['Dat of the Week'] = df_reason_mod['Date'].apply(date_to_weekday)


# In[207]:


df_reason_mod.head()


# In[208]:


df_reason_date_mod = df_reason_mod.copy()
df_reason_date_mod


# In[209]:


type(df_reason_date_mod['Transportation Expense'][0])


# In[210]:


type(df_reason_date_mod['Distance to Work'][0])


# In[211]:


type(df_reason_date_mod['Age'][0])


# In[212]:


type(df_reason_date_mod['Daily Work Load Average'][0])


# In[213]:


type(df_reason_date_mod['Body Mass Index'][0])


# In[214]:


display(df_reason_date_mod)


# In[215]:


df_reason_date_mod['Education'].unique()


# In[216]:


df_reason_mod['Education'].value_counts()


# In[217]:


df_reason_date_mod['Education'] = df_reason_date_mod['Education'].map({1:0, 2:1, 3:1, 4:1})


# In[218]:


df_reason_date_mod['Education'].unique()


# In[219]:


df_reason_date_mod['Education'].value_counts()


# In[220]:


df_cleaned = df_reason_date_mod.copy()
df_cleaned.head(10)


# In[221]:


############# Visual


# In[222]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()


# In[223]:


df_used_cars = pd.read_csv("bar_chart_data.csv")


# In[224]:


df_used_cars


# In[225]:


plt.figure(figsize = (9, 6))
plt.bar(x = df_used_cars["Brand"],
        height = df_used_cars["Cars Listings"],
        color = "midnightblue")
plt.xticks(rotation = 45, fontsize = 13)
plt.yticks(fontsize = 13)
plt.title("Cars Listings by brand", fontsize = 16, fontweight = "bold")
plt.ylabel("Number of Listings", fontsize = 13)
plt.show()
plt.savefig("Used Cars Bar.png")


# In[226]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()


# In[227]:


df_fuel_engine_type = pd.read_csv("pie_chart_data.csv")


# In[228]:


df_fuel_engine_type


# In[229]:


sns.set_palette('colorblind')


# In[230]:


plt.figure(figsize = (10, 8))
plt.pie(df_fuel_engine_type['Number of Cars'],
        labels = df_fuel_engine_type['Engine Fuel Type'].values,
        autopct = '%.2f%%',
        textprops = {'size' : 'x-large',
                     'fontweight' : 'bold',
                     'rotation' : '30',
                     'color' : 'w'})
plt.legend()
plt.title('Cars by Engine Fuel Type', fontsize = 18, fontweight = 'bold')
plt.show()


# In[231]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()


# In[232]:


df_fuel_engine_types = pd.read_csv("stacked_area_chart_data.csv")


# In[233]:


df_fuel_engine_types


# In[234]:


colors = ['#011638','#7e2987', 'ef2026']
labels = ['Diesel','Patrol','Gas']
sns.set_style('white')
plt.figure(figsize = (12, 6))
plt.stackplot(df_fuel_engine_types['Year'],
              df_fuel_engine_types['Diesel'],
              df_fuel_engine_types['Petrol'],
              df_fuel_engine_types['Gas'],
              colors = colors,
              edgecolor = 'none')
plt.xticks(df_fuel_engine_types['Year'], rotation = 45)
plt.legend(labels = labels, loc = 'upper left')
plt.ylabel('Number of Cars', fontsize = 13)
plt.title('Popularity of Engine Fuel Types (1982 - 2016)',fontsize = 14, weight = 'bold')
sns.despine()
plt.show()


# In[235]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()


# In[236]:


df_spx_ftse_00_10 = pd.read_csv("line_chart_data.csv")


# In[237]:


df_spx_ftse_00_10


# In[238]:


df_spx_ftse_00_10["new_date"] = pd.to_datetime(df_spx_ftse_00_10["Date"])


# In[239]:


df_spx_ftse_00_10["new_date"]


# In[240]:


labels = ["S&P 500", "FTSE 100"]
plt.figure(figsize = (20, 8))
plt.plot(df_spx_ftse_00_10["new_date"], df_spx_ftse_00_10["GSPC500"])
plt.plot(df_spx_ftse_00_10["new_date"], df_spx_ftse_00_10["FTSE100"])
plt.title("S&P vs FTSE Returns (2000 - 2010)", fontsize = 14, fontweight = "bold")
plt.ylabel("Returns")
plt.xlabel("Date")
plt.legend(labels = labels, fontsize = "large")
plt.show()


# In[241]:


df_spx_ftse_H2_08 = df_spx_ftse_00_10[(df_spx_ftse_00_10.new_date >= '2008-07-01') &
                                      (df_spx_ftse_00_10.new_date <= '2008-12-31')]

df_spx_ftse_H2_08


# In[242]:


labels = ["S&P 500", "FTSE 100"]
plt.figure(figsize = (20, 8))
plt.plot(df_spx_ftse_H2_08["new_date"], df_spx_ftse_H2_08["GSPC500"], color = 'midnightblue')
plt.plot(df_spx_ftse_H2_08["new_date"], df_spx_ftse_H2_08["FTSE100"], color = 'crimson')
plt.title("S&P vs FTSE Returns (H2 2008)", fontsize = 14, fontweight = "bold")
plt.ylabel("Returns")
plt.xlabel("Date")
plt.legend(labels = labels, fontsize = "large")
plt.show()


# In[243]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()


# In[244]:


df_real_estate = pd.read_csv("histogram_data.csv")


# In[245]:


df_real_estate


# In[246]:


sns.set_style
plt.figure(figsize = (8, 6))
plt.hist(df_real_estate["Price"],
         bins = 8,
         color ="#108A99")
plt.title("Distribution of Real Estate Prices", fontsize = 14, weight = "bold")
plt.xlabel("Price in (000' $)")
plt.ylabel("Numer of Properties")
sns.despine()
plt.show()


# In[247]:


plt.hist()


# In[248]:


import  pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()


# In[249]:


df_real_estate = pd.read_csv("scatter_data.csv")


# In[250]:


df_real_estate


# In[251]:


plt.figure(figsize = (12, 8))
scatter = plt.scatter(df_real_estate['Area (ft.)'],
            df_real_estate['Price'],
            alpha = 0.6,
            c = df_real_estate['Building Type'],
            cmap = 'viridis')
plt.legend(*scatter.legend_elements(),
           loc = "upper left",
           title = "Building Type")
plt.title("Relationship between Area and Price of California Real Estete",
          fontsize = 14,
          weight = "bold")
plt.xlabel("Area (sq.ft.)", weight = "bold")
plt.ylabel("Price (000's of $)")
plt.show()


# In[252]:


plt.figure(figsize = (12, 8))
sns.scatterplot(df_real_estate['Price'],
                df_real_estate['Area (ft.)'],
                hue = df_real_estate['Building Type'],
                palette = ['black','darkblue','purple','pink','white'],
                s = 100)
plt.title("Relationship between Area and Price of California Real Estete",
          fontsize = 14,
          weight = "bold")
plt.xlabel("Area (sq.ft.)", weight = "bold")
plt.ylabel("Price (000's of $)")
plt.show()


# In[255]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()


# In[256]:


df_ad_expenditure = pd.read_csv("scatter_plot_ii.csv")


# In[257]:


df_ad_expenditure


# In[258]:


#plt.figure(figsize = (10, 8))
sns.set(rc = {'figure.figsize': (9,6)})
sns.regplot(x = "Budget",
            y = "Sales",
            data = df_ad_expenditure,
            scatter_kws = {'color':'k'},
            line_kws = {'color': 'red'})
plt.xlabel("Ad Expenditure in  (000's $)")
plt.ylabel("Sales in (000's Units)")
plt.title("Effect of Ad Expenditure on Sales", fontsize = 14, weight = 'bold')
plt.show()


# In[259]:


sns.lmplot(x = "Budget",
           y = "Sales",
           data = df_ad_expenditure,
           height = 10,
           scatter_kws = {'color':'k'},
           line_kws = {'color': 'red'})
plt.xlabel("Ad Expenditure in  (000's $)")
plt.ylabel("Sales in (000's Units)")
plt.title("Effect of Ad Expenditure on Sales", fontsize = 14, weight = 'bold')
plt.show()


# In[260]:


import pandas as pd
import matplotlib.pyplot as plt

from matplotlib.ticker import PercentFormatter


# In[261]:


df_kdnuggets = pd.read_csv("bar_line_chart_data.csv")


# In[262]:


df_kdnuggets 


# In[263]:


fig, ax = plt.subplots(figsize = (10, 7))

ax.bar(df_kdnuggets["Year"],
       df_kdnuggets["Participants"],
       color = "k")
ax.set_ylabel("Number of Participants",
              weight = "bold")
ax.tick_params(axis = "y",
               width = 2,
               labelsize = "large")
ax1 = ax.twinx()
ax1.set_ylim(0, 1)
ax1.yaxis.set_major_formatter(PercentFormatter(xmax = 1.0))
ax1.plot(df_kdnuggets["Year"],
         df_kdnuggets["Python Users"],
         color = "#b60000",
         marker = "D")

ax1.set_ylabel("Python Users",
               color = "#b60000",
               weight = "bold")
ax1.tick_params(axis = "y",
                color = "#b60000",
                width = 2,
                labelsize = "large")
ax.set_title("KD Nuggets Survey Python Users (2012 - 2019)", fontsize = "14", weight = "bold")


# In[ ]:




