#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
os.path.abspath("")


# In[2]:


import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import seaborn as sns


# ### Correction of the US database

# In[ ]:


data=pd.read_csv('C:/Users/jadea/Documents/stat consulting/Data_AL_US_TSEConseilStats_2023.csv',sep=';')
print(data.columns)
data1=data.iloc[:,0:4]
data2=data.iloc[:,4:]
print(data2)
# Specify the condition for removal (e.g., column A == 'abc')
condition = data2['OPE_REV'] == ' Inc.'
print(condition.sum())
# Set the 'A' column to an empty string for the specified row
data2.loc[condition, 'OPE_REV'] = ''
data2.loc[condition, :] = data2.loc[condition, :].shift(periods=-1, axis=1, fill_value='')
data2.to_csv('C:/Users/jadea/Documents/stat consulting/Data_AL_US_modified2_TSEConseilStats_2023.csv')
data=pd.concat([data1,data2],axis=1)
data.loc[data['UniqueCarrierName']=='Midwest Airline','UniqueCarrierName']='Midwest Airline Inc.'
data.drop('Unnamed: 40', axis=1, inplace=True)
data.to_csv('C:/Users/jadea/Documents/stat consulting/Data_AL_US_modified_TSEConseilStats_2023.csv')
# Display the updated DataFrame
data


# # data exploration Global Database

# In[3]:


data=pd.read_csv('C:/Users/jadea/Documents/stat consulting/Data_AL_MUST_TSEConseilStats_2023.csv',sep=';')
data


# In[4]:


# Convert the entire column to float
data['POPULATION'] = data['POPULATION'].str.replace(',00', '').astype(float)


# In[4]:


data_missing=data

data_missing['UTKT_PRICE_missing'] = 0
data.loc[data_missing['UTKT_PRICE'].isna(), 'UTKT_PRICE_missing'] = 1
data_missing['UTKT_PRICE_missing'].value_counts()


# In[164]:


data_missing.groupby(by=['AIRLINE_ID'])['UTKT_PRICE_missing'].value_counts()


# In[10]:


data.groupby(by=['AIRLINE_ID'])['YEAR'].value_counts().sum()


# In[80]:


# Check for missing 'YIELD' values for each AIRLINE_ID
missing_yield_airline_ids = data[data['UTKT_PRICE'].isnull()]['AIRLINE_ID'].unique()

# Print the number of unique AIRLINE_ID values with at least one missing 'YIELD'
print("Number of AIRLINE_ID values with missing 'UTkT_PRICE':", len(missing_yield_airline_ids))


# In[11]:


data.sort_values(by=['AIRLINE_ID','YEAR']) #data by year and not by quarter as US data


# In[75]:


data['FUEL_COSTS']


# In[6]:


data.duplicated().sum() #no duplicates rows


# In[12]:


len(data['AIRLINE_ID'].unique()) #230 different airlines


# In[15]:


data.corr()


# In[118]:


data['SEC_LENGTH_km']


# In[137]:


data.groupby(by=[['REGION','YEAR']])[['ASK_REGION','YEAR']].value_counts()


# In[133]:


data.groupby(by=['ASK_REGION'])['YEAR'].value_counts()


# In[ ]:


evoluation UTKT_PRICE par region
UTKT_PRICE vs yield


# In[59]:


data_2=data[data['YIELD']<80]


# In[62]:


sns.scatterplot(data=data, y='YIELD', x='UTKT_PRICE')


# In[50]:


sns.scatterplot(data=data[data['YIELD']<80], y='YIELD', x='UTKT_PRICE')


# In[61]:


data_2.corr()['YIELD']
0.65 UTKT_PRICE


# In[41]:


sns.lineplot(data=data, y='ASK_REGION', x='YEAR', hue='REGION', ci=None)
#maybe some missing years


# In[40]:


data.columns
data['ASK_REGION']


# # Missing years ?

# In[48]:


data_sorted=data.sort_values(by=['AIRLINE_ID','YEAR'])
data_sorted
#how many companies do miss at least 1 year ?

grouped_data = data_sorted.groupby('AIRLINE_ID')

# Initialize a variable to count the number of AIRLINE_ID values with missing years
count_missing_years = 0

# Initialize a list to store AIRLINE_ID values with missing years
airline_ids_with_missing_years = []

# Check for missing years in each group
for airline_id, group in grouped_data:
    first_year = group['YEAR'].min()
    last_year = group['YEAR'].max()

    # Check if there is at least one missing year
    if len(range(first_year, last_year + 1)) != len(group['YEAR'].unique()):
        count_missing_years += 1
        airline_ids_with_missing_years.append(airline_id)

print(f'Number of AIRLINE_ID values with at least one missing year: {count_missing_years}')
print('AIRLINE_ID values with at least one missing year:', airline_ids_with_missing_years)
for airline_id in airline_ids_with_missing_years:
    names = data.loc[data['AIRLINE_ID'] == airline_id, 'CURRENT_AIRLINE_NAME'].unique()
    print(f"AIRLINE_ID: {airline_id}, CURRENT_AIRLINE_NAME: {', '.join(names)}")


# # OUTLIERS

# In[5]:


from sklearn.ensemble import IsolationForest


# In[ ]:


# Create an instance of OneHotEncoder
encoder = OneHotEncoder(sparse=False, drop='first')  # 'drop' parameter removes one of the dummy columns to avoid multicollinearity

# Fit and transform the data
encoded_data = encoder.fit_transform(data[['Category']])

# Create a DataFrame with the one-hot encoded columns
encoded_df = pd.DataFrame(encoded_data, columns=encoder.get_feature_names_out(['Category']))

# Concatenate the original DataFrame with the encoded DataFrame
df_encoded = pd.concat([df, encoded_df], axis=1)

# Display the result
print(df_encoded)


# In[6]:


data['POPULATION'].dtype


# In[10]:


# Check if some cells in the 'MixedColumn' are strings
are_strings = data['POPULATION'].apply(lambda x: isinstance(x, float))

# Check if any True values exist (indicating the presence of strings)
if any(are_strings):
    print("Some cells in 'MixedColumn' are strings.")
else:
    print("No strings found in 'MixedColumn'.")


# In[15]:


data['POPULATION']


# In[14]:


data[data['POPULATION']=='1,00E+07']


# In[26]:


data.iloc[80]['POP']


# In[22]:


data['POP']


# In[116]:


# Apply one-hot encoding to all categorical columns
data_encoded = data.copy()
data_encoded.drop(columns=['COUNTRY','CURRENT_AIRLINE_NAME','AIRLINE_ID','ISO','OAG_REGION_NAME','PERIOD_AIRLINE_IATA','PERIOD_AIRLINE_ICAO','PERIOD_AIRLINE_NAME'],inplace=True)
data_encoded=data_encoded[['ASK_REGION','ASK_m_final','CASK_SLA','FUEL_COSTS',
 'FUEL_COSTS_ASK_SLA',
 'FUEL_CRUDE_AVG',
 'FUEL_JET_GULF', 'LH_RATIO',
 'LOAD_FACTOR',
'LOW_COST_FIN','REGION',
 'RPKs_m','UTKT_PRICE',
 'YEAR',
 'YIELD_SLA']]
#data_encoded=data_encoded[['YIELD_SLA','UTKT_PRICE']]
data_encoded = pd.get_dummies(data_encoded)
#print(data_encoded.shape)
#data_encoded.dropna(inplace=True)
data_encoded


# In[65]:


sorted(data.columns)


# In[70]:


sorted(data_encoded.columns)


# In[160]:


print(len(data['POPULATION'].unique()))
data['POPULATION']

#--> population treated as quali !!


# In[117]:


#clf = IsolationForest(contamination=0.01)  # Contamination is the expected proportion of outliers
#clf.fit(data_encoded)

from sklearn.svm import OneClassSVM
clf = OneClassSVM(nu=0.001)
clf.fit(data_encoded)

# Predict outliers
outliers = clf.predict(data_encoded)
outliers


# In[109]:


len(data_encoded)


# In[110]:


len((data_encoded[outliers==-1]).index)


# In[111]:


outliers_data1=data.iloc[(data_encoded[outliers==-1]).index]
inliers_data1=data.iloc[(data_encoded[outliers==1]).index]
#print(inliers_data1)
name_outliers1=sorted(outliers_data1['CURRENT_AIRLINE_NAME'].unique())
name_inliers1=sorted(inliers_data1['CURRENT_AIRLINE_NAME'].unique())
print('outliers:',name_outliers1)
print('inliers:',name_inliers1)
print('intersection:',list(set(name_outliers1) & set(name_inliers1)))


# In[113]:


data_encoded[data['CURRENT_AIRLINE_NAME']=='VIVA AEROBUS'].mean()


# In[115]:


round(data_encoded.mean(),2)


# In[87]:


data.iloc[data_encoded[data['CURRENT_AIRLINE_NAME']=='AEGEAN AIRLINES'].index]


# In[88]:


round(data_encoded.mean(),2)


# In[79]:


data_encoded[data['CURRENT_AIRLINE_NAME']=='EMIRATES']


# In[116]:


correlation_matrix = data.corr()

# Find pairs where the correlation coefficient is greater than 0.8
high_correlation_pairs = [(i, j) for i in range(len(correlation_matrix.columns)) 
                          for j in range(i+1, len(correlation_matrix.columns)) 
                          if abs(correlation_matrix.iloc[i, j]) > 0.8]

# Display the high correlation pairs
for pair in high_correlation_pairs:
    var1, var2 = correlation_matrix.columns[pair[0]], correlation_matrix.columns[pair[1]]
    print(f'Correlation between {var1} and {var2}: {correlation_matrix.iloc[pair]}')


# # Verification of data coherence 

# In[35]:


((data['FLOW_INT']+data['FLOW_REG'])>1).sum() #graph ?


# In[36]:


data.loc[(data['FLOW_INT']+data['FLOW_REG'])>1][['FLOW_INT','FLOW_REG']] #okay : just a rounding issue 


# In[70]:


sns.boxplot(data=data, x='FLOW_INT') #or FLOW_INT


# In[81]:


sns.boxplot(data=data, x='UTKT_PRICE')


# In[52]:


sns.boxplot(data['CASK_SLA']/data['CASK'])


# In[47]:


print(data['CTRL_TYPE'].unique())
#sns.piechart(data=data, x='CTRL_TYPE')
data['CTRL_TYPE'].value_counts()/len(data)


# In[66]:


sns.boxplot(data=data, x='ASK_REGION')


# In[72]:


sns.boxplot(data=data, x='YIELD_SLA',showfliers=False)


# In[74]:


data[(data['UTKT_PRICE'].isna())&(data['YIELD'].isna())]['YEAR'].value_counts()


# In[151]:


data


# In[74]:


data[data['YIELD']>80]


# In[73]:


data3=data.dropna()
data3


# In[69]:


data_2.corr()['UTKT_PRICE']


# In[73]:


data.corr()['UTKT_PRICE']
#bien correlé avec : YEAR - , PAX_PURCHASING_POWER + , SEC_LENGTH_AVG - ??, pas du tout avec le YIELD !!!!!
FUEL_COSTS ~ 0.17 et ~ 0.13 SLA, FUEL_COSTS_ASK 0.36, ASK_GLOBAL et RPK_GLOBAL -, POPULATION - 0.4, HHI_GLOBAL 0.28, CPI_WW 0.36
#plein de variables macro


# In[57]:


len(data[(data['UTKT_PRICE'].isna())&(data['YIELD'].isna())]['AIRLINE_ID'].unique()) #149 airlines for which yield AND UTKT
#Price missing at some point
#toutes les années, toutes les régions, un peu plus les années récentes


# In[64]:


data[data['LOAD_FACTOR_REGION']<0.5]['REGION'].unique() #africa/middle east load factor regional bas


# In[ ]:


ccl

ISO=COUNTRY
ASK_m_final semble faible

YIELD                         874
YIELD_SLA                     874
UTKT_PRICE                    667

Column POPULATION transformée en float ok mais mauvais arrondi (RWANDA = 10M) 

questions

PERIOD_AIRLINE_IATA ? ok
OAG_REGION_NAME manquant, comment faire ?
regional = changer de pays
FUEL_COSTS : unité ?
SEC_LENGTH_km/SEC_LENGTH_AVG ?
LOAD_FACTOR>1 ??
could not convert string to float: '1,00E+07' in POPULATION
ASK_m_final semble faible
# In[98]:


X.reshape(-1,1)


# In[99]:


X


# In[68]:


#CHANGE REGION DO OBSERVE RESULTS : --> in all regions, increasing fuel_costs increases UTKT_PRICE, except MIDDLE EAST
#were it decreases, and south america where it seems to level out! 

print(data['REGION'].unique())

print(data['UTKT_PRICE'].corr(data['FUEL_COSTS']))
region='AFRICA'
sns.scatterplot(data=data[data['REGION']==region], y='UTKT_PRICE', x='FUEL_COSTS', hue='REGION')

#regression per region
from scipy.stats import linregress
from sklearn.linear_model import LinearRegression

data_reg=data.dropna(subset=['FUEL_COSTS', 'UTKT_PRICE'])
data_reg=data_reg[data_reg['REGION']==region]

X = data_reg['FUEL_COSTS']  # Features
X=X.to_numpy()
X=X.reshape(-1,1)
y = data_reg['UTKT_PRICE']     # Target
y=y.to_numpy()
y=y.reshape(-1,1)

# Create and fit the linear regression model
model = LinearRegression()
model.fit(X, y)

# Get the slope and intercept
slope = model.coef_[0]
intercept = model.intercept_
print(slope)

# Create a regression line
regression_line = slope * X + intercept

# Plot the regression line
plt.plot(X, regression_line, color='red', label='Linear Regression Line (Scikit-learn)')
plt.show()


# In[34]:


sns.scatterplot(data=data, x='FUEL_COSTS', y='FUEL_COSTS_ASK')


# In[35]:


data['ratio_fuel_ask']=data['FUEL_COSTS']/data['FUEL_COSTS_ASK']
data[['FUEL_COSTS','FUEL_COSTS_ASK','ratio_fuel_ask']]


# In[ ]:


duplicates ?
anomalies ? 
correlation matrix
verifier cohérence variable (=interval?...)
boxplot resume
utkt_price + yield manquants
FUEL_COSTS : prix different en fonction des regions ? manquants pour tjrs les memes airlines ? 

to do:
UTKT_PRICE to fuel price regression/scatterplot
evolution du trafic aerien dans le temps (en RPK et ASK, par region)
timeline rpk avec une courbe pour chaque airline
evolution du prix des billets dans notre base, pour comparer avec docs airbus
relation yield/gdp per capita
graphiques classiques : RPK/ASK, Load factor, 
expliquer UTKT_PRICE_missing pour savoir si il existe un biais à enlever ces variables ?
données manquantes complétement (=trou année ?)



# In[159]:


data.isna().sum()


# In[144]:


corres={}
for enum, var in enumerate(['ASK_GLOBAL','ASK_REGION','ASK_m_final']):  
    print(var)
    print(enum)
    df1=data
    df1.sort_values(by=[var,'YEAR'],inplace=True)
    df1=data.drop_duplicates(subset='YEAR')
    df1['{}_GROWTH'.format(var)]=np.log(df1[var]) - np.log(df1[var].shift(1))
    corres[var]=df1[[var,'{}_GROWTH'.format(var)]]
corres


# In[55]:


corres={}
for var,enum in enumerate(['ASK_GLOBAL','ASK_REGION','ASK_m_final']):  
    df1=data
    df1.sort_values(by=[var,'YEAR'],inplace=True)
    df1=data.drop_duplicates(subset='YEAR')
    df1['{}_GROWTH'.format(var)]=np.log(df1[var]) - np.log(df1[var].shift(1))
    corres[i]=df1[[var,'{}_GROWTH'.format(var)]]
    corres

df2=data
df2.sort_values(by=['ASK_REGION','YEAR'],inplace=True)
df2=data.drop_duplicates(subset='YEAR')
df2['ASK_REGION_GROWTH']=np.log(df2['ASK_REGION']) - np.log(df2['ASK_REGION'].shift(1))
corres2=df2[['ASK_REGION_GROWTH','ASK_m_final_GROWTH','ASK_GLOBAL_GROWTH','ASK_m_final','ASK_GLOBAL','ASK_REGION']]
corres2

df2=data
df2.sort_values(by=['ASK_REGION','YEAR'],inplace=True)
df2=data.drop_duplicates(subset='YEAR')
df2['ASK_REGION_GROWTH']=np.log(df2['ASK_REGION']) - np.log(df2['ASK_REGION'].shift(1))
corres2=df2[['ASK_REGION_GROWTH','ASK_m_final_GROWTH','ASK_GLOBAL_GROWTH','ASK_m_final','ASK_GLOBAL','ASK_REGION']]
corres2

df1['ASK_REGION_GROWTH']=np.log(df1['ASK_REGION']) - np.log(df1['ASK_REGION'].shift(1))
df1['ASK_m_final_GROWTH']=np.log(df1['ASK_m_final']) - np.log(df1['ASK_m_final'].shift(1))
#df1['ASK_GLOBAL_GROWTH']=df1['ASK_GLOBAL'].apply(lambda x:math.log(x)-math.log(x-1))
print(df1)
corres1=df1[['ASK_REGION_GROWTH','ASK_m_final_GROWTH','ASK_GLOBAL_GROWTH','ASK_m_final','ASK_GLOBAL','ASK_REGION']]
corres1
pd.merge(on="ASK_GLOBAL")
#df['A_squared'] = df['A'].apply(lambda x: x ** 2)

