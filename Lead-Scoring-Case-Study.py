#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Supress unnecessary warnings
import warnings
warnings.filterwarnings('ignore')


# In[2]:


#Importing the libraries
import numpy as np
import pandas as pd


# In[3]:


#importing the visualization libraries
import matplotlib.pyplot as plt
import seaborn as sns
# Setting the sytle as Whitegrid to keep it same style
sns.set(style="whitegrid")


# In[4]:


#To see all columns and rows changing the pandas options
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)


# In[5]:


# #Reading the "LEAD" Dataset CSV
df = pd.read_csv(r"C:\Users\Mohamed Thahir N\Documents\Lead+Scoring+Case+Study\Lead Scoring Assignment\Leads.csv")
df.head()


# In[6]:


#Checking the Shape of dataset
df.shape


# ### Dataset has 9240 rows and 37 columns

# In[7]:


# Lets check what are the different variables in the dataset
df.columns


# In[8]:


# Checking the summary of the dataset
df.describe().T


# In[9]:


# Checking the info to see the types of the feature variables
df.info()


# Based on the info quite lot of null values present in the dataset and few categorical variables as well. For we need to treat them accordingly.

# In[10]:


#Let Clean the dataset of missing values in each column
df.isnull().sum().sort_values(ascending=False)


# In[11]:


#Converting 'Select' values to NaN.
df = df.replace('Select', np.nan)


# In[12]:


round(100*(df.isnull().sum()/len(df.index)), 2).sort_values(ascending=False)


# In[13]:


#Let drop the column which has greater missing and threshold value as 45%
for i in df.columns:
    if ((100*(df[i].isnull().sum()/len(df.index))) >= 45):
        df.drop(i, axis=1,inplace=True)


# In[14]:


round(100*(df.isnull().sum()/len(df.index)), 2).sort_values(ascending=False)


# In[15]:


#checking the Categorical Attributes value counts of "Country" column 
df['Country'].value_counts(dropna=False)


# In[16]:


#plotting the Country columnn 
plt.figure(figsize=(15, 5))
s1 = sns.countplot(x='Country', hue='Converted', data=df)
s1.set_xticklabels(s1.get_xticklabels(), rotation=90)
plt.show()


# More than 97% Highest number of leads from INDIA so we can drop this column

# In[17]:


# dropping the "Country" feature
df.drop(['Country'], axis = 1, inplace = True)


# In[18]:


#checking the Categorical Attributes value counts of "City" column 
df['City'].value_counts(dropna=False)


# In[19]:


# Let replace NaN as Mumbai
df['City'] = df['City'].replace(np.nan,'Mumbai')


# In[20]:


#plotting the City columnn 
plt.figure(figsize=(15, 5))
s1 = sns.countplot(x='City', hue='Converted', data=df)
s1.set_xticklabels(s1.get_xticklabels(), rotation=90)
plt.show()


# Mumbai has highest numbers of leads so the variable City can use in our analysis. So it's better to keep it as of now.

# In[21]:


#checking the Categorical Attributes value counts of "Specialization" column 
df['Specialization'].value_counts(dropna=False)


# Here Leads not mentioned specialization but this variable is little important hence replacing it with 'Not Specified'.

# In[22]:


df['Specialization'] = df['Specialization'].replace(np.nan, 'Not Specified')


# In[23]:


#plotting the Specialization columnn 
plt.figure(figsize=(15, 5))
s1 = sns.countplot(x='Specialization', hue='Converted', data=df)
s1.set_xticklabels(s1.get_xticklabels(), rotation=90)
plt.show()


# As per the chart, Management Specialization has higher number of leads as well as leads converted. Hence its a Significant variable so doesn't required to drop this column as well.

# In[25]:


# Let merge all the management course 
df['Specialization'] = df['Specialization'].replace(['Finance Management','Human Resource Management','Marketing Management','Operations Management', 'IT Projects Management','Supply Chain Management','Healthcare Management','Hospitality Management','Retail Management'] ,'Management_Specializations')


# In[26]:


#plotting the Specialization columnn 
plt.figure(figsize=(15, 5))
s1 = sns.countplot(x='Specialization', hue='Converted', data=df)
s1.set_xticklabels(s1.get_xticklabels(), rotation=90)
plt.show()


# In[27]:


#Let focus on "What is your current occupation" this variable
df['What is your current occupation'].value_counts(dropna=False)


# In[28]:


#Let impute the Unemployee itself to NAN 
df['What is your current occupation'] = df['What is your current occupation'].replace(np.nan, 'Unemployed')


# In[29]:


#checking count of values
df['What is your current occupation'].value_counts(dropna=False)


# In[30]:


#plotting the "What is your current occupation" columnn 
plt.figure(figsize=(15, 5))
s1 = sns.countplot(x='What is your current occupation', hue='Converted', data=df)
s1.set_xticklabels(s1.get_xticklabels(), rotation=90)
plt.show()


# Here Unemployed leads are the most in numbers and Working Professionals as well and having high chances of joining it.

# In[31]:


#Let focus on "What matters most to you in choosing a course" this variable
df['What matters most to you in choosing a course'].value_counts(dropna=False)


# In[32]:


#Lets impute the Nan values with Mode "Better Career Prospects"

df['What matters most to you in choosing a course'] = df['What matters most to you in choosing a course'].replace(np.nan,'Better Career Prospects')


# In[33]:


#plotting the "What is your current occupation" columnn 
plt.figure(figsize=(15, 5))
s1 = sns.countplot(x='What matters most to you in choosing a course', hue='Converted', data=df)
s1.set_xticklabels(s1.get_xticklabels(), rotation=90)
plt.show()


# In[35]:


#checking value counts of variable
df['What matters most to you in choosing a course'].value_counts(dropna=False)


# "Better Career Prospects" has highest count so we can drop this column

# In[36]:


# dropping the "What matters most to you in choosing a course" feature
df.drop(['What matters most to you in choosing a course'], axis = 1, inplace = True)


# In[37]:


#checking value counts of Tag variable
df['Tags'].value_counts(dropna=False)


# In[38]:


#replacing Nan values with "Not Specified"
df['Tags'] = df['Tags'].replace(np.nan,'Not Specified')


# In[39]:


#plotting the "Tags" columnn 
plt.figure(figsize=(15, 5))
s1 = sns.countplot(x='Tags', hue='Converted', data=df)
s1.set_xticklabels(s1.get_xticklabels(), rotation=90)
plt.show()


# In[41]:


#replacing tags with low frequency with "Other Tags"
df['Tags'] = df['Tags'].replace(['In confusion whether part time or DLP', 'in touch with EINS','Diploma holder (Not Eligible)',
                                     'Approached upfront','Graduation in progress','number not provided', 'opp hangup','Still Thinking',
                                    'Lost to Others','Shall take in the next coming month','Lateral student','Interested in Next batch',
                                    'Recognition issue (DEC approval)','Want to take admission but has financial problems',
                                    'University not recognized','switched off','Already a student','Not doing further education',
                                       'invalid number','wrong number given','Interested  in full time MBA'] , 'Other_Tags')


# In[42]:


#plotting the "Tags" columnn 
plt.figure(figsize=(15, 5))
s1 = sns.countplot(x='Tags', hue='Converted', data=df)
s1.set_xticklabels(s1.get_xticklabels(), rotation=90)
plt.show()


# In[44]:


#checking percentage of missing values
round(100*(df.isnull().sum()/len(df.index)), 2).sort_values(ascending=False)


# In[45]:


#checking value counts of "Lead Source" column
df['Lead Source'].value_counts(dropna=False)


# In[46]:


#replacing Nan Values and combining low frequency values
df['Lead Source'] = df['Lead Source'].replace(np.nan,'Others')


# In[47]:


df['Lead Source'] = df['Lead Source'].replace('google','Google')
df['Lead Source'] = df['Lead Source'].replace('Facebook','Social Media')


# In[48]:


df['Lead Source'] = df['Lead Source'].replace(['bing','Click2call','Press_Release',
                                                     'youtubechannel','welearnblog_Home',
                                                     'WeLearn','blog','Pay per Click Ads',
                                                    'testone','NC_EDM'] ,'Others') 


# In[49]:


df['Lead Source'].value_counts(dropna=False)


# In[50]:


#plotting the "Lead Source" columnn 
plt.figure(figsize=(15, 5))
s1 = sns.countplot(x='Lead Source', hue='Converted', data=df)
s1.set_xticklabels(s1.get_xticklabels(), rotation=90)
plt.show()


# - Google and direct traffic create the greatest quantity of leads.
# - The conversion rate of reference leads and leads generated by the Welingak website is very high.
# - To increase total lead conversion rate, focus on boosting lead conversion of olark chat, organic search, direct traffic, and google leads, as well as generating more leads from reference and welingak website.

# In[51]:


#checking value counts of "Last Activity" column
df['Last Activity'].value_counts(dropna=False)


# In[52]:


#Imputing the NaN Values and combining low frequency values
df['Last Activity'] = df['Last Activity'].replace(np.nan,'Others')


# In[53]:


df['Last Activity'] = df['Last Activity'].replace(['Unreachable','Unsubscribed',
                                                        'Had a Phone Conversation', 
                                                        'Approached upfront',
                                                        'View in browser link Clicked',       
                                                        'Email Marked Spam',                  
                                                        'Email Received','Resubscribed to emails',
                                                         'Visited Booth in Tradeshow'],'Others')


# In[54]:


df['Last Activity'].value_counts(dropna=False)


# In[55]:


#plotting the "Last Activity" columnn 
plt.figure(figsize=(15, 5))
s1 = sns.countplot(x='Last Activity', hue='Converted', data=df)
s1.set_xticklabels(s1.get_xticklabels(), rotation=90)
plt.show()


# In[57]:


#Check the Null Values in All Columns:
round(100*(df.isnull().sum()/len(df.index)),2).sort_values(ascending=False)


# In[58]:


#Drop all rows which have NaN Values. Since the number of Dropped rows is less than 2%, it will not affect the model
df = df.dropna()


# In[59]:


#Checking percentage of Null Values in All Columns:
round(100*(df.isnull().sum()/len(df.index)), 2).sort_values(ascending=False)


# All the Null values are removed

# In[60]:


#Lead Origin
df['Lead Origin'].value_counts(dropna=False)


# In[61]:


#plotting the "Lead Origin" columnn 
plt.figure(figsize=(15, 5))
s1 = sns.countplot(x='Lead Origin', hue='Converted', data=df)
s1.set_xticklabels(s1.get_xticklabels(), rotation=90)
plt.show()


# - API and Landing Page Submission generate more leads and conversions.
# - The Lead Add Form has a very good conversion rate, however the lead count is not particularly high.
# - Lead Import and Quick Add Form both generate a small number of leads.
# - To increase total lead conversion rate, we must boost lead generation from API and Landing Page Submission origins and produce more leads via Lead Add Form.

# In[63]:


#Lets Visualize "Do Not Email" & "Do Not Call" Variables based on Converted value
plt.figure(figsize=(15,5))
ax1=plt.subplot(1, 2, 1)
s1 = sns.countplot(x='Do Not Call', hue='Converted', data=df)
s1.set_xticklabels(s1.get_xticklabels(), rotation=90)
ax2=plt.subplot(1, 2, 2)
s1 = sns.countplot(x='Do Not Email', hue='Converted', data=df)
s1.set_xticklabels(s1.get_xticklabels(), rotation=90)
plt.show()


# In[64]:


#checking value counts for Do Not Call
df['Do Not Call'].value_counts(dropna=False)


# In[66]:


#checking value counts for Do Not Email
df['Do Not Email'].value_counts(dropna=False)


# Since its "Do Not Call" has 90% only one value

# In[67]:


# dropping the "Do Not Call" feature
df.drop(['Do Not Call'], axis = 1, inplace = True)


# In[69]:


df.Search.value_counts(dropna=False)


# In[70]:


df.Magazine.value_counts(dropna=False)


# In[71]:


df['Newspaper Article'].value_counts(dropna=False)


# In[72]:


df['X Education Forums'].value_counts(dropna=False)


# In[73]:


df['Newspaper'].value_counts(dropna=False)


# In[74]:


df['Digital Advertisement'].value_counts(dropna=False)


# In[75]:


df['Through Recommendations'].value_counts(dropna=False)


# In[76]:


df['Receive More Updates About Our Courses'].value_counts(dropna=False)


# In[77]:


df['Update me on Supply Chain Content'].value_counts(dropna=False)


# In[78]:


df['Get updates on DM Content'].value_counts(dropna=False)


# In[79]:


df['I agree to pay the amount through cheque'].value_counts(dropna=False)


# In[80]:


df['A free copy of Mastering The Interview'].value_counts(dropna=False)


# In[81]:


# dropping the "Do Not Call" feature
df.drop(['Search','Magazine','Newspaper Article','X Education Forums','Newspaper',
                 'Digital Advertisement','Through Recommendations','Receive More Updates About Our Courses',
                 'Update me on Supply Chain Content',
                 'Get updates on DM Content','I agree to pay the amount through cheque'], axis = 1, inplace = True)


# In[82]:


#checking value counts of last Notable Activity
df['Last Notable Activity'].value_counts()


# In[83]:


#Merging all the lower frequency values
df['Last Notable Activity'] = df['Last Notable Activity'].replace(['Had a Phone Conversation','Email Marked Spam','Unreachable','Unsubscribed','Email Bounced',                                                                    
                                                                       'Resubscribed to emails','View in browser link Clicked',
                                                                       'Approached upfront','Form Submitted on Website','Email Received'],'Other_Notable_activity')


# In[84]:


#plotting the "Last Notable Activity" columnn 
plt.figure(figsize=(15, 5))
s1 = sns.countplot(x='Last Notable Activity', hue='Converted', data=df)
s1.set_xticklabels(s1.get_xticklabels(), rotation=90)
plt.show()


# In[85]:


#checking value counts for variable
df['Last Notable Activity'].value_counts()


# In[86]:


df.info()


# So far we have handled every categorical Variable

# In[87]:


#Now lets check the numerical % of Data that has Converted Values = 1:

Converted = (sum(df['Converted'])/len(df['Converted'].index))*100
Converted


# In[89]:


#Checking correlations of numeric values
# figure size
plt.figure(figsize=(10,8))
# heatmap
sns.heatmap(df.corr(), cmap="YlGnBu", annot=True)
plt.show()


# In[92]:


#visualizing the variable of Total Visits
plt.figure(figsize=(6,4))
sns.boxplot(y=df['TotalVisits'])
plt.show()


# Based on the boxplot we can see presence of outliers.

# In[97]:


#checking percentile values for "Total Visits"
df['TotalVisits'].describe(percentiles=[0.05,.25, .5, .75, .90, .95, .99])


# In[100]:


#Outlier Treatment: Remove top & bottom 1% of the Column Outlier values
Q3 = df.TotalVisits.quantile(0.99)
leads = df[(df.TotalVisits <= Q3)]
Q1 = df.TotalVisits.quantile(0.01)
leads = df[(df.TotalVisits >= Q1)]
sns.boxplot(y=df['TotalVisits'])
plt.show()


# In[101]:


df.shape


# In[102]:


#checking percentiles for "Total Time Spent on Website"
df['Total Time Spent on Website'].describe(percentiles=[0.05,.25, .5, .75, .90, .95, .99])


# In[106]:


#visualizing spread of numeric variable
plt.figure(figsize=(6,4))
sns.boxplot(y=df['Total Time Spent on Website'])
plt.show()


# We are not doing any Outlier Treatment for the above Column because there are no significant outliers.

# In[107]:


#checking spread of "Page Views Per Visit"
df['Page Views Per Visit'].describe()


# In[108]:


#visualizing spread of numeric variable
plt.figure(figsize=(6,4))
sns.boxplot(y=df['Page Views Per Visit'])
plt.show()


# In[109]:


#Outlier Treatment: Remove top & bottom 1% 
Q3 = df['Page Views Per Visit'].quantile(0.99)
df = df[df['Page Views Per Visit'] <= Q3]
Q1 = df['Page Views Per Visit'].quantile(0.01)
df = df[df['Page Views Per Visit'] >= Q1]
sns.boxplot(y=df['Page Views Per Visit'])
plt.show()


# In[110]:


df.shape


# In[111]:


#checking Spread of "Total Visits" vs Converted variable
sns.boxplot(y = 'TotalVisits', x = 'Converted', data = df)
plt.show()


# - The medians for converted and non-converted leads are very close.
# - On the basis of Total Visits, nothing conclusive can be claimed.

# In[112]:


#checking Spread of "Total Time Spent on Website" vs Converted variable
sns.boxplot(x=df.Converted, y=df['Total Time Spent on Website'])
plt.show()


# - Leads who spend more time on our website are more likely to convert.
# - To encourage leads to spend more time on the website, it should be made more interesting.

# In[113]:


#checking Spread of "Page Views Per Visit" vs Converted variable
sns.boxplot(x=df.Converted,y=df['Page Views Per Visit'])
plt.show()


# - The median is the same for converted and unconverted leads.
# - Nothing specific can be said about lead conversion from Page Views Per Visit.

# In[114]:


#checking missing values in leftover columns
round(100*(df.isnull().sum()/len(df.index)),2)


# There are no missing values in the columns that need to be examined further.

# In[115]:


# Let create the list of categorical columns for Dummy Variable Creation
catcategorical_columns= df.select_dtypes(include=['object']).columns
catcategorical_columns


# Here we can't use the Prospect ID for creating dummy variable before that will check is there any duplicate ID is present or not

# In[116]:


#check for duplicates
sum(df.duplicated(subset = 'Prospect ID')) == 0


# In[117]:


# let drop the Prospect ID since they have all unique values
df.drop(['Prospect ID'], 1, inplace = True)


# In[118]:


catcategorical_columns= df.select_dtypes(include=['object']).columns
catcategorical_columns


# In[119]:


# List of variables to map
varlist =  ['A free copy of Mastering The Interview','Do Not Email']

# Defining the map function
def binary_map(x):
    return x.map({'Yes': 1, "No": 0})
# Applying the function to the housing list
df[varlist] = df[varlist].apply(binary_map)


# In[120]:


#lets get dummies and dropping the first column and adding the results to the master dataframe
dummy = pd.get_dummies(df[['Lead Origin','What is your current occupation',
                             'City']], drop_first=True)
df = pd.concat([df,dummy],1)


# In[122]:


dummy = pd.get_dummies(df['Specialization'], prefix  = 'Specialization')
dummy = dummy.drop(['Specialization_Not Specified'], 1)
df = pd.concat([df, dummy], axis = 1)


# In[121]:


dummy = pd.get_dummies(df['Lead Source'], prefix  = 'Lead Source')
dummy = dummy.drop(['Lead Source_Others'], 1)
df = pd.concat([df, dummy], axis = 1)


# In[123]:


dummy = pd.get_dummies(df['Last Activity'], prefix  = 'Last Activity')
dummy = dummy.drop(['Last Activity_Others'], 1)
df = pd.concat([df, dummy], axis = 1)


# In[124]:


dummy = pd.get_dummies(df['Last Notable Activity'], prefix  = 'Last Notable Activity')
dummy = dummy.drop(['Last Notable Activity_Other_Notable_activity'], 1)
df = pd.concat([df, dummy], axis = 1)


# In[125]:


dummy = pd.get_dummies(df['Tags'], prefix  = 'Tags')
dummy = dummy.drop(['Tags_Not Specified'], 1)
df = pd.concat([df, dummy], axis = 1)


# In[126]:


#Let drop the original columns after dummy variable creation
df.drop(catcategorical_columns,1,inplace = True)


# In[127]:


df.head()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




