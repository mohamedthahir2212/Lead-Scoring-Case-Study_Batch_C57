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


#check for duplicates for these two columns
sum(df.duplicated(subset = 'Prospect ID')) == 0
sum(df.duplicated(subset = 'Lead Number')) == 0


# No duplicate values found in Prospect ID & Lead Number and for these varaibles we can't create the duplicate variables.

# In[12]:


#dropping Lead Number and Prospect ID since they have all unique values
df.drop(['Prospect ID', 'Lead Number'], 1, inplace = True)


# In[13]:


#Converting 'Select' values to NaN.
df = df.replace('Select', np.nan)


# In[14]:


#checking percentage of null values in each column
round(100*(df.isnull().sum()/len(df.index)), 2).sort_values(ascending=False)


# In[15]:


#Let drop the column which has greater missing and threshold value as 45%
for i in df.columns:
    if ((100*(df[i].isnull().sum()/len(df.index))) >= 45):
        df.drop(i, axis=1,inplace=True)


# In[16]:


round(100*(df.isnull().sum()/len(df.index)), 2).sort_values(ascending=False)


# In[17]:


#checking the Categorical Attributes value counts of "Country" column 
df['Country'].value_counts(dropna=False)


# In[18]:


#plotting the Country columnn 
plt.figure(figsize=(15, 5))
s1 = sns.countplot(x='Country', hue='Converted', data=df)
s1.set_xticklabels(s1.get_xticklabels(), rotation=90)
plt.show()


# More than 97% Highest number of leads from INDIA so we can drop this column

# In[19]:


#creating a list of columns to be droppped
columns_to_drop=['Country']


# In[20]:


#checking the Categorical Attributes value counts of "City" column 
df['City'].value_counts(dropna=False)


# In[21]:


# Let replace NaN as Mumbai
df['City'] = df['City'].replace(np.nan,'Mumbai')


# In[22]:


#plotting the City columnn 
plt.figure(figsize=(15, 5))
s1 = sns.countplot(x='City', hue='Converted', data=df)
s1.set_xticklabels(s1.get_xticklabels(), rotation=90)
plt.show()


# Mumbai has highest numbers of leads so the variable City can use in our analysis. So it's better to keep it as of now.

# In[23]:


#checking the Categorical Attributes value counts of "Specialization" column 
df['Specialization'].value_counts(dropna=False)


# Here Leads not mentioned specialization but this variable is little important hence replacing it with 'Not Specified'.

# In[24]:


df['Specialization'] = df['Specialization'].replace(np.nan, 'Not Specified')


# In[25]:


#plotting the Specialization columnn 
plt.figure(figsize=(15, 5))
s1 = sns.countplot(x='Specialization', hue='Converted', data=df)
s1.set_xticklabels(s1.get_xticklabels(), rotation=90)
plt.show()


# As per the chart, Management Specialization has higher number of leads as well as leads converted. Hence its a Significant variable so doesn't required to drop this column as well.

# In[26]:


# Let merge all the management course 
df['Specialization'] = df['Specialization'].replace(['Finance Management','Human Resource Management','Marketing Management','Operations Management', 'IT Projects Management','Supply Chain Management','Healthcare Management','Hospitality Management','Retail Management'] ,'Management_Specializations')


# In[27]:


#plotting the Specialization columnn 
plt.figure(figsize=(15, 5))
s1 = sns.countplot(x='Specialization', hue='Converted', data=df)
s1.set_xticklabels(s1.get_xticklabels(), rotation=90)
plt.show()


# In[28]:


#Let focus on "What is your current occupation" this variable
df['What is your current occupation'].value_counts(dropna=False)


# In[29]:


#Let impute the Unemployee itself to NAN 
df['What is your current occupation'] = df['What is your current occupation'].replace(np.nan, 'Unemployed')


# In[30]:


#checking count of values
df['What is your current occupation'].value_counts(dropna=False)


# In[31]:


#plotting the "What is your current occupation" columnn 
plt.figure(figsize=(15, 5))
s1 = sns.countplot(x='What is your current occupation', hue='Converted', data=df)
s1.set_xticklabels(s1.get_xticklabels(), rotation=90)
plt.show()


# Here Unemployed leads are the most in numbers and Working Professionals as well and having high chances of joining it.

# In[32]:


#Let focus on "What matters most to you in choosing a course" this variable
df['What matters most to you in choosing a course'].value_counts(dropna=False)


# In[33]:


#Lets impute the Nan values with Mode "Better Career Prospects"
df['What matters most to you in choosing a course'] = df['What matters most to you in choosing a course'].replace(np.nan,'Better Career Prospects')


# In[34]:


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


# Adding to drop the "What matters most to you in choosing a course" feature
columns_to_drop.append('What matters most to you in choosing a course')
columns_to_drop


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


# In[40]:


#replacing tags with low frequency with "Other Tags"
df['Tags'] = df['Tags'].replace(['In confusion whether part time or DLP', 'in touch with EINS','Diploma holder (Not Eligible)',
                                     'Approached upfront','Graduation in progress','number not provided', 'opp hangup','Still Thinking',
                                    'Lost to Others','Shall take in the next coming month','Lateral student','Interested in Next batch',
                                    'Recognition issue (DEC approval)','Want to take admission but has financial problems',
                                    'University not recognized','switched off','Already a student','Not doing further education',
                                       'invalid number','wrong number given','Interested  in full time MBA'] , 'Other_Tags')


# In[41]:


#plotting the "Tags" columnn 
plt.figure(figsize=(15, 5))
s1 = sns.countplot(x='Tags', hue='Converted', data=df)
s1.set_xticklabels(s1.get_xticklabels(), rotation=90)
plt.show()


# In[42]:


#checking percentage of missing values
round(100*(df.isnull().sum()/len(df.index)), 2).sort_values(ascending=False)


# In[43]:


#checking value counts of "Lead Source" column
df['Lead Source'].value_counts(dropna=False)


# In[44]:


#replacing Nan Values and combining low frequency values
df['Lead Source'] = df['Lead Source'].replace(np.nan,'Others')


# In[45]:


df['Lead Source'] = df['Lead Source'].replace('google','Google')
df['Lead Source'] = df['Lead Source'].replace('Facebook','Social Media')


# In[46]:


df['Lead Source'] = df['Lead Source'].replace(['bing','Click2call','Press_Release',
                                                     'youtubechannel','welearnblog_Home',
                                                     'WeLearn','blog','Pay per Click Ads',
                                                    'testone','NC_EDM'] ,'Others') 


# In[47]:


df['Lead Source'].value_counts(dropna=False)


# In[48]:


#plotting the "Lead Source" columnn 
plt.figure(figsize=(15, 5))
s1 = sns.countplot(x='Lead Source', hue='Converted', data=df)
s1.set_xticklabels(s1.get_xticklabels(), rotation=90)
plt.show()


# - Google and direct traffic create the greatest quantity of leads.
# - The conversion rate of reference leads and leads generated by the Welingak website is very high.
# - To increase total lead conversion rate, focus on boosting lead conversion of olark chat, organic search, direct traffic, and google leads, as well as generating more leads from reference and welingak website.

# In[49]:


#checking value counts of "Last Activity" column
df['Last Activity'].value_counts(dropna=False)


# In[50]:


#Imputing the NaN Values and combining low frequency values
df['Last Activity'] = df['Last Activity'].replace(np.nan,'Others')


# In[51]:


df['Last Activity'] = df['Last Activity'].replace(['Unreachable','Unsubscribed',
                                                        'Had a Phone Conversation', 
                                                        'Approached upfront',
                                                        'View in browser link Clicked',       
                                                        'Email Marked Spam',                  
                                                        'Email Received','Resubscribed to emails',
                                                         'Visited Booth in Tradeshow'],'Others')


# In[52]:


df['Last Activity'].value_counts(dropna=False)


# In[53]:


#plotting the "Last Activity" columnn 
plt.figure(figsize=(15, 5))
s1 = sns.countplot(x='Last Activity', hue='Converted', data=df)
s1.set_xticklabels(s1.get_xticklabels(), rotation=90)
plt.show()


# In[54]:


#Check the Null Values in All Columns:
round(100*(df.isnull().sum()/len(df.index)),2).sort_values(ascending=False)


# In[55]:


#Drop all rows which have NaN Values. Since the number of Dropped rows is less than 2%, it will not affect the model
#threshold_percentage = 2
#missing_values_per_row = df.isnull().sum(axis=1)
#percentage_missing_per_row = (missing_values_per_row / len(df.columns)) * 100
#df = df[percentage_missing_per_row >= threshold_percentage]
df = df.dropna()


# In[56]:


#Checking percentage of Null Values in All Columns:
round(100*(df.isnull().sum()/len(df.index)), 2).sort_values(ascending=False)


# In[57]:


#Lead Origin
df['Lead Origin'].value_counts(dropna=False)


# In[58]:


#plotting the "Lead Origin" columnn 
plt.figure(figsize=(15, 5))
s1 = sns.countplot(x='Lead Origin', hue='Converted', data=df)
s1.set_xticklabels(s1.get_xticklabels(), rotation=90)
plt.show()


# - API and Landing Page Submission generate more leads and conversions.
# - The Lead Add Form has a very good conversion rate, however the lead count is not particularly high.
# - Lead Import and Quick Add Form both generate a small number of leads.
# - To increase total lead conversion rate, we must boost lead generation from API and Landing Page Submission origins and produce more leads via Lead Add Form.

# In[59]:


#Lets Visualize "Do Not Email" & "Do Not Call" Variables based on Converted value
plt.figure(figsize=(15,5))
ax1=plt.subplot(1, 2, 1)
s1 = sns.countplot(x='Do Not Call', hue='Converted', data=df)
s1.set_xticklabels(s1.get_xticklabels(), rotation=90)
ax2=plt.subplot(1, 2, 2)
s1 = sns.countplot(x='Do Not Email', hue='Converted', data=df)
s1.set_xticklabels(s1.get_xticklabels(), rotation=90)
plt.show()


# In[60]:


#checking value counts for Do Not Call
df['Do Not Call'].value_counts(dropna=False)


# In[61]:


#checking value counts for Do Not Email
df['Do Not Email'].value_counts(dropna=False)


# Since its "Do Not Call" has 90% only one value

# In[62]:


columns_to_drop.append('Do Not Call')
columns_to_drop


# In[63]:


df.Search.value_counts(dropna=False)


# In[64]:


df.Magazine.value_counts(dropna=False)


# In[65]:


df['Newspaper Article'].value_counts(dropna=False)


# In[66]:


df['X Education Forums'].value_counts(dropna=False)


# In[67]:


df['Newspaper'].value_counts(dropna=False)


# In[68]:


df['Digital Advertisement'].value_counts(dropna=False)


# In[69]:


df['Through Recommendations'].value_counts(dropna=False)


# In[70]:


df['Receive More Updates About Our Courses'].value_counts(dropna=False)


# In[71]:


df['Update me on Supply Chain Content'].value_counts(dropna=False)


# In[72]:


df['Get updates on DM Content'].value_counts(dropna=False)


# In[73]:


df['I agree to pay the amount through cheque'].value_counts(dropna=False)


# In[74]:


df['A free copy of Mastering The Interview'].value_counts(dropna=False)


# In[75]:


#Adding all this columns to the list of columns to be dropped
columns_to_drop.extend(['Search','Magazine','Newspaper Article','X Education Forums','Newspaper',
                 'Digital Advertisement','Through Recommendations','Receive More Updates About Our Courses',
                 'Update me on Supply Chain Content',
                 'Get updates on DM Content','I agree to pay the amount through cheque'])


# In[76]:


#checking value counts of last Notable Activity
df['Last Notable Activity'].value_counts()


# In[77]:


#Merging all the lower frequency values
df['Last Notable Activity'] = df['Last Notable Activity'].replace(['Had a Phone Conversation','Email Marked Spam','Unreachable','Unsubscribed','Email Bounced',                                                                    
                                                                       'Resubscribed to emails','View in browser link Clicked',
                                                                       'Approached upfront','Form Submitted on Website','Email Received'],'Other_Notable_activity')


# In[78]:


#plotting the "Last Notable Activity" columnn 
plt.figure(figsize=(15, 5))
s1 = sns.countplot(x='Last Notable Activity', hue='Converted', data=df)
s1.set_xticklabels(s1.get_xticklabels(), rotation=90)
plt.show()


# In[79]:


#checking value counts for variable
df['Last Notable Activity'].value_counts()


# In[80]:


#list of columns to be dropped
columns_to_drop


# In[81]:


#dropping columns
df = df.drop(columns_to_drop,1)
df.info()


# So far we have handled every categorical Variable

# In[82]:


#Now lets check the numerical % of Data that has Converted Values = 1:

Converted = (sum(df['Converted'])/len(df['Converted'].index))*100
Converted


# In[83]:


#Checking correlations of numeric values
# figure size
plt.figure(figsize=(10,8))
# heatmap
sns.heatmap(df.corr(), cmap="YlGnBu", annot=True)
plt.show()


# In[84]:


#visualizing the variable of Total Visits
plt.figure(figsize=(6,4))
sns.boxplot(y=df['TotalVisits'])
plt.show()


# Based on the boxplot we can see presence of outliers.

# In[85]:


#checking percentile values for "Total Visits"
df['TotalVisits'].describe(percentiles=[0.05,.25, .5, .75, .90, .95, .99])


# In[86]:


#Outlier Treatment: Remove top & bottom 1% of the Column Outlier values
Q3 = df.TotalVisits.quantile(0.99)
df = df[(df.TotalVisits <= Q3)]
Q1 = df.TotalVisits.quantile(0.01)
df = df[(df.TotalVisits >= Q1)]
sns.boxplot(y=df['TotalVisits'])
plt.show()


# In[87]:


df.shape


# In[88]:


#checking percentiles for "Total Time Spent on Website"
df['Total Time Spent on Website'].describe(percentiles=[0.05,.25, .5, .75, .90, .95, .99])


# In[89]:


#visualizing spread of numeric variable
plt.figure(figsize=(6,4))
sns.boxplot(y=df['Total Time Spent on Website'])
plt.show()


# We are not doing any Outlier Treatment for the above Column because there are no significant outliers.

# In[90]:


#checking spread of "Page Views Per Visit"
df['Page Views Per Visit'].describe()


# In[91]:


#visualizing spread of numeric variable
plt.figure(figsize=(6,4))
sns.boxplot(y=df['Page Views Per Visit'])
plt.show()


# In[92]:


#Outlier Treatment: Remove top & bottom 1% 
Q3 = df['Page Views Per Visit'].quantile(0.99)
df = df[df['Page Views Per Visit'] <= Q3]
Q1 = df['Page Views Per Visit'].quantile(0.01)
df = df[df['Page Views Per Visit'] >= Q1]
sns.boxplot(y=df['Page Views Per Visit'])
plt.show()


# In[93]:


df.shape


# In[94]:


#checking Spread of "Total Visits" vs Converted variable
sns.boxplot(y = 'TotalVisits', x = 'Converted', data = df)
plt.show()


# - The medians for converted and non-converted leads are very close.
# - On the basis of Total Visits, nothing conclusive can be claimed.

# In[95]:


#checking Spread of "Total Time Spent on Website" vs Converted variable
sns.boxplot(x=df.Converted, y=df['Total Time Spent on Website'])
plt.show()


# - Leads who spend more time on our website are more likely to convert.
# - To encourage leads to spend more time on the website, it should be made more interesting.

# In[96]:


#checking Spread of "Page Views Per Visit" vs Converted variable
sns.boxplot(x=df.Converted,y=df['Page Views Per Visit'])
plt.show()


# - The median is the same for converted and unconverted leads.
# - Nothing specific can be said about lead conversion from Page Views Per Visit.

# In[97]:


#checking missing values in leftover columns
round(100*(df.isnull().sum()/len(df.index)),2)


# There are no missing values in the columns that need to be examined further.

# In[98]:


# Let create the list of categorical columns for Dummy Variable Creation
catcategorical_columns= df.select_dtypes(include=['object']).columns
catcategorical_columns


# In[99]:


# List of variables to map
varlist =  ['A free copy of Mastering The Interview','Do Not Email']

# Defining the map function
def binary_map(x):
    return x.map({'Yes': 1, "No": 0})
# Applying the function to the housing list
df[varlist] = df[varlist].apply(binary_map)


# In[100]:


#lets get dummies and dropping the first column and adding the results to the master dataframe
dummy = pd.get_dummies(df[['Lead Origin','What is your current occupation',
                             'City']], drop_first=True)
df = pd.concat([df,dummy],1)


# In[101]:


dummy = pd.get_dummies(df['Specialization'], prefix  = 'Specialization')
dummy = dummy.drop(['Specialization_Not Specified'], 1)
df = pd.concat([df, dummy], axis = 1)


# In[102]:


dummy = pd.get_dummies(df['Lead Source'], prefix  = 'Lead Source')
dummy = dummy.drop(['Lead Source_Others'], 1)
df = pd.concat([df, dummy], axis = 1)


# In[103]:


dummy = pd.get_dummies(df['Last Activity'], prefix  = 'Last Activity')
dummy = dummy.drop(['Last Activity_Others'], 1)
df = pd.concat([df, dummy], axis = 1)


# In[104]:


dummy = pd.get_dummies(df['Last Notable Activity'], prefix  = 'Last Notable Activity')
dummy = dummy.drop(['Last Notable Activity_Other_Notable_activity'], 1)
df = pd.concat([df, dummy], axis = 1)


# In[105]:


dummy = pd.get_dummies(df['Tags'], prefix  = 'Tags')
dummy = dummy.drop(['Tags_Not Specified'], 1)
df = pd.concat([df, dummy], axis = 1)


# In[106]:


#Let drop the original columns after dummy variable creation
df.drop(catcategorical_columns,1,inplace = True)


# In[107]:


df.head()


# As per the data currently no categorical value are present in the dataset

# In[108]:


# lets create the Train-Test Split & Logistic Regression Model Building:
from sklearn.model_selection import train_test_split

# Putting response variable to y
y = df['Converted']

y.head()

X=df.drop('Converted', axis=1)


# In[109]:


# Splitting the data into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, test_size=0.3, random_state=100)


# In[110]:


X_train.info()


# In[111]:


#let Start the Scaling process for our Dataset:

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

numerical_columns=X_train.select_dtypes(include=['float64', 'int64']).columns

X_train[numerical_columns] = scaler.fit_transform(X_train[numerical_columns])

X_train.head()


# In[112]:


# Let built the Model using Stats Model & RFE:
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE


# In[113]:


# Create a logistic regression model
logreg = LogisticRegression()

rfe = RFE(logreg, n_features_to_select=15) 
rfe = rfe.fit(X_train, y_train)


# In[126]:


# Displaying the selected features
selected_features = X_train.columns[rfe.support_]
selected_features


# In[127]:


# Displaying the not selected features
Not_selected_features = list(X_train.columns[~rfe.support_])
Not_selected_features


# In[128]:


# Let start the Model_1 
X_train_sm = sm.add_constant(X_train[selected_features])
logm1 = sm.GLM(y_train,X_train_sm, family = sm.families.Binomial())
res = logm1.fit()
res.summary()


# p-value of variable Lead Origin_Lead Add Form is high, so we can drop it.

# In[129]:


#Let dropping column with high p-value
selected_features = selected_features.drop('Lead Origin_Lead Add Form',1)


# In[130]:


# Let go ahead with Model_2
X_train_sm = sm.add_constant(X_train[selected_features])
logm2 = sm.GLM(y_train,X_train_sm, family = sm.families.Binomial())
res = logm2.fit()
res.summary()


# - p-value of variable Tags_Closed by Horizzon is high, so we can drop it.

# In[131]:


#dropping column with high p-value
selected_features = selected_features.drop('Tags_Closed by Horizzon',1)


# In[132]:


# Let go ahead with Model_3
X_train_sm = sm.add_constant(X_train[selected_features])
logm2 = sm.GLM(y_train,X_train_sm, family = sm.families.Binomial())
res = logm2.fit()
res.summary()


# - p-value of variable Last Notable Activity_Modified is high, so we can drop it.

# In[133]:


#dropping column with high p-value
selected_features = selected_features.drop('Last Notable Activity_Modified',1)


# In[134]:


# Let go ahead with Model_4
X_train_sm = sm.add_constant(X_train[selected_features])
logm2 = sm.GLM(y_train,X_train_sm, family = sm.families.Binomial())
res = logm2.fit()
res.summary()


# - p-value of variable Last Activity_Page Visited on Website is high so we can drop it

# In[135]:


#dropping column with high p-value
selected_features = selected_features.drop('Last Activity_Page Visited on Website',1)


# In[136]:


# Let go ahead with Model_5
X_train_sm = sm.add_constant(X_train[selected_features])
logm2 = sm.GLM(y_train,X_train_sm, family = sm.families.Binomial())
res = logm2.fit()
res.summary()


# - p-value of variable Tags_Busy is high so we can drop it

# In[137]:


#dropping column with high p-value
selected_features = selected_features.drop('Tags_Busy',1)


# In[138]:


# Let go ahead with Model_6
X_train_sm = sm.add_constant(X_train[selected_features])
logm2 = sm.GLM(y_train,X_train_sm, family = sm.families.Binomial())
res = logm2.fit()
res.summary()


# - We can check the Variance Inflation Factor to see whether there is a correlation between the variables because 'All' of the p-values are fewer.

# In[140]:


# Check for the VIF values of the feature variables. 
from statsmodels.stats.outliers_influence import variance_inflation_factor


# In[141]:


# Creating the dataframe with all the feature variables and their respective VIFs
vif = pd.DataFrame()
vif['Features'] = X_train[selected_features].columns
vif['VIF'] = [variance_inflation_factor(X_train[selected_features].values, i) for i in range(X_train[selected_features].shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif


# Based on the values, everything appears to be in order, therefore we can proceed to derive the Probabilities, Lead Score, and Predictions on Train Data:

# In[142]:


# Getting the Predicted values on the train set
y_train_pred = res.predict(X_train_sm)
y_train_pred[:10]


# In[143]:


y_train_pred = y_train_pred.values.reshape(-1)
y_train_pred[:10]


# In[144]:


y_train_pred_final = pd.DataFrame({'Converted':y_train.values, 'Converted_prob':y_train_pred})
y_train_pred_final['Prospect ID'] = y_train.index
y_train_pred_final.head()


# In[145]:


y_train_pred_final['Predicted'] = y_train_pred_final.Converted_prob.map(lambda x: 1 if x > 0.5 else 0)
y_train_pred_final.head()


# In[146]:


#lets check confusion matrix
from sklearn import metrics

# Confusion matrix 
confusion = metrics.confusion_matrix(y_train_pred_final.Converted, y_train_pred_final.Predicted )
confusion


# In[147]:


# Let's focus on the overall accuracy.
metrics.accuracy_score(y_train_pred_final.Converted, y_train_pred_final.Predicted)


# In[148]:


# True Positive 
TP = confusion[1,1]
# True Negatives
TN = confusion[0,0] 
# False Positives
FP = confusion[0,1]
# False Negatives
FN = confusion[1,0] 


# In[149]:


#Sensitivity of our logistic regression model
TP / float(TP+FN)


# In[150]:


# Then specificity
TN / float(TN+FP)


# In[151]:


# False Postive Rate
FP/ float(TN+FP)


# In[152]:


# positive predictive value 
TP / float(TP+FP)


# In[153]:


# Negative predictive value
TN / float(TN+ FN)


# In[154]:


#Lets check with the ROC Curve
def draw_roc( actual, probs ):
    fpr, tpr, thresholds = metrics.roc_curve( actual, probs,
                                              drop_intermediate = False )
    auc_score = metrics.roc_auc_score( actual, probs )
    plt.figure(figsize=(5, 5))
    plt.plot( fpr, tpr, label='ROC curve (area = %0.2f)' % auc_score )
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate or [1 - True Negative Rate]')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()
    return None


# In[155]:


# Performance of a classification model
fpr, tpr, thresholds = metrics.roc_curve( y_train_pred_final.Converted, y_train_pred_final.Converted_prob, drop_intermediate = False )


# In[156]:


# Performance of a classification model
draw_roc(y_train_pred_final.Converted, y_train_pred_final.Converted_prob)


# - The ROC Curve should be close to 1. We receive a nice value of 0.97, which refers to a good prediction model.

# In[ ]:


# Let's find out the Optimal Cutoff point


# We picked an arbitrary cut-off value of 0.5 earlier. We must identify the best cut-off value, which is covered in the following section:

# In[157]:


# Let's create columns with different probability cutoffs 
numbers = [float(x)/10 for x in range(10)]
for i in numbers:
    y_train_pred_final[i]= y_train_pred_final.Converted_prob.map(lambda x: 1 if x > i else 0)
y_train_pred_final.head()


# In[159]:


# Now let's calculate accuracy sensitivity and specificity for various probability cutoffs.
cutoff_df = pd.DataFrame( columns = ['probability','accuracy','sensitivity','specificity'])
from sklearn.metrics import confusion_matrix
num = [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
for i in num:
    cm1 = metrics.confusion_matrix(y_train_pred_final.Converted, y_train_pred_final[i] )
    total1=sum(sum(cm1))
    accuracy = (cm1[0,0]+cm1[1,1])/total1    
    specificity = cm1[0,0]/(cm1[0,0]+cm1[0,1])
    sensitivity = cm1[1,1]/(cm1[1,0]+cm1[1,1])
    cutoff_df.loc[i] =[ i ,accuracy,sensi,speci]
cutoff_df


# In[160]:


# Let's plot accuracy of sensitivity and specificity for various probabilities.
cutoff_df.plot.line(x='probability', y=['accuracy','sensitivity','specificity'])
plt.show()


# In[161]:


# According to the curve above, 0.3 is the best point to use as a cutoff probability.
y_train_pred_final['final_Predicted'] = y_train_pred_final.Converted_prob.map( lambda x: 1 if x > 0.3 else 0)
y_train_pred_final.head()


# In[162]:


y_train_pred_final['Lead_Score'] = y_train_pred_final.Converted_prob.map( lambda x: round(x*100))
y_train_pred_final[['Converted','Converted_prob','Prospect ID','final_Predicted','Lead_Score']].head()


# In[163]:


# Let's check the overall accuracy again.
metrics.accuracy_score(y_train_pred_final.Converted, y_train_pred_final.final_Predicted)


# In[165]:


# Let's go ahead with second confusion matrix
confusion_2 = metrics.confusion_matrix(y_train_pred_final.Converted, y_train_pred_final.final_Predicted )
confusion_2


# In[166]:


# True Positive 
TP = confusion_2[1,1]
# True Negatives
TN = confusion_2[0,0] 
# False Positives
FP = confusion_2[0,1]
# False Negatives
FN = confusion_2[1,0] 


# In[167]:


# Repeating the process again sensitivity 
TP/float(TP+FN)


# In[168]:


#specificity
TN/float(TN+FP)


# Observation:
# As we can see above, the model appears to be working well. The ROC curve has a value of 0.97, which is good. The Train Data has the following values:
# - Accuracy : 90.81%
# - Sensitivity : 92.05%
# - Specificity : 90.10%
# 
# Some of the additional statistics are shown below, including the False Positive Rate, Positive Predictive Value, Negative Predictive Values, Precision, and Recall.

# In[169]:


#False Postive Rate
FP/float(TN+FP)


# In[171]:


# Positive predictive value 
TP/float(TP+FP)


# In[173]:


# Negative predictive value
TN/float(TN+ FN)


# In[174]:


#Again Looking at the confusion matrix
confusion = metrics.confusion_matrix(y_train_pred_final.Converted, y_train_pred_final.final_Predicted )
confusion


# In[175]:


# Precision
TP/TP+FP


# In[176]:


confusion[1,1]/(confusion[0,1]+confusion[1,1])


# In[177]:


# Recall
TP/TP+FN


# In[178]:


confusion[1,1]/(confusion[1,0]+confusion[1,1])


# In[179]:


#importing precision_score, recall_score library
from sklearn.metrics import precision_score, recall_score


# In[180]:


precision_score(y_train_pred_final.Converted , y_train_pred_final.final_Predicted)


# In[181]:


recall_score(y_train_pred_final.Converted, y_train_pred_final.final_Predicted)


# In[182]:


#importing precision recall curve library
from sklearn.metrics import precision_recall_curve


# In[183]:


y_train_pred_final.Converted, y_train_pred_final.final_Predicted
p, r, thresholds = precision_recall_curve(y_train_pred_final.Converted, y_train_pred_final.Converted_prob)


# In[184]:


plt.plot(thresholds, p[:-1], "g-")
plt.plot(thresholds, r[:-1], "r-")
plt.show()


# In[185]:


#scaling test dataset
numerical_columns=X_test.select_dtypes(include=['float64', 'int64']).columns
X_test[numerical_columns] = scaler.fit_transform(X_test[numerical_columns])
X_test.head()


# In[186]:


X_test = X_test[selected_features]
X_test.head()


# In[187]:


X_test_sm = sm.add_constant(X_test)


# In[189]:


#Let's predict on Test Dataset
y_test_pred = res.predict(X_test_sm)
y_test_pred[:10]


# In[191]:


# Converting y_pred to a dataframe which is an array
y_pred_1 = pd.DataFrame(y_test_pred)
y_pred_1.head()


# In[192]:


# Converting y_test to dataframe
y_test_df = pd.DataFrame(y_test)


# In[193]:


# Putting CustID to index
y_test_df['Prospect ID'] = y_test_df.index


# In[194]:


# Removing the index from each dataframes in order to append them side by side.
y_pred_1.reset_index(drop=True, inplace=True)
y_test_df.reset_index(drop=True, inplace=True)


# In[196]:


# Appending y_test_df and y_pred_1
y_pred_final = pd.concat([y_test_df, y_pred_1],axis=1)
y_pred_final.head()


# In[198]:


# Renaming the column 
y_pred_final= y_pred_final.rename(columns={ 0 : 'Converted_prob'})
y_pred_final.head()


# In[199]:


# Rearranging the columns
y_pred_final = y_pred_final[['Prospect ID','Converted','Converted_prob']]
y_pred_final['Lead_Score'] = y_pred_final.Converted_prob.map( lambda x: round(x*100))
y_pred_final.head()


# In[200]:


y_pred_final['final_Predicted'] = y_pred_final.Converted_prob.map(lambda x: 1 if x > 0.3 else 0)
y_pred_final.head()


# In[201]:


# Finally let check the overall accuracy.
metrics.accuracy_score(y_pred_final.Converted, y_pred_final.final_Predicted)


# In[203]:


confusion_3 = metrics.confusion_matrix(y_pred_final.Converted, y_pred_final.final_Predicted )
confusion_3


# In[204]:


# True Positive 
TP = confusion_3[1,1]
# True Negatives
TN = confusion_3[0,0] 
# False Positives
FP = confusion_3[0,1]
# False Negatives
FN = confusion_3[1,0]


# In[205]:


#sensitivity
TP/float(TP+FN)


# In[206]:


#specificity
TN/float(TN+FP)


# In[207]:


precision_score(y_pred_final.Converted , y_pred_final.final_Predicted)


# In[208]:


recall_score(y_pred_final.Converted, y_pred_final.final_Predicted)


# Observation:
# 
# The following figures are obtained after running the model on the Test Data:
# - Specificity: 90.62% 
# - Accuracy: 90.92%
# - Sensitivity: 91.41%
# Finally, consider the following:
# Let us compare the results for Train and Test:
# Train Data:
# - Accuracy : 90.81%
# - Sensitivity : 92.05% 
# - specificity : 90.10%
# Test Data:
# - Accuracy : 90.92%
# - Sensitivity : 91.41%
# - Specificity : 90.62%
# 
# 
# The Model appears to accurately anticipate the Conversion Rate, and we should be able to give the CEO confidence in making appropriate decisions based on this model.
