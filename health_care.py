# import os
# import numpy as np
# import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt
#
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import LabelEncoder
# from sklearn.preprocessing import OneHotEncoder
#
#
# ## loading the file
# health_care = pd.read_csv('./healthcare/train_data.csv')
#
#
#
#
# """
# The columns to be pre-processed from string to numerical values are :
# 1. Hospital_type_code
# 2. Hospital_region_code
# 3. Department
# 4. Ward_Type
# 5. Ward_Facility_Code
# 6. City_Code_Patient
# 7. Type of Admission
# 8. Severity of Illness
# 9. Age
# 10. Admission_Deposit
# 11. Stay
# """
#
# ## Pre processing these columns
#
# string_columns = ['Hospital_type_code', 'Hospital_region_code', 'Department', 'Ward_Type', 'Ward_Facility_Code', 'City_Code_Patient', 'Type of Admission', 'Severity of Illness', 'Age',  'Stay' ]
# encoder = LabelEncoder()
#
# ## label_encoded data
# X_le = health_care.drop(['Stay', 'case_id', 'patientid'], axis=1)
# y_le = health_care['Stay']
#
# for column in string_columns :
#     if column == 'Stay' :
#         y_le = encoder.fit_transform(y_le)
#     else :
#         X_le[column] = encoder.fit(X_le[column]).transform(X_le[column])
#
# ## getting the inputs and labels
#
# X = X_le.values
# y = y_le


#TS_CW_draft
import os
import pandas as pd
import numpy as np
import sklearn as sk
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
%matplotlib inline


# Importing the data set
path = "healthcare"

filename_read = os.path.join(path, "train_data.csv")
df = pd.read_csv(filename_read, na_values=['NA', '?'])

print(df.shape)
print(df.columns)
df.head(10)

# Exploratory data analysis
#creating a copy of df to have an original for further manipulation
new_df = df.copy()
new_df = new_df.select_dtypes(include=["int","float"])

headers = list(new_df.columns.values)

fields = []
for field in headers:
    fields.append({
        "name":field,
        "mean":df[field].mean(),
        "var":df[field].var(),
        "sdev":df[field].std()
    })
for field in fields:
    print(field)

#????here should an analysis of columns means and st dev

#Missing Values
#checking for missing values
df.isnull().values.any()
#Checking number of NANs for each column, in order to understand how many missing values there are in a dataframe.
print("# of NaN in each columns:", df.isnull().sum(), sep='\n')

"""
To make sure we consider all the correct features to make an accurate prediction, it may be useful to create some plots to have a better understanding of our data.
We will be using the Python data visualisation library seaborn.
We could begin by visualising the length of stays by type of admissions etc., by using a countplot(), that shows the counts of observations in each categorical bin using bars.
"""

#sorting by Stay for better representation in the visualisations
df= df.sort_values(by = "Stay", ascending = True)
#visualising the length of stays by age
#plot size
plt.figure(figsize = (15,4))
#plot title
plt.title("Age", fontdict = {'fontsize':15})
ax = sns.countplot(x = "Age", hue = 'Stay', data = df)
#?? comments
"""
the age group risk - 21-80y.o.
"""

#visualising the length of stays by type of admissions
#plot size
plt.figure(figsize = (15,4))
#plot title
plt.title("Type of Admission", fontdict = {'fontsize': 15})
ax = sns.countplot(x = "Type of Admission", hue = 'Stay', data = df)
"""???? comments on the graph
Emergency and Trauma and mostly from 11 - 40 and range 51-60
Next, it may be interesting to look at the Severity of Illness segments by stay.
Again, we can use a count plot to visualise this.'
""""

#visualising the length of stays by Severity of Illness
#plot size
plt.figure(figsize = (15,4))
#plot title
plt.title("Severity of Illness", fontdict = {'fontsize':15})
ax = sns.countplot(x = "Severity of Illness", hue = 'Stay', data = df)

"""
comments on the graph:
mostly moderate and minor, and in age range 11-40 and range 51-60
"""

#visualising the length of stays by Available Extra Rooms in Hospital
#plot size
plt.figure(figsize = (20,4))
#plot title
plt.title("Available Extra Rooms in Hospital", fontdict = {'fontsize':15})
ax = sns.countplot(x = "Available Extra Rooms in Hospital", hue = 'Stay', data = df)

"""
comments on the graph
most rooms have additional 2-4 beds in the room available.
"""
#visualising the length of stays by Department
#plot size
plt.figure(figsize = (15,8))
#plot title
plt.title("Department", fontdict = {'fontsize':15})
ax = sns.countplot(x = "Department", hue = 'Stay', data = df)

"""
mostly gynecology department
"""

#Preparing our data
"""
We saw in that columns "Bed Grade" and "City_Code_patient" have missing values. 
Before deciding whether we shall drop these columns or keep some of them and try to fill the missing values with medians, 
we can make and analysis to understand the percentage of missing values by creating this function "perc_mv".
"""

def perc_mv(x, y):
    perc = y.isnull().sum() / len(x) * 100
    return perc

print('Missing value ratios:\nBed Grade: {}\nCity_Code_Patient: {}'.format(
    perc_mv(df, df['Bed Grade']),
    perc_mv(df, df['City_Code_Patient'])))

"""
As we can see, 0.03% of Bed Grade column and 1.42% of City_Code_Patient has missing values. 
Therefore, we have enough values to fill the rows of the company column via any imputation method.
"""

# Assuming importance of the City_Code_Patient in the future analysis is minimalistic
# let s drop it

df = df.drop(['City_Code_Patient'], axis = 1)

#let's fill missing values of 'Bed Grade' with its median of the column
med = df['Bed Grade'].median()
df['Bed Grade'] = df['Bed Grade'].fillna(med)

#missing values check after data manipulation
df.isnull().values.any()

#Next, we may want to check the features to create some more meaningful variables and reduce the number of features.
#????think of smth to work on the type
#In the code cell below, we use the attribute dtype on df to retrieve the data type for each column.
print (df.dtypes)

#We want to map the name of each Severity of Illness to the corresponding number in ascending order.
df['Severity of Illness'] = df['Severity of Illness'].map({'Minor':1, 'Moderate': 2, 'Extreme':3})
df['Type of Admission'] = df['Type of Admission'].map({'Trauma':1, 'Emergency': 2, 'Urgent':3})
df

"""
For now, we assume these columns are not important for our analysis:
0case_id
1Hospital_code
2Hospital_type_code
3City_Code_Hospital
7Ward_Type - anyways we know department, so we can guess the ward type
8Ward_Facility_Code
"""
#dropping unwanted columns
df.drop(df.columns[0:4],axis=1, inplace=True)
df.drop(df.columns[3:5], axis=1, inplace=True)
print(df.shape)
#df
#print (df.dtypes)

#using LabelEncoder to change and transform the object format of 'Hospital_region_code','Department','Age', 'Stay' columns
le = LabelEncoder()
for col in ['Hospital_region_code','Department','Age', 'Stay']:
    df[col]= df[col].astype('str')
    df[col]= le.fit_transform(df[col])


#shuffling
df= df.reindex(np.random.permutation(df.index))
df.reset_index(inplace=True, drop=True)

#Correlation
"""
One way to decide which features are going to be more impactful when running an analysis is to look at the correlation between them.
Pearson's correlations are single value numerical summaries that represent the strength and direction of a linear relationship. 
Correlation values range from -1 to 1. Values further away from 0 represent stronger relationships, and the sign of the correlation (positive or negative) 
represents the direction of the relationship. 
The graphs below depict a visual representation of Pearson correlations.
"""
#creating a copy of df to have an original for further manipulation
cor_df = df.copy()
le = LabelEncoder()
cor_df['Stay'] = le.fit_transform(cor_df['Stay'])


#use corr() on cor_df
cor_df.corr()
#Because we want to try to predict the likelihood of a length of stay depending on the type of admission, it is useful to visualise the sorted correlation values for the column Type of Admission.
# here, we are appling sort_values() to visualise the correlation values in a descending order.
cor_df.corr()['Type of Admission'].sort_values(ascending=False)

#let's examine likelihood of a length of stay depending on the Severity of Illness.
cor_df.corr()['Severity of Illness'].sort_values(ascending=False)
#???What is the most impactful feature?


"""
Splitting our data for modelling
"""
#using from sklearn.model_selection import train_test_split model
X = df.drop(columns=["Stay"])
y = df["Stay"]
print(X.shape)
print(y.shape)
#test size = 20%, train size = 80%
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)
