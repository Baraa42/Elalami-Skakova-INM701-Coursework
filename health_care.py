import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder


## loading the file
health_care = pd.read_csv('./healthcare/train_data.csv')




"""
The columns to be pre-processed from string to numerical values are :
1. Hospital_type_code
2. Hospital_region_code
3. Department
4. Ward_Type
5. Ward_Facility_Code
6. City_Code_Patient
7. Type of Admission
8. Severity of Illness
9. Age
10. Admission_Deposit
11. Stay
"""

## Pre processing these columns

string_columns = ['Hospital_type_code', 'Hospital_region_code', 'Department', 'Ward_Type', 'Ward_Facility_Code', 'City_Code_Patient', 'Type of Admission', 'Severity of Illness', 'Age',  'Stay' ]
encoder = LabelEncoder()

## label_encoded data
X_le = health_care.drop(['Stay', 'case_id', 'patientid'], axis=1)
y_le = health_care['Stay']

for column in string_columns :
    if column == 'Stay' :
        y_le = encoder.fit_transform(y_le)
    else : 
        X_le[column] = encoder.fit(X_le[column]).transform(X_le[column])

## getting the inputs and labels

X = X_le.values
y = y_le







