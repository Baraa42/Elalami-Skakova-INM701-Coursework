# INM701-Coursework

## Postgraduate students: Tamila Skakova and Elbaraa Elalami.


## Introduction
We have decided to study a healthcare industry problem for our project found on Kaggle - "Healthcare Analytics II" (hackathon). The question we set for our investigation is to predict the length of stay of patients in hospitals, which would increase efficiency of healthcare management. Our strategy commences with study of dataset’s features and target: which features have major impact on prediction, statistical analysis of the target, existence of correlation and structure of features (imbalanced data). The prediction of these problems is very critical, especially in time of a pandemic: it can help in allocation of beds to patients in hospitals, according to their recovery period for the specific department, type of admission, severity of illness and other important features.
Our dataset contains 318,438 records of patients, each comprising 18 representational features. Our expected output feature contains 11 different classes, each of them is a length of stay at the hospital, ranging from 0-10 days to more than 100 days.
Healthcare dataset can be trained as both classification and regression model. In this coursework, we study both methods using: K-Nearest Neighbours, Decision Trees, Random Forest, Naive Bayes, Artificial Neural Network with different number of hidden layers, Logistic Regression, Linear Regression and Lasso Regression models. As an addition to our analysis, for the comparison purposes we have studied and implemented Gradient Boost, CatBoost and XGB Boost Classifiers.


## Link to a dataset:
Kaggle (2020) AV: Healthcare Analytics II. Analytics Vidhya Hackathon about Healthcare Analytics. Analytics Vidhya: Kaggle. <https://www.kaggle.com/nehaprabhavalkar/av-healthcare-analytics-ii>


## Content:
- **[Part 1](#part1)- Importing the data set, packages used**
- **[Part 2](#part2)- Exploratory data analysis**
- [Part 2.1](#part2.1)- Analysis of the features
- [Part 2.2](#part2.2)- Analysis of the Target
- [Part 2.3](#part2.3)- Statistical analysis of the dataset
- **[Part 3](#part3) -  Preparing our data**
- [Part 3.1](#part3.1) -  Missing Values
- [Part 3.2](#part3.2) -  Enconding, Shuffling, Scaling
- [Part 3.3](#part3.3) -  PCA
- [Part 3.4](#part3.4) -  SMOTE Analysis
- **[Part 4/1](#part4/1) -  Models**
- [Part 4.1.1](#part4.1.1) -  Score method
- [Part 4.1.2](#part4.1.2) -  KNN
- [Part 4.1.3](#part4.1.3) -  Decision tree model
- [Part 4.1.4](#part4.1.4) -  Random Forest
- [Part 4.1.5](#part4.1.5) -  Naive Bayes
- [Part 4.1.6](#part4.1.6) -  ANN
- **[Part 4/2](#part4/2) -  Models with SMOTE**
- [Part 4.2.1](#part4.2.1) -  KNN
- [Part 4.2.2](#part4.2.2) -  Decision tree model 
- [Part 4.2.3](#part4.2.3) -  Random Forest
- [Part 4.2.4](#part4.2.4) -  Naive Bayes
- [Part 4.2.5](#part4.2.5) -  ANN
- **[Part 5](#part5) -  Additional Models**
- [Part 5.1](#part5.1) -  Gradient Boost Classifier
- [Part 5.2](#part5.2) -  CatBoost Classifier
- [Part 5.2](#part5.2) -  XGB BOOST Classifier


## Description of features in the dataframe:

- `Column`	Description
- `case_id`	Case_ID registered in Hospital
- `Hospital_code`	Unique code for the Hospital
- `Hospital_type_code`	Unique code for the type of Hospital
- `City_Code_Hospital` City Code of the Hospital
- `Hospital_region_code`	Region Code of the Hospital
- `Available Extra Rooms in Hospital`	Number of Extra rooms available in the Hospital
- `Department`	Department overlooking the case
- `Ward_Type`	Code for the Ward type
- `Ward_Facility_Code`	Code for the Ward Facility
- `Bed Grade`	Condition of Bed in the Ward
- `patientid`	Unique Patient Id
- `City_Code_Patient`	City Code for the patient
- `Type of Admission`	Admission Type registered by the Hospital
- `Severity of Illness`	Severity of the illness recorded at the time of admission
- `Visitors with Patient`	Number of Visitors with the patient
- `Age`	Age of the patient
- `Admission_Deposit`	Deposit at the Admission Time
- `Stay`	Stay Days by the patient, the length of stay - 11 different classes ranging from 0-10 days to more than 100 days.
