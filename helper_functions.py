import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

def preprocess(data_all):
    data_all['Age'].fillna(data_all['Age'].median(), inplace = True)

    #complete embarked with mode
    data_all['Embarked'].fillna(data_all['Embarked'].mode()[0], inplace = True)

    #complete missing fare with median
    data_all['Fare'].fillna(data_all['Fare'].median(), inplace = True)

    #delete the cabin feature/column and others previously stated to exclude in train dataset
    drop_column = ['PassengerId','Cabin', 'Ticket']
    data_all.drop(drop_column, axis=1, inplace = True)

    #Discrete variables
    data_all['FamilySize'] = data_all['SibSp'] + data_all['Parch'] + 1

    data_all['IsAlone'] = 1 #initialize to yes/1 is alone
    data_all['IsAlone'].loc[data_all['FamilySize'] > 1] = 0 # now update to no/0 if family size is greater than 1

    #quick and dirty code split title from name: http://www.pythonforbeginners.com/dictionary/python-split
    data_all['Title'] = data_all['Name'].str.split(", ", expand=True)[1].str.split(".", expand=True)[0]


    #Continuous variable bins; qcut vs cut: https://stackoverflow.com/questions/30211923/what-is-the-difference-between-pandas-qcut-and-pandas-cut
    #Fare Bins/Buckets using qcut or frequency bins: https://pandas.pydata.org/pandas-docs/stable/generated/pandas.qcut.html
    data_all['FareBin'] = pd.qcut(data_all['Fare'], 4)

    #Age Bins/Buckets using cut or value bins: https://pandas.pydata.org/pandas-docs/stable/generated/pandas.cut.html
    data_all['AgeBin'] = pd.cut(data_all['Age'].astype(int), 5)

    #cleanup rare title names
    #print(data1['Title'].value_counts())
    stat_min = 10 #while small is arbitrary, we'll use the common minimum in statistics: http://nicholasjjackson.com/2012/03/08/sample-size-is-10-a-magic-number/
    title_names = (data_all['Title'].value_counts() < stat_min) #this will create a true false series with title name as index

    #apply and lambda functions are quick and dirty code to find and replace with fewer lines of code: https://community.modeanalytics.com/python/tutorial/pandas-groupby-and-python-lambda-functions/
    data_all['Title'] = data_all['Title'].apply(lambda x: 'Misc' if title_names.loc[x] == True else x)

    #code categorical data
    label = LabelEncoder()
    data_all['Sex_Code'] = label.fit_transform(data_all['Sex'])
    data_all['Embarked_Code'] = label.fit_transform(data_all['Embarked'])
    data_all['Title_Code'] = label.fit_transform(data_all['Title'])
    data_all['AgeBin_Code'] = label.fit_transform(data_all['AgeBin'])
    data_all['FareBin_Code'] = label.fit_transform(data_all['FareBin'])

    return data_all