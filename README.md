# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the required libraries.

2.Load the dataset and check for null data values and duplicate data values in the dataframe.

3.Import label encoder from sklearn.preprocessing to encode the dataset.

4.Apply Logistic Regression on to the model.

5.Predict the y values.

6.Calculate the Accuracy,Confusion and Classsification report.


## Program:
```
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: SANDHIYA R
RegisterNumber:  212222230129
```
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
df=pd.read_csv("Placement_Data.csv")
df
df.head()
df.tail()
df=df.drop(['sl_no','gender','salary'],axis=1)
df=df.drop(['ssc_b','hsc_b'],axis=1)
df.shape
df.info()
df["degree_t"]=df["degree_t"].astype("category")
df["hsc_s"]=df["hsc_s"].astype("category")
df["workex"]=df["workex"].astype("category")
df["status"]=df["status"].astype("category")
df["specialisation"]=df["specialisation"].astype("category")
df["degree_t"]=df["degree_t"].cat.codes
df["hsc_s"]=df["hsc_s"].cat.codes
df["workex"]=df["workex"].cat.codes
df["status"]=df["status"].cat.codes
df["specialisation"]=df["specialisation"].cat.codes
x=df.iloc[: ,:-1].values
y=df.iloc[:,- 1].values
y
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)

df.head()
from sklearn.linear_model import LogisticRegression

#printing its accuracy
clf=LogisticRegression()
clf.fit(x_train,y_train)
clf.score(x_test,y_test)
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(solver = "liblinear") 
lr.fit(x_train,y_train)
y_pred = lr.predict(x_test)
y_pred

from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test,y_pred)
accuracy

from sklearn.metrics import confusion_matrix
confusion = (y_test,y_pred)
confusion 

from sklearn.metrics import classification_report
classification_report1 = classification_report(y_test,y_pred)
print(classification_report1)
# Predicting for random value
clf.predict([[1	,78.33,	1,	2,	77.48,	2,	86.5,	0,	66.28]])
```

## Output:

### Dataset
![image](https://github.com/SandhiyaR1/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/113497571/46679c99-ac41-48eb-a089-37d536847389)

### Dropping the unwanted columns

![image](https://github.com/SandhiyaR1/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/113497571/a39e88df-4cc6-4cff-b8ac-28b9006bf78d)


### df.info()

![image](https://github.com/SandhiyaR1/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/113497571/4641415b-00ec-4056-a04e-86686f0e7d41)

### df.info() after changing object into category

![image](https://github.com/SandhiyaR1/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/113497571/a663d998-c3bf-4d3a-950c-a06212808717)

### df.info() after changing into integer

![image](https://github.com/SandhiyaR1/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/113497571/b044bd5a-7060-4817-b6ab-95df52f5fc47)

### selecting features and lables

![image](https://github.com/SandhiyaR1/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/113497571/4b7e1797-872e-4bad-be09-bb6456dee85a)

### Training and testing

![image](https://github.com/SandhiyaR1/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/113497571/298da98f-f1f6-4135-acfe-72a1dc1f148e)

### Creating a Classifier using Sklearn:

![image](https://github.com/SandhiyaR1/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/113497571/563ba84b-c13b-4cc2-9b3b-04bdfa9834ef)

###  confusion matrix and classification report, 

![image](https://github.com/SandhiyaR1/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/113497571/b218ab95-a3a4-4082-ae09-fae1f2f0579b)


### Predicting for random value:

![image](https://github.com/SandhiyaR1/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/113497571/648f82a5-0837-4069-8ead-70c0aaac1d6b)

## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
