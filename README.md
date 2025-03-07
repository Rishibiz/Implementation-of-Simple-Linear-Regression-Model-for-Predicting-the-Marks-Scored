# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
#### 1.Import the standard Libraries. 
#### 2.Set variables for assigning dataset values. 
#### 3.Import linear regression from sklearn. 
#### 4.Assign the points for representing in the graph. 
#### 5.Predict the regression for marks by using the representation of the graph. 
#### 6.Compare the graphs and hence we obtained the linear regression for the given datas.

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: Rishi chandran R
RegisterNumber:  212223043005
*/
```
```
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error,mean_squared_error
import matplotlib.pyplot as plt

dataset=pd.read_csv('student_scores.csv')
print(dataset.head())
print(dataset.tail())
x=dataset.iloc[:,:-1].values
print(x)
y=dataset.iloc[:,1].values
print(y)

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
reg=LinearRegression()
reg.fit(x_train,y_train)
y_pred = reg.predict(x_test)
print(y_pred)
print(y_test)

plt.scatter(x_train,y_train,color='purple')
plt.plot(x_train,reg.predict(x_train),color='black')
plt.title("Hours vs Scores(Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

plt.scatter(x_test,y_test,color='red')
plt.plot(x_train,reg.predict(x_train),color='black')
plt.title("Hours vs Scores(Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

mse=mean_absolute_error(y_test,y_pred)
print('Mean Square Error = ',mse)
mae=mean_absolute_error(y_test,y_pred)
print('Mean Absolute Error = ',mae)
rmse=np.sqrt(mse)
print("Root Mean Square Error = ",rmse) 

```

## Output:

## HEAD VALUES
![WhatsApp Image 2025-03-07 at 09 12 45_6b931b67](https://github.com/user-attachments/assets/a42ee2ab-6308-4073-99be-cc8e546653e0)

## TAIL VALUES
![image](https://github.com/user-attachments/assets/90cfd7eb-86c8-4687-a62a-6278f2fe36d7)

## COMPARE DATASET
![image](https://github.com/user-attachments/assets/413f16c3-5eb9-4ae5-afb2-cb4cef40de24)

## Predication values of X and Y
![image](https://github.com/user-attachments/assets/0f4a3728-8271-4898-80b0-7c3955e061de)

## Training set
![image](https://github.com/user-attachments/assets/1b73484b-2564-45e6-83de-432a25729223)

## Testing Set
![image](https://github.com/user-attachments/assets/d057a5ce-8623-48a5-8189-aca59f3b9ec6)

## MSE,MAE and RMSE
![image](https://github.com/user-attachments/assets/cc6f9fe8-77af-4684-be1c-d02198cf0e93)

## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
