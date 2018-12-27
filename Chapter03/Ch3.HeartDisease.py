import pandas as pd

#Import data
HDNames= ['age','sex','cp','trestbps','chol','fbs','restecg','thalach','exang','oldpeak','slope','ca','hal','HeartDisease']
Data = pd.read_excel('ClevelandData.xlsx', names=HDNames)


print(Data.head(20))
print(Data.info())
summary = Data.describe()
print(summary)


#Removing missing values
import numpy as np

DataNew = Data.replace('?', np.nan)

print(DataNew.info())

print(DataNew.describe())


print(DataNew.isnull().sum())

DataNew = DataNew.dropna()

print(DataNew.info())

print(DataNew.isnull().sum())



#Divide DataFrame
InputNames = HDNames
InputNames.pop()
Input = pd.DataFrame(DataNew.iloc[:, 0:13],columns=InputNames)

Target = pd.DataFrame(DataNew.iloc[:, 13],columns=['HeartDisease'])

#Data scaling
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
print(scaler.fit(Input))
InputScaled = scaler.fit_transform(Input)


InputScaled = pd.DataFrame(InputScaled,columns=InputNames)

summary = InputScaled.describe()
summary = summary.transpose()
print(summary)

#Data visualitation
#DataScaled = pd.concat([InputScaled, Target], axis=1)

import matplotlib.pyplot as plt
boxplot = InputScaled.boxplot(column=InputNames,showmeans=True)
plt.show()

pd.plotting.scatter_matrix(InputScaled, figsize=(6, 6))
plt.show()

CorData = InputScaled.corr(method='pearson')

with pd.option_context('display.max_rows', None, 'display.max_columns', CorData.shape[1]):
    print(CorData)

plt.matshow(CorData)
plt.xticks(range(len(CorData.columns)), CorData.columns)
plt.yticks(range(len(CorData.columns)), CorData.columns)
plt.colorbar()
plt.show()

#Split the data
from sklearn.model_selection import train_test_split

Input_train, Input_test, Target_train, Target_test = train_test_split(InputScaled, Target, test_size = 0.30, random_state = 5)
print(Input_train.shape)
print(Input_test.shape)
print(Target_train.shape)
print(Target_test.shape)

from keras.models import Sequential
from keras.layers import Dense

model = Sequential()
model.add(Dense(30, input_dim=13, activation='tanh'))
model.add(Dense(20, activation='tanh'))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
model.fit(Input_train, Target_train, epochs=1000, verbose=1)

model.summary()

score = model.evaluate(Input_test, Target_test, verbose=0)

print('Keras Model Accuracy = ',score[1])

Target_Classification = model.predict(Input_test)
Target_Classification = (Target_Classification > 0.5)



from sklearn.metrics import confusion_matrix

print(confusion_matrix(Target_test, Target_Classification))