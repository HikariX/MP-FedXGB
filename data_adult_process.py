import pandas as pd
import numpy as np

training_data = './adult.data'
columns = ['Age','Workclass','fnlwgt','Education','EdNum','MaritalStatus',
           'Occupation','Relationship','Race','Sex','CapitalGain',
           'CapitalLoss','HoursPerWeek','Country','Income']
income = pd.read_csv(training_data, names=columns)
income.dropna(inplace=True)
income['Income'].replace(' <=50K', 0,inplace=True)
income['Income'].replace(' >50K', 1,inplace=True)
y = income['Income']
temp = income.iloc[:, :-1]
# 将文本转换为数值型用于拟合模型
income_=pd.get_dummies(temp,columns=['Relationship','Sex','MaritalStatus','Workclass',
                        'Education','Country','Occupation','Race'])

income = np.concatenate([y.values.reshape(-1, 1), income_.values,], axis=1)
print(income.shape)
np.save('adult.npy', income)