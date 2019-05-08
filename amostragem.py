#desafio titanic

import pandas as pd
import seaborn as sns
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn import metrics
import matplotlib.pyplot as plt
import csv
from sklearn.externals.six import StringIO
import pydotplus
import matplotlib.image as mpimg
from sklearn import tree

df_test = pd.read_csv('test.csv', delimiter=",")
df = pd.read_csv('train.csv', delimiter=",")
#print(df['Survived'].value_counts())
#print(df.size)

#Associando as colunas necessárias para a predição para as variáveis X e Y
X = np.array(df[['Pclass', 'Sex']])
Y = df[['Survived']]
X_test = np.array(df_test[['Pclass','Sex']])

#Trantando as variáveis categóricas de X
le_sex = preprocessing.LabelEncoder()
le_sex.fit(['female', 'male'])
X[:, 1] = le_sex.transform(X[:, 1])
X_test[:, 1] = le_sex.transform(X_test[:, 1])

#Definindo os testes e os treinos em 30% da database
X_trainset, X_testset, y_trainset, y_testset = train_test_split(X, Y, test_size=0.3, random_state=3)
arvore = DecisionTreeClassifier(criterion="entropy", max_depth=4)

arvore.fit(X_trainset, y_trainset)
arvore.predict(X_testset)

#Calcular a precisão do modelo
print("Precisão em porcentagem: ", metrics.accuracy_score(Y_test, predTree)*100)

#Testar
predTree = arvore.predict(X_test)

#Armazenar a predição em um arquivo CSV (ID,sobreviveu)
count = 0
with open("submition.csv",'w',newline='') as csv_file:
    fieldnames = ['PassengerId', 'Survived']
    writer = csv.writer(csv_file)
    X = np.array(df_test['PassengerId'])
    writer.writerow(fieldnames)
    for i in predTree:
        dado = [X[count],i]
        print(dado)
        writer.writerow(dado)
        count += 1

