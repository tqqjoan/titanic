#-*- coding:UTF-8 -*-
import pandas as pd
import numpy as np
from pandas import Series,DataFrame
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor,GradientBoostingClassifier,RandomForestClassifier
from sklearn import preprocessing
from sklearn import linear_model
from sklearn import cross_validation
import math

class analysis(object):
    def __init__(self,addr):
        self.data_train = pd.read_csv(addr)
        # print self.data_train
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False
    def survive(self):
        fig = plt.figure()
        fig.set(alpha = 0.2)

        plt.subplot2grid((2,3),(0,0))
        self.data_train.Survived.value_counts().plot(kind = 'bar')
        plt.title(u"获救情况（1为获救)")
        plt.ylabel(u"人数")

        plt.subplot2grid((2,3),(0,1))
        self.data_train.Pclass.value_counts().plot(kind = "bar")
        plt.title(u"乘客等级分布")
        plt.ylabel(u"人数")

        plt.subplot2grid((2,3),(0,2))
        plt.scatter(self.data_train.Survived,self.data_train.Age)
        plt.title(u"按年龄看获救分布（1获救）")
        plt.ylabel(u"年龄")

        plt.subplot2grid((2,3),(1,0),colspan = 2)
        self.data_train.Age[self.data_train.Pclass == 1].plot(kind ="kde")
        self.data_train.Age[self.data_train.Pclass == 2].plot(kind ="kde")
        self.data_train.Age[self.data_train.Pclass == 3].plot(kind ="kde")
        plt.title(u"乘客年龄分布")
        plt.xlabel(u"年龄")
        plt.ylabel(u"密度")
        plt.legend((u"头等舱",u"2等舱",u"3等舱"),loc = "best")

        plt.subplot2grid((2, 3), (1, 2))
        self.data_train.Embarked.value_counts().plot(kind='bar')
        plt.title(u"各登船口岸上船人数")
        plt.ylabel(u"人数")
        plt.show()
    def data_dealing(self):
        age_df = self.data_train[['Age','Fare','Parch','SibSp','Pclass']]
        # print age_df
        known_age = age_df[age_df.Age.notnull()].as_matrix()
        unknown_age = age_df[age_df.Age.isnull()].as_matrix()

        y = known_age[:,0]
        x = known_age[:,1:]

        rfr = RandomForestRegressor(random_state=0, n_estimators=2000, n_jobs=-1)
        rfr.fit(x,y)
        pred = rfr.predict(unknown_age[:,1:])
        self.data_train.loc[(self.data_train.Age.isnull()),'Age'] = pred

        self.data_train.loc[(self.data_train.Cabin.notnull()),'Cabin'] = "YES"
        self.data_train.loc[(self.data_train.Cabin.isnull()),'Cabin'] = "NO"
        # print self.data_train
        dummies_Cabin = pd.get_dummies(self.data_train['Cabin'], prefix='Cabin')
        dummies_Embarked = pd.get_dummies(self.data_train['Embarked'], prefix='Embarked')

        dummies_Sex = pd.get_dummies(self.data_train['Sex'], prefix='Sex')

        dummies_Pclass = pd.get_dummies(self.data_train['Pclass'], prefix='Pclass')
        df = pd.concat([self.data_train, dummies_Cabin, dummies_Embarked, dummies_Sex, dummies_Pclass], axis=1)
        df.drop(['Pclass', 'Name', 'Sex', 'Ticket', 'Cabin', 'Embarked'], axis=1, inplace=True)

        scaler = preprocessing.StandardScaler()
        age_scale_param = scaler.fit(df['Age'])
        df['Age_scaled'] = scaler.fit_transform(df['Age'],age_scale_param)
        fare_scale_param = scaler.fit(df['Fare'])
        df["Fare_scaled"] = scaler.fit_transform(df['Fare'],fare_scale_param)
        # print df
        train_df = df.filter(regex = 'Survived|Age_.*|SibSp|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')
        train_data = train_df.as_matrix()
        return train_data
    def get_model(self,train_data):
        y = train_data[:,0]
        x = train_data[:,1:]
        t = int(math.sqrt(x.shape[1]))
        clf = RandomForestClassifier(n_estimators=100,max_features=t)
        # clf = GradientBoostingClassifier(n_estimators=200, learning_rate=0.1, max_depth=10, random_state=0)
        # clf = linear_model.LogisticRegression(penalty='l1',tol = 1e-6)
        # clf.fit(x,y)
        print(cross_validation.cross_val_score(clf,x,y,cv=5))

if __name__ == '__main__':
    data = analysis("train.csv")
    # data.survive()
    train_data = data.data_dealing()
    data.get_model(train_data)



