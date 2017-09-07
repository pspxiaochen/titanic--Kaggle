import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from collections import Counter
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier, VotingClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold, learning_curve

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
IDtest = test.PassengerId

def detect_outliers(df,n,features):
    outlier_indices = []
    for col in features:
        Q1 = np.percentile(df[col],25)
        Q3 = np.percentile(df[col],75)

        IQR = Q3 - Q1

        outlier_step = 1.5 * IQR

        outlier_list_col = df[(df[col] < Q1 - outlier_step)|(df[col] > Q3 + outlier_step)].index

        outlier_indices.extend(outlier_list_col)
    outlier_indices = Counter(outlier_indices)
    multiple_outliers = list(k for k,v in outlier_indices.items() if v>n)
    return multiple_outliers
#找出异常值的索引
Outliers_to_drop = detect_outliers(train,2,["Age","SibSp","Parch","Fare"])

train = train.drop(Outliers_to_drop).reset_index(drop=True)

train_len = len(train)
dataset = pd.concat([train,test]).reset_index(drop = True)
dataset = dataset.fillna(np.nan)

# g = sns.heatmap(train[["Survived","SibSp","Parch","Age","Fare"]].corr(),annot=True, fmt = ".2f", cmap = "coolwarm")

# g = sns.factorplot(x="SibSp",y="Survived",data=train,kind="bar", size = 6 ,
# palette = "muted")
# g = g.set_ylabels("survival probability")
# plt.show()

#用中位数填补Fare的缺失
dataset.Fare = dataset.Fare.fillna(dataset.Fare.median())

dataset["Fare"] = dataset["Fare"].map(lambda i: np.log(i) if i > 0 else 0)

#用S来填补Embarked的缺失
dataset.Embarked = dataset.Embarked.fillna("S")

#将男性变成0 女性变成1
dataset.Sex = dataset.Sex.map({"male":0,"female":1})

#填补年龄的缺失值
index_NaN_age = list(dataset.Age[dataset.Age.isnull()].index)
for i in index_NaN_age:
    age_med = dataset.Age.median()
    age_pred = dataset["Age"][((dataset['SibSp'] == dataset.iloc[i]["SibSp"]) & (dataset['Parch'] == dataset.iloc[i]["Parch"]) & (dataset['Pclass'] == dataset.iloc[i]["Pclass"]))].median()
    if not np.isnan(age_pred):
        dataset.Age.iloc[i] = age_pred
    else:
        dataset.Age.iloc[i] = age_med

#从名字信息中获取头衔
dataset_title = [i.split(",")[1].split(".")[0].strip() for i in dataset.Name]

#创建Title属性
dataset["Title"] = pd.Series(dataset_title)

dataset["Title"] = dataset["Title"].replace(['Lady', 'the Countess','Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
dataset["Title"] = dataset["Title"].map({"Master":0, "Miss":1, "Ms" : 1 , "Mme":1, "Mlle":1, "Mrs":1, "Mr":2, "Rare":3})
dataset["Title"] = dataset["Title"].astype(int)
#将名字属性丢掉
dataset.drop(labels="Name",axis = 1,inplace = True)

#创建Family size 属性 是SibSp 和 Parch 和自己本人的总和
dataset["Fsize"] = dataset["SibSp"] + dataset["Parch"] + 1
#关于家庭成员的数量创建新的属性
dataset['Single'] = dataset['Fsize'].map(lambda i:1 if i == 1 else 0)
dataset['SmallF'] = dataset['Fsize'].map(lambda i:1 if i == 2 else 0)
dataset['MedF'] = dataset['Fsize'].map(lambda i:1 if 3<=i<=4 else 0)
dataset['LargeF'] = dataset['Fsize'].map(lambda i:1 if i>=5 else 0)

#将原来的变成数值型
dataset = pd.get_dummies(dataset,columns=["Title"])
dataset = pd.get_dummies(dataset,columns=["Embarked"],prefix="Em")

dataset.Cabin = pd.Series(i[0] if not pd.isnull(i) else 'X' for i in dataset.Cabin)
dataset = pd.get_dummies(dataset,columns=["Cabin"],prefix="Cabin")

#通过提取出票前缀来处理票，当没有前缀时，返回X
Ticket = []
for i in list(dataset.Ticket):
    if not i.isdigit():
        Ticket.append(i.replace(".", "").replace("/", "").strip().split(' ')[0])
    else:
        Ticket.append("X")
dataset.Ticket = Ticket

dataset = pd.get_dummies(dataset,columns=["Ticket"],prefix="T")

dataset.Pclass = dataset.Pclass.astype("category")
dataset = pd.get_dummies(dataset,columns=["Pclass"],prefix="Pc")

dataset.drop(labels="PassengerId",axis=1,inplace=True)

##########################模型
train = dataset[:train_len]
test = dataset[:train_len]
test.drop(labels=["Survived"],axis = 1,inplace = True)

train["Survived"] = train["Survived"].astype(int)
Y_train = train["Survived"]
X_train = train.drop(labels="Survived",axis = 1)


kfold = StratifiedKFold(n_splits=10)
# Adaboost

DTC = DecisionTreeClassifier()

adaDTC = AdaBoostClassifier(DTC, random_state=7)

ada_param_grid = {"base_estimator__criterion" : ["gini", "entropy"],
              "base_estimator__splitter" :   ["best", "random"],
              "algorithm" : ["SAMME","SAMME.R"],
              "n_estimators" :[1,2],
              "learning_rate":  [0.0001, 0.001, 0.01, 0.1, 0.2, 0.3,1.5]}

gsadaDTC = GridSearchCV(adaDTC,param_grid = ada_param_grid, cv=kfold, scoring="accuracy", n_jobs= 1, verbose = 1)

gsadaDTC.fit(X_train,Y_train)

ada_best = gsadaDTC.best_estimator_

#ExtraTrees
ExtC = ExtraTreesClassifier()


## Search grid for optimal parameters
ex_param_grid = {"max_depth": [None],
              "max_features": [1, 3, 10],
              "min_samples_split": [2, 3, 10],
              "min_samples_leaf": [1, 3, 10],
              "bootstrap": [False],
              "n_estimators" :[100,300],
              "criterion": ["gini"]}


gsExtC = GridSearchCV(ExtC,param_grid = ex_param_grid, cv=kfold, scoring="accuracy", n_jobs= 1, verbose = 1)

gsExtC.fit(X_train,Y_train)

ExtC_best = gsExtC.best_estimator_

# RFC Parameters tunning
RFC = RandomForestClassifier()


## Search grid for optimal parameters
rf_param_grid = {"max_depth": [None],
              "max_features": [1, 3, 10],
              "min_samples_split": [2, 3, 10],
              "min_samples_leaf": [1, 3, 10],
              "bootstrap": [False],
              "n_estimators" :[100,300],
              "criterion": ["gini"]}


gsRFC = GridSearchCV(RFC,param_grid = rf_param_grid, cv=kfold, scoring="accuracy", n_jobs= 1, verbose = 1)

gsRFC.fit(X_train,Y_train)

RFC_best = gsRFC.best_estimator_

# Gradient boosting tunning

GBC = GradientBoostingClassifier()
gb_param_grid = {'loss' : ["deviance"],
              'n_estimators' : [100,200,300],
              'learning_rate': [0.1, 0.05, 0.01],
              'max_depth': [4, 8],
              'min_samples_leaf': [100,150],
              'max_features': [0.3, 0.1]
              }

gsGBC = GridSearchCV(GBC,param_grid = gb_param_grid, cv=kfold, scoring="accuracy", n_jobs= 1, verbose = 1)

gsGBC.fit(X_train,Y_train)

GBC_best = gsGBC.best_estimator_

### SVC classifier
SVMC = SVC(probability=True)
svc_param_grid = {'kernel': ['rbf'],
                  'gamma': [ 0.001, 0.01, 0.1, 1],
                  'C': [1, 10, 50, 100,200,300, 1000]}

gsSVMC = GridSearchCV(SVMC,param_grid = svc_param_grid, cv=kfold, scoring="accuracy", n_jobs= 1, verbose = 1)

gsSVMC.fit(X_train,Y_train)

SVMC_best = gsSVMC.best_estimator_

test_Survived_RFC = pd.Series(RFC_best.predict(test), name="RFC")
test_Survived_ExtC = pd.Series(ExtC_best.predict(test), name="ExtC")
test_Survived_SVMC = pd.Series(SVMC_best.predict(test), name="SVC")
test_Survived_AdaC = pd.Series(ada_best.predict(test), name="Ada")
test_Survived_GBC = pd.Series(GBC_best.predict(test), name="GBC")

votingC = VotingClassifier(estimators=[('rfc', RFC_best), ('extc', ExtC_best),
('svc', SVMC_best), ('adac',ada_best),('gbc',GBC_best)], voting='soft', n_jobs=4)

votingC = votingC.fit(X_train, Y_train)

test_Survived = pd.Series(votingC.predict(test), name="Survived")

results = pd.concat([IDtest,test_Survived],axis=1)

results.to_csv("ensemble_python_voting.csv",index=False)