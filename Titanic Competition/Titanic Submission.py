
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier,  AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler,
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

####################################
#Fonks
#####################################
def target_summary_with_cat(dataframe, target, categorical_col):
    print(pd.DataFrame({"TARGET_MEAN": dataframe.groupby([categorical_col, target])[target].count()}), end="\n\n\n")

def grab_col_names(dataframe, cat_th=10, car_th=20):
    """

    Veri setindeki kategorik, numerik ve kategorik fakat kardinal değişkenlerin isimlerini verir.
    Not: Kategorik değişkenlerin içerisine numerik görünümlü kategorik değişkenler de dahildir.

    Parameters
    ------
        dataframe: dataframe
                Değişken isimleri alınmak istenilen dataframe
        cat_th: int, optional
                numerik fakat kategorik olan değişkenler için sınıf eşik değeri
        car_th: int, optinal
                kategorik fakat kardinal değişkenler için sınıf eşik değeri

    Returns
    ------
        cat_cols: list
                Kategorik değişken listesi
        num_cols: list
                Numerik değişken listesi
        cat_but_car: list
                Kategorik görünümlü kardinal değişken listesi

    Examples
    ------
        import seaborn as sns
        df = sns.load_dataset("iris")
        print(grab_col_names(df))


    Notes
    ------
        cat_cols + num_cols + cat_but_car = toplam değişken sayısı
        num_but_cat cat_cols'un içerisinde.
        Return olan 3 liste toplamı toplam değişken sayısına eşittir: cat_cols + num_cols + cat_but_car = değişken sayısı

    """

    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    # print(f"Observations: {dataframe.shape[0]}")
    # print(f"Variables: {dataframe.shape[1]}")
    # print(f'cat_cols: {len(cat_cols)}')
    # print(f'num_cols: {len(num_cols)}')
    # print(f'cat_but_car: {len(cat_but_car)}')
    # print(f'num_but_cat: {len(num_but_cat)}')
    return cat_cols, num_cols, cat_but_car

def target_summary_with_num(dataframe, target, numerical_col):
    print(dataframe.groupby(target).agg({numerical_col: "mean"}), end="\n\n\n")

def one_hot_encoder(dataframe, categorical_cols, drop_first=False):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe

def label_encoder(dataframe, binary_col):
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe

def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("##########################################")
    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show(block=True)

def outlier_thresholds(dataframe, col_name, q1=0.25, q3=0.75):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

def check_outlier(dataframe, col_name, q1=0.25, q3=0.75):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name, q1, q3)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False

def one_hot_encoder(dataframe, categorical_cols, drop_first=False):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe

def base_models(X, y, scoring="roc_auc"):
    print("Base Models....")
    classifiers = [('LR', LogisticRegression()),
                   ('KNN', KNeighborsClassifier()),
                   ("SVC", SVC()),
                   ("CART", DecisionTreeClassifier()),
                   ("RF", RandomForestClassifier()),
                   ('Adaboost', AdaBoostClassifier()),
                   ('GBM', GradientBoostingClassifier()),
                   ('XGBoost', XGBClassifier(use_label_encoder=False, eval_metric='logloss')),
                   ('LightGBM', LGBMClassifier()),
                   # ('CatBoost', CatBoostClassifier(verbose=False))
                   ]

    for name, classifier in classifiers:
        cv_results = cross_validate(classifier, X, y, cv=3, scoring=scoring)
        print(f"{scoring}: {round(cv_results['test_score'].mean(), 4)} ({name}) ")


####################################
#Feature Engineering
####################################


pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
pd.set_option('display.max_rows', None)

train = pd.read_csv("C:/Users/izzet/PycharmProjects/pythonProject/Titanic Competition/train.csv")
test= pd.read_csv("C:/Users/izzet/PycharmProjects/pythonProject/Titanic Competition/test.csv")
df= pd.concat([train, test], sort=False, ignore_index=True)


###################################
# Base Model with Train Set
###################################
#Missing Value Fix
train.isna().sum()
train['Age'].replace(np.nan, 0, inplace=True)
train.loc[(train['Sex'] == 'female') & (train['Age'] == 0), 'Age'] = train.loc[(train['Sex'] == 'female')]['Age'].median()
train.loc[(train['Sex'] == 'male') & (train['Age'] == 0), 'Age'] = train.loc[(train['Sex'] == 'male')]['Age'].median()
train['Cabin'].replace(np.nan, 0, inplace=True)
train.drop('Embarked', inplace=True, axis=1)
train.isna().sum()

# Dropping some columns
train.drop(['PassengerId',"Ticket","Cabin","Name"],inplace=True,axis=1)

train.nunique()
cat_cols, num_cols, cat_but_car = grab_col_names(train, cat_th=10, car_th=20)


df.loc[df['Age']==0]
#Encoding

#Binary
labelencoder = LabelEncoder()
train['Sex'] = labelencoder.fit_transform(train['Sex'])

#One hot encoder
train = one_hot_encoder(train, cat_cols, drop_first=True)

#Scaling num_cols

scaler = StandardScaler()
train[num_cols] = scaler.fit_transform(train[num_cols])

y = train["Survived_1"]
X = train.drop(["Survived_1"], axis=1)

base_models(X, y, scoring="accuracy")

# Base Models....
# accuracy: 0.8025 (LR)
# accuracy: 0.7834 (KNN)
# accuracy: 0.8159 (SVC)
# accuracy: 0.7576 (CART)
# accuracy: 0.8137 (RF)
# accuracy: 0.798 (Adaboost)
# accuracy: 0.8058 (GBM)
# accuracy: 0.8058 (XGBoost)
# accuracy: 0.8148 (LightGBM)

rf_model = RandomForestClassifier().fit(X,y)
cv_results = cross_validate(rf_model, X, y, cv=10, scoring=["accuracy", "f1", "roc_auc"])
cv_results['test_accuracy'].mean()

def plot_importance(model, features, num=len(X), save=False):
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
    print(feature_imp.sort_values("Value",ascending=False))
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value",
                                                                     ascending=False)[0:num])
    plt.title('Features')
    plt.tight_layout()
    plt.show()
    if save:
        plt.savefig('importances.png')

plot_importance(rf_model, X)

#        Value   Feature
# 1   0.302093      Fare
# 0   0.267406       Age
# 2   0.262571     Sex_1
# 4   0.066898  Pclass_3
# 5   0.021123   SibSp_1
# 11  0.015578   Parch_1
# 3   0.015500  Pclass_2
# 12  0.014369   Parch_2
# 7   0.007311   SibSp_3
# 8   0.006999   SibSp_4
# 6   0.006366   SibSp_2
# 10  0.004500   SibSp_8

#####
#Missing Values
#####
df.isna().sum()
df['Age'].replace(np.nan,0, inplace=True)
df.loc[(df['Sex']=='female') & (df['Age'] == 0), 'Age'] = df.loc[(df['Sex']=='female')]['Age'].median()
df.loc[(df['Sex']=='male') & (df['Age'] == 0), 'Age'] = df.loc[(df['Sex']=='male')]['Age'].median()
df['Fare'].replace(np.nan, 0, inplace=True)
df.loc[(df['Fare']==0), 'Age'] = df['Fare'].mean()
df['Embarked'].replace(np.nan, 0, inplace=True)
df.loc[df['Embarked']==0,'Embarked'] = 'C'
df.isna().sum()


#################
#oUTLIERS
#################

def outlier_thresholds(dataframe, col_name, q1=0.25, q3=0.75):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

def check_outlier(dataframe, col_name, q1=0.25, q3=0.75):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name, q1, q3)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False

outlier_thresholds(df,'Age')
check_outlier(df,'Age')
replace_with_thresholds(df,'Age')
df['Age'].describe(percentiles=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.95]).T

outlier_thresholds(df,'Fare')
check_outlier(df,'Fare')
replace_with_thresholds(df,'Fare')
df['Fare'].describe(percentiles=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.95]).T

########################
# Feature Engineering
########################

#Cabins
df['Cabin_Class']=df['Cabin'].str.slice(0,1)
df['Cabin_Class'].replace(np.nan, "O", inplace=True)


df.groupby(['Cabin_Class', 'Survived']).agg({'Survived': 'count'})
df.groupby(['Cabin_Class', 'Survived','Sex']).agg({'Survived': 'count'})
df

#                       Survived
# Cabin_Class Survived
# A           0.0              8
#             1.0              7
# B           0.0             12
#             1.0             35
# C           0.0             24
#             1.0             35
# D           0.0              8
#             1.0             25
# E           0.0              8
#             1.0             24
# F           0.0              5
#             1.0              8
# G           0.0              2
#             1.0              2
# O           0.0            481
#             1.0            206
# T           0.0              1


#                              Survived
# Cabin_Class Survived Sex
# A           0.0      male           8
#             1.0      female         1
#                      male           6
# B           0.0      male          12
#             1.0      female        27
#                      male           8
# C           0.0      female         3
#                      male          21
#             1.0      female        24
#                      male          11
# D           0.0      male           8
#             1.0      female        18
#                      male           7
# E           0.0      female         1
#                      male           7
#             1.0      female        14
#                      male          10
# F           0.0      male           5
#             1.0      female         5
#                      male           3
# G           0.0      female         2
#             1.0      female         2
# O           0.0      female        75
#                      male         406
#             1.0      female       142
#                      male          64
# T           0.0      male           1

# Lucky Woman
df.loc[(df['Cabin_Class']=='A') & (df['Sex']=='female'), 'LON'] = 'LW'
df.loc[(df['Cabin_Class']=='B') & (df['Sex']=='female'), 'LON'] = 'WLW'
df.loc[(df['Cabin_Class']=='C') & (df['Sex']=='female'), 'LON'] = 'WLW'
df.loc[(df['Cabin_Class']=='D') & (df['Sex']=='female'), 'LON'] = 'WLW'
df.loc[(df['Cabin_Class']=='E') & (df['Sex']=='female'), 'LON'] = 'WLW'
df.loc[(df['Cabin_Class']=='F') & (df['Sex']=='female'), 'LON'] = 'WLW'
df.loc[(df['Cabin_Class']=='F') & (df['Sex']=='female'), 'LON'] = 'NSLW'
df.loc[(df['Cabin_Class']=='O') & (df['Sex']=='female'), 'LON'] = 'NSLW'
# Lucky Male
df.loc[(df['Cabin_Class']=='A') & (df['Sex']=='male'), 'LON'] = 'NSLM'
df.loc[(df['Cabin_Class']=='B') & (df['Sex']=='male'), 'LON'] = 'NLM'
df.loc[(df['Cabin_Class']=='C') & (df['Sex']=='male'), 'LON'] = 'BLM'
df.loc[(df['Cabin_Class']=='D') & (df['Sex']=='male'), 'LON'] = 'NSLM'
df.loc[(df['Cabin_Class']=='E') & (df['Sex']=='male'), 'LON'] = 'LM'
df.loc[(df['Cabin_Class']=='F') & (df['Sex']=='male'), 'LON'] = 'NSLM'
df.loc[(df['Cabin_Class']=='O') & (df['Sex']=='male'), 'LON'] = 'BLM'

###############
#Fares and Ages
###############

df.groupby('Survived').agg({'Fare' : 'mean'})
#                Fare
# Survived
# 0.0       18.966058
# 1.0       32.443379
df.groupby('Survived').agg({'Age' : 'mean'})
#                 Age
# Survived
# 0.0       28.893043
# 1.0       27.551959

df.loc[(df['Fare']<22) & (df['Age']>31), 'Ages_Fares'] = 'Oh_no'
df.loc[(df['Fare']>33) & (df['Age']<27), 'Ages_Fares'] = 'You_cool'
df['Ages_Fares'].replace(np.nan, "Good_Luck", inplace=True)

############################


#########################
####Titles and Classes
#########################
df['Title'] = df['Name'].str.extract('([A-Za-z]+)\.')
df.groupby(['Pclass','Title','Survived']).agg({'Survived':'count'})

df['Title'] = df['Title'].replace('Mlle', 'Miss')
df['Title'] = df['Title'].replace('Ms', 'Miss')
df['Title'] = df['Title'].replace('Mme', 'Mrs')


df.loc[((df['Pclass']==1) | (df['Pclass']==2)) & (df['Title']=='Mrs'), 'Class&Title'] = 'WOW'
df.loc[((df['Pclass']==1) | (df['Pclass']==2)) & (df['Title']=='Miss'), 'Class&Title'] = 'OK'
df.loc[((df['Pclass']==1) | (df['Pclass']==2)) & (df['Title']=='Master'), 'Class&Title'] = 'MoP'
df.loc[((df['Pclass']==2) | (df['Pclass']==3)) & (df['Title']=='Mr'), 'Class&Title'] = 'GG'
df.loc[(df['Pclass']==3) & (df['Title'].isin(['Mrs','Miss','Master'])), 'Class&Title'] = 'Well'
df.loc[(df['Pclass']==1) & (df['Title']== 'Mr'), 'Class&Title'] = 'Meh'
df['Class&Title'].replace(np.nan, 'BoL', inplace=True)

################
#Parch and SibSp
################
df['LoneVlogger'] = df.apply(lambda x: 0 if (x['SibSp']== 0) and (x['Parch']==0) else 1, axis=1)

###############
#Family Size
###############
df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
df['Age_bin'] = pd.cut(df['Age'], bins=[0, 12, 20, 40, 120], labels=['Children', 'Teenage', 'Adult', 'Elder'])
df['Fare_bin'] = pd.cut(df['Fare'], bins=[0,7.91,14.45,31,120], labels=['Low_fare','median_fare',
                                                                                      'Average_fare','high_fare'])





yz=df
df=yz
##############
#Dropping unnecessary columns
##############
df.drop(['PassengerId','Name','Parch','SibSp', 'Ticket','Cabin','Cabin_Class','Title','Embarked','Fare','Age'], inplace=True, axis=1)
df.head()

############
#Encoding
############
labelencoder = LabelEncoder()
df['Sex'] = labelencoder.fit_transform(df['Sex'])

one_hot = ['LON','Ages_Fares','Class&Title','Pclass','Fare_bin','Age_bin','FamilySize']
df = one_hot_encoder(df, one_hot, drop_first=False)
num_cols= ['Age', 'Fare']

scaler = StandardScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])

###################
#Modelling
###################

train=df.iloc[0:891]
test=df.iloc[891:1310]
test.drop('Survived', inplace=True, axis=1)

y = train["Survived"]
X = train.drop(["Survived"], axis=1)

rf_model = RandomForestClassifier(max_depth=25,max_features=11,min_samples_split=27,n_estimators=185).fit(X,y)


cv_results = cross_validate(rf_model, X, y, cv=10, scoring=["accuracy", "f1", "roc_auc"])
cv_results['test_accuracy'].mean()
cv_results['test_f1'].mean()
cv_results['test_roc_auc'].mean()

# cv_results['test_accuracy'].mean()
# Out[81]: 0.8092759051186018
# cv_results['test_f1'].mean()
# Out[82]: 0.7334224024114738
# cv_results['test_roc_auc'].mean()
# Out[83]: 0.8498477774948363


y_pred = rf_model.predict(test)

y_pred

test_ids = test['PassengerId']

submission = pd.DataFrame({'PassengerId': test_ids.values, 'Survived': y_pred})
submission
submission['Survived'] = submission['Survived'].astype(int)
submission.to_csv('submission.csv', index=False, header = 1)

#################
# HyperParameterOpt
################

rf_params = {"max_depth": [20, 25, None],
             "max_features": [9, 10, 11, "auto"],
             "min_samples_split": [27, 28, 26],
             "n_estimators": [185, 190, 195]}


rf_best_grid = GridSearchCV(rf_model, rf_params, cv=5, n_jobs=-1, verbose=True).fit(X, y)

rf_best_grid.best_params_



#####################################
####################################
####################################
##################################
df.groupby(['Survived','Pclass','SibSp'])['Survived'].count()


df.groupby(['Survived','Parch','SibSp'])['Survived'].count()
df.groupby(['Survived','SibSp'])['Survived'].count()

df.columns



df.groupby(['Cabin_Class','Pclass'])['Fare'].mean()

# Cabin_Class
# A     41.244314
# B    122.383078
# C    107.926598
# D     53.007339
# E     54.564634
# F     18.079367
# G     14.205000
# O     19.113839
# T     35.500000

df.loc[(df['Cabin_Class']=='G'), 'Cabin_Type'] =1
df.loc[(df['Cabin_Class']=='O'), 'Cabin_Type'] =2
df.loc[(df['Cabin_Class']=='F'), 'Cabin_Type'] =3
df.loc[(df['Cabin_Class']=='T'), 'Cabin_Type'] =4
df.loc[(df['Cabin_Class']=='A'), 'Cabin_Type'] =5
df.loc[(df['Cabin_Class']=='D'), 'Cabin_Type'] =6
df.loc[(df['Cabin_Class']=='E'), 'Cabin_Type'] =7
df.loc[(df['Cabin_Class']=='C'), 'Cabin_Type'] =8
df.loc[(df['Cabin_Class']=='B'), 'Cabin_Type'] =9




# Age and Fare
df.groupby(['Survived'])['Age'].describe(percentiles=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.95]).T
df.groupby(['Survived'])['Fare'].describe(percentiles=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.95]).T



df.groupby(['Survived','Sex','Pclass']).agg({'Age':['mean','count']})
Q =[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.95]




df.columns

    label_encoder(df, col)

labelencoder = LabelEncoder()
df['Sex'] = labelencoder.fit_transform(df['Sex'])

cat_cols = ["Embarked","LON","Pclass"]
df['Fare_Class'].value_counts()
df = one_hot_encoder(df, cat_cols, drop_first=False)


num_cols= ['Age', 'Fare',"SibSp","Cabin_Type"]

scaler = StandardScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])

df.drop(['Name','Ticket','Cabin','Cabin_Class'], inplace=True, axis=1)


train=df.iloc[0:891]
test=df.iloc[891:1310]
test.drop('Survived',inplace=True,axis=1)


y = train["Survived"]
X = train.drop(["Survived"], axis=1)

rf_model = RandomForestClassifier().fit(X,y)


rf_model = RandomForestClassifier(max_depth=6,min_samples_leaf=5, min_samples_split=14, n_estimators=109).fit(X,y)
rf_model.get_params()



y = df["Survived"]
X = df.drop(["Survived"], axis=1)
cv_results = cross_validate(rf_model, X, y, cv=10, scoring=["accuracy", "f1", "roc_auc"])
cv_results['test_accuracy'].mean()
cv_results['test_f1'].mean()
cv_results['test_roc_auc'].mean()


y_pred = rf_model.predict(test)

y_pred

y_pred =y_pred.str.slice(0,1)

test_ids = test['PassengerId']

submission = pd.DataFrame({'PassengerId': test_ids.values, 'Survived': y_pred})
submission
submission.to_csv('submission.csv', index=False, header = 1)

df.dtypes
submission['Survived'] = submission['Survived'].astype(int)



base_models(X, y, scoring="accuracy")

cv_results = cross_validate(rf_model, X, y, cv=10, scoring=["accuracy", "f1", "roc_auc"])
cv_results['test_accuracy'].mean()
cv_results['test_f1'].mean()
cv_results['test_roc_auc'].mean()



def plot_importance(model, features, num=len(X), save=False):
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
    print(feature_imp.sort_values("Value",ascending=False))
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value",
                                                                     ascending=False)[0:num])
    plt.title('Features')
    plt.tight_layout()
    plt.show()
    if save:
        plt.savefig('importances.png')

plot_importance(rf_model, X)



df.isna().sum()
df['Fare'].replace(np.nan, 0, inplace=True)

y_pred = rf_model.predict(df)

y_pred

test_ids = df['PassengerId']

submission = pd.DataFrame({'PassengerId': test_ids.values, 'Survived': y_pred})
submission
submission.to_csv('submission.csv', index=False, header = 1)

ab = df

######################################################
# 4. Automated Hyperparameter Optimization
######################################################

knn_params = {"n_neighbors": range(2, 50)}

cart_params = {'max_depth': range(1, 20),
               "min_samples_split": range(2, 30)}

rf_params = {"max_depth": [8, 15, None],
             "max_features": [5, 7, "auto"],
             "min_samples_split": [15, 20],
             "n_estimators": [200, 300]}

xgboost_params = {"learning_rate": [0.1, 0.01],
                  "max_depth": [5, 8],
                  "n_estimators": [100, 200]}

lightgbm_params = {"learning_rate": [0.01, 0.1],
                   "n_estimators": [300, 500]}


classifiers = [('KNN', KNeighborsClassifier(), knn_params),
               ("CART", DecisionTreeClassifier(), cart_params),
               ("RF", RandomForestClassifier(), rf_params),
               ('XGBoost', XGBClassifier(use_label_encoder=False, eval_metric='logloss'), xgboost_params),
               ('LightGBM', LGBMClassifier(), lightgbm_params)]



def hyperparameter_optimization(X, y, cv=3, scoring="roc_auc"):
    print("Hyperparameter Optimization....")
    best_models = {}
    for name, classifier, params in classifiers:
        print(f"########## {name} ##########")
        cv_results = cross_validate(classifier, X, y, cv=cv, scoring=scoring)
        print(f"{scoring} (Before): {round(cv_results['test_score'].mean(), 4)}")

        gs_best = GridSearchCV(classifier, params, cv=cv, n_jobs=-1, verbose=False).fit(X, y)
        final_model = classifier.set_params(**gs_best.best_params_)

        cv_results = cross_validate(final_model, X, y, cv=cv, scoring=scoring)
        print(f"{scoring} (After): {round(cv_results['test_score'].mean(), 4)}")
        print(f"{name} best params: {gs_best.best_params_}", end="\n\n")
        best_models[name] = final_model
    return best_models

best_models = hyperparameter_optimization(X, y)

######################################################
# 4. Automated Hyperparameter Optimization
######################################################

y_pred = rf_model.predict(df)

y_pred

test_ids = test['PassengerId']

submission = pd.DataFrame({'PassengerId': test_ids.values, 'Survived': y_pred})
submission
submission.to_csv('submission.csv', index=False, header = 1)

ab = df
