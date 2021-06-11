import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics

data=pd.read_csv('../data/CurrentdetailData.csv')

data=data.apply(preprocessing.LabelEncoder().fit_transform)

features=data.drop('Target',axis=1)
target=data.iloc[:,5:6].values.reshape(-1,1)

print(target.shape[0])
print(features.shape[0])
features_train, features_test, target_train, target_test = train_test_split(features,target,test_size=0.3,random_state=50)
clf=RandomForestClassifier(n_estimators=200)
clf.fit(features_train, target_train.ravel())

target_pred=clf.predict(features_test)

print("Accuracy:",metrics.accuracy_score(target_test, target_pred))