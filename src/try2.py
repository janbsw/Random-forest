import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing as le
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics


def load_dataset():
    return pd.read_csv('../data/CurrentdetailData.csv')

def eveluate_preditctions(target_test,target_pred):
    return metrics.accuracy_score(target_test, target_pred)

def preprocessing(data):
    return data.apply(le.LabelEncoder().fit_transform)

def predict(features_train, target_train,features_test):
    clf=RandomForestClassifier(n_estimators=200)
    clf.fit(features_train, target_train.ravel())
    target_pred=clf.predict(features_test)
    return target_pred

def create_datasets(data,target_label,target):
    return create_features(data,target_label), create_targets(data,target)

def create_features(data, target_label):
    return data.drop(target_label,axis=1)

def create_targets(data,target):
    return data.iloc[:,target-1:target].values.reshape(-1,1)

def create_test_train(feature,target):
    return train_test_split(feature,target,test_size=0.3,random_state=50)

def main():
    df=load_dataset()
    df=preprocessing(df)
    feature, target = create_datasets(df,"Target",6)
    features_train, features_test, target_train, target_test = create_test_train(feature,target)
    target_pred=predict(features_train,target_train,features_test)
    print("Accuracy",eveluate_preditctions(target_test,target_pred))


if __name__ == '__main__':
    main()

    