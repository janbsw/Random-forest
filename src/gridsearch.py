import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing as le
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.model_selection import GridSearchCV

grid_search = {
    'criterion': [model.best_params_['criterion']],
    'max_depth': [model.best_params_['max_depth']],
    'max_features': [model.best_params_['max_features']],
    'min_samples_leaf': [model.best_params_['min_samples_leaf'] - 2, 
                         model.best_params_['min_samples_leaf'], 
                         model.best_params_['min_samples_leaf'] + 2],
    'min_samples_split': [model.best_params_['min_samples_split'] - 3, 
                          model.best_params_['min_samples_split'], 
                          model.best_params_['min_samples_split'] + 3],
    'n_estimators': [model.best_params_['n_estimators'] - 150, 
                     model.best_params_['n_estimators'] - 100, 
                     model.best_params_['n_estimators'], 
                     model.best_params_['n_estimators'] + 100, 
                     model.best_params_['n_estimators'] + 150]
}

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
    clf = RandomForestClassifier()
    model = GridSearchCV(estimator = clf, param_grid = grid_search, 
                               cv = 4, verbose= 10, n_jobs = -1)
    model.fit(features_train,target_train)
    target_pred=model.best_estimator_.predict(features_test)
    print(eveluate_preditctions(target_test,target_pred))


if __name__ == '__main__':
    main()

    