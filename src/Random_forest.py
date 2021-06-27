import pandas as pd #Bibliothek zum Importieren und zur Manipulation von Daten
from sklearn.model_selection import train_test_split #Bibliothek zum Aufteilen der Daten in Trainings- und Arbeits-set
from sklearn import preprocessing as le #Bibliothek zur Daten Vorverarbeitung
from sklearn.ensemble import RandomForestClassifier #Importieren des Random Forest Classifiers
from sklearn import metrics #Importieren der Bibliothek zum erstellen von Graphen des Random Forest Classifiers
from sklearn import tree #Bibliothek zum exportieren des Visullen Baums
from matplotlib import pyplot as plt #Bibliothek zur Visuallisierung
from sklearn.metrics import plot_roc_curve #Bibliothek zur Visuallisierung ROC curve

# Laden des datasets
def load_dataset():
    return pd.read_csv('../data/CurrentdetailData.csv')

# Die genauigkeit des Modells bestimmen
def eveluate_preditctions(target_test,target_pred):
    return metrics.accuracy_score(target_test, target_pred)

# Daten vorverarbeiten
def preprocessing(data):
    return data.apply(le.LabelEncoder().fit_transform)

# Modell trainieren
def train(features_train, target_train):
    clf=RandomForestClassifier(n_estimators=200)
    clf.fit(features_train, target_train.ravel())
    return clf

# Modell anwenden     
def preditct(clf,features_test):
    target_pred=clf.predict(features_test)
    return target_pred

# Datensatz in features und targets aufteilen
def create_datasets(data,target_label,target):
    return create_features(data,target_label), create_targets(data,target)

# Features aus roh Datensatz
def create_features(data, target_label):
    return data.drop(target_label,axis=1)

# Targets aus roh Datensatz
def create_targets(data,target):
    return data.iloc[:,target-1:target].values.reshape(-1,1)

# Teilen des Datensatzes
def create_test_train(feature,target):
    return train_test_split(feature,target,test_size=0.3,random_state=50)

# Visualisierung des Baumes
def visualization(clf,feature):
    clff=clf.estimators_[0]
    dotfile = open("../tree.dot", 'w')
    tree.export_graphviz(clff, out_file = dotfile, feature_names = feature.columns)
    dotfile.close()

# Visualisierung der wichtigkeit des features
def feature_importance(clf,feature):
    importance=clf.feature_importances_
    sorted_importance=importance.argsort()
    plt.barh(feature.columns[sorted_importance],importance[sorted_importance])
    plt.xlabel("Random Forest Feature Importance")
    plt.savefig('../graphs/feature_importance.png', bbox_inches="tight")

# Visualisierung der ROC curve
def roc_curve(clf,feature_test,target_test):
    plot_roc_curve(clf,feature_test,target_test)
    plt.savefig('../graphs/roc_curve.png', bbox_inches="tight")


# Ausführung
def main():
    df=load_dataset()
    df=preprocessing(df)
    feature, target = create_datasets(df,"Target",6)
    features_train, features_test, target_train, target_test = create_test_train(feature,target)
    clf=train(features_train,target_train)
    target_pred=preditct(clf,features_test)
    #visualization(clf,feature)
    feature_importance(clf,feature)
    roc_curve(clf, features_test,target_test)
    
    print("Accuracy",eveluate_preditctions(target_test,target_pred))

# Bei start führe Funktion main aus
if __name__ == '__main__':
    main()

    