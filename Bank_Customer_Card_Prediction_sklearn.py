import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

bank = pd.read_csv("drive/kongkea/Dataset/card.csv")
bank.head(5)
bank.shape
bank.isnull().sum()
sns.set_theme(style="whitegrid")
sns.boxplot(bank["Age"])
bank[["Gender", "Credit_Limit"]].groupby("Gender").agg(["mean", "count"])
bank[["Gender", "Mean_use_ratio"]].groupby(
    "Gender").agg(["mean", "count"])
bank_cards = bank.groupby("Card")
bank_cards["Age"].max()
bank_cards["Age"].min()
bank.columns
bank_cards["Mean_use_ratio"].mean()
bank_marital = bank.groupby("Marital")
bank_marital["Card"].value_counts()
bank.head(3)
bank["Attenuation"].value_counts()
bank["Gender"].value_counts()
bank["Education"].value_counts()
bank["Marital"].value_counts()


def ref1(x):
    if x == "M":
        return 1
    else:
        return 0


bank["Gender"] = bank["Gender"].map(ref1)


def ref2(x):
    if x == "Existing Customer":
        return 1
    else:
        return 0


bank["Attenuation"] = bank["Attenuation"].map(ref2)
y = bank["Card"]
X = bank.copy()
X.head(3)
X["Income"].value_counts()


def label_encoded(feat):
    le = LabelEncoder()
    le.fit(feat)
    print(feat.name, le.classes_)
    return le.transform(feat)


X["Income"] = label_encoded(X["Income"])
X["Education"] = label_encoded(X["Education"])
X["Marital"] = label_encoded(X["Marital"])
X.head(3)
X.describe()
X.shape

pca = PCA(n_components=7)
pca2 = PCA(n_components=10)
pca_fit = pca.fit_transform(X)
pca_fit2 = pca2.fit_transform(X)

Xtrain, Xtest, ytrain, ytest = train_test_split(
    pca_fit2, y, test_size=0.2, random_state=42
)
random_model = RandomForestClassifier(n_estimators=300, n_jobs=-1)
random_model.fit(Xtrain, ytrain)
y_pred = random_model.predict(Xtest)
random_model_accuracy = round(random_model.score(Xtrain, ytrain) * 100, 2)
print(round(random_model_accuracy, 2), "%")
random_model_accuracy1 = round(random_model.score(Xtest, ytest) * 100, 2)
print(round(random_model_accuracy1, 2), "%")

saved_model = pickle.dump(
    random_model, open("drive/kongkea/Dataset/Models/card.pickle", "wb")
)
saved_pca = pickle.dump(
    pca2, open("drive/kongkea/Dataset/Models/card_pca.pickle", "wb")
)
