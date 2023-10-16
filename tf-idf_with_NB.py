import pandas as pd
from sklearn import feature_extraction 
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.model_selection import train_test_split
df=pd.read_csv("train.txt",sep="\t",names=["LABEL","REVIEW"])
print(df["LABEL"].value_counts()) #Balanced dataset
print(df.shape)
df= df.loc[:, ::-1] #Reverse the order, so that LABEL is in the second column
X_train, X_test, y_train, y_test = train_test_split(df["REVIEW"], df["LABEL"], test_size=0.2, random_state=42)
vectorizer = TfidfVectorizer(ngram_range=(1,1))
X_train=vectorizer.fit_transform(X_train)
clf = MultinomialNB(force_alpha=True)
clf.fit(X_train, y_train)
X_test=vectorizer.transform(X_test)
pred=clf.predict(X_test)
print(accuracy_score(y_test,pred))
