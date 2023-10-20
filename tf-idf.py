import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.model_selection import train_test_split
from collections import Counter
import nltk
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import nltk
from utils import *


nltk.download('punkt')
df=pd.read_csv("train.txt",sep="\t",names=["LABEL","REVIEW"])
print(df["LABEL"].value_counts()) #Balanced dataset
print(df.shape)
df= df.loc[:, ::-1] #Reverse the order, so that LABEL is in the second column

#Splitting
X_train, X_test, y_train, y_test = train_test_split(np.array(df["REVIEW"]), np.array(df["LABEL"]), test_size=0.2)


X_train_preprocessed=pre_processing(X_train)
most_common_words=get_most_common_words_among_label(X_train_preprocessed,y_train,k=10)
print(most_common_words)
X_train_preprocessed=remove_common_words(X_train_preprocessed,most_common_words)
analyse_feature_per_class(X_train_preprocessed,y_train)







