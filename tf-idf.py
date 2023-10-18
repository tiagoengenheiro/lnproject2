import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.model_selection import train_test_split
from collections import Counter
import nltk
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from utils import *
nltk.download('punkt')
df=pd.read_csv("train.txt",sep="\t",names=["LABEL","REVIEW"])
print(df["LABEL"].value_counts()) #Balanced dataset
print(df.shape)
df= df.loc[:, ::-1] #Reverse the order, so that LABEL is in the second column

#Splitting
X_train, X_test, y_train, y_test = train_test_split(np.array(df["REVIEW"]), np.array(df["LABEL"]), test_size=0.2, random_state=42)




X_train_preprocessed=pre_processing(X_train)
most_common_words=get_most_common_words_among_label(X_train_preprocessed,y_train,k=10)
print(most_common_words)
X_train_preprocessed=remove_common_words(X_train_preprocessed,most_common_words)
analyse_feature_per_class(X_train_preprocessed,y_train)


#Classification
from sklearn.ensemble import RandomForestClassifier
# vectorizer = TfidfVectorizer(ngram_range=(1,1),lowercase=False)
# # #best_number_of_features(X_train,X_test,y_test)
# #Get size feature before transformation
# train_size_feature=get_size_feature(X_train)
# #Tranform to obtain tf-idf count matrix
# X_train_tf_idf=vectorizer.fit_transform(X_train)

# #Combine with tf-idf features with size_feature for trainset
# # X_train_tf_idf=np.array(X_train_tf_idf.todense())
# # X_train_tf_idf=np.hstack((X_train_tf_idf,train_size_feature))

# clf=LinearSVC(dual='auto', random_state=0, tol=1e-05,max_iter=1000,class_weight='balanced')

# clf.fit(X_train_tf_idf, y_train)

# test_size_feature=get_size_feature(X_test)
# X_test_tf_idf=vectorizer.transform(X_test)
# #Combine with tf-idf features with size_feature for test set
# # X_test_tf_idf=np.array(X_test_tf_idf.todense())
# # X_test_tf_idf=np.hstack((X_test_tf_idf,test_size_feature))

# #Prediction
# pred=clf.predict(X_test_tf_idf)
# print(accuracy_score(y_test,pred))






