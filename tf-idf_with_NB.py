import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.model_selection import train_test_split
from collections import Counter
import nltk
from nltk.tokenize import word_tokenize
import string
import re
import copy
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
nltk.download('punkt')
df=pd.read_csv("train.txt",sep="\t",names=["LABEL","REVIEW"])
print(df["LABEL"].value_counts()) #Balanced dataset
print(df.shape)
df= df.loc[:, ::-1] #Reverse the order, so that LABEL is in the second column

#Splitting
X_train, X_test, y_train, y_test = train_test_split(np.array(df["REVIEW"]), np.array(df["LABEL"]), test_size=0.2, random_state=42)

X_train_preprocessed=copy.deepcopy(X_train)
for i,review in enumerate(X_train):
    review_pre_processed=word_tokenize(review) #tokenize in words
    review_pre_processed=[re.sub("[/`-]"," ",word) for word in review_pre_processed]
    review_pre_processed=[word for word in review_pre_processed if word not in (string.punctuation)]
    review_pre_processed=[word for word in review_pre_processed if word not in ["''","..."]]
    review_pre_processed=[word for word in review_pre_processed if word.strip()]
    review_pre_processed=[word.lower() for word in review_pre_processed]
    X_train_preprocessed[i]=np.array(review_pre_processed)

most_common_words=set()

for index,label in enumerate(["TRUTHFULPOSITIVE","TRUTHFULNEGATIVE","DECEPTIVEPOSITIVE","DECEPTIVENEGATIVE"]):
    print(label)
    examples=X_train_preprocessed[y_train==label]
    size=[]
    for example in examples:
        size.append(len(example))
    examples=np.concatenate(examples)
    print("Total number of tokens:",examples.shape[0],"avg number of tokens:", np.mean(size), "std dev:",np.std(size))
    word_counter=Counter(examples)
    
    label_most_frequent_words=word_counter.most_common(150)
    label_most_frequent_words=set(aux[0] for aux in label_most_frequent_words)
    if index==0:
        most_common_words=most_common_words.union(label_most_frequent_words)
    else:
        most_common_words=most_common_words.intersection(label_most_frequent_words)
    print("---------------------------------")

def get_size_feature(X_array):
    size_of_sentence=[]
    for example in X_array:
        size_of_sentence.append(len(example))
    size_of_sentence=np.array(size_of_sentence)
    size_of_sentence=size_of_sentence/700
    return size_of_sentence.reshape(len(size_of_sentence),1)

def best_number_of_features(X_train_array,X_test_array,y_test_array):
    for max_features in np.arange(1000,8000,1000):
        vectorizer = TfidfVectorizer(ngram_range=(1,1),stop_words=list(most_common_words),max_features=max_features)
        X_train_tf_idf=vectorizer.fit_transform(X_train_array)
        print(X_train_tf_idf.shape)
        #X_train=np.array(X_train.todense())
        # X_train=np.hstack((X_train,train_size_feature))
        clf = MultinomialNB(force_alpha=True)
        clf.fit(X_train_tf_idf, y_train)

        #test_size_feature=get_size_feature(X_test)
        X_test_tf_idf=vectorizer.transform(X_test_array)
        #X_test=np.array(vectorizer.transform(X_test).todense())
        #X_test=np.hstack((X_test,test_size_feature))
        pred=clf.predict(X_test_tf_idf)
        print(accuracy_score(y_test_array,pred))

#train_size_feature=get_size_feature(X_train)
#print(list(most_common_words)) #=0.774
print("Number of common words in all classes:", len(most_common_words))

from sklearn.ensemble import RandomForestClassifier
#vectorizer = TfidfVectorizer(ngram_range=(1,1),stop_words=list(most_common_words),max_features=2000)
vectorizer = TfidfVectorizer(ngram_range=(1,1),max_features=3500)
#best_number_of_features(X_train,X_test,y_test)
#Get size feature before transformation
train_size_feature=get_size_feature(X_train)
#Tranform to obtain tf-idf count matrix
X_train_tf_idf=vectorizer.fit_transform(X_train)

#Combine with tf-idf features with size_feature for trainset
X_train_tf_idf=np.array(X_train_tf_idf.todense())
X_train_tf_idf=np.hstack((X_train_tf_idf,train_size_feature))

clf=LinearSVC(dual='auto', random_state=0, tol=1e-05,max_iter=1000,class_weight='balanced')

clf.fit(X_train_tf_idf, y_train)

test_size_feature=get_size_feature(X_test)
X_test_tf_idf=vectorizer.transform(X_test)
#Combine with tf-idf features with size_feature for test set
X_test_tf_idf=np.array(X_test_tf_idf.todense())
X_test_tf_idf=np.hstack((X_test_tf_idf,test_size_feature))

#Prediction
pred=clf.predict(X_test_tf_idf)
print(accuracy_score(y_test,pred))






