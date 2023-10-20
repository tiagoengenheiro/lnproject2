import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from collections import Counter
import copy
import re,string
from nltk.tokenize import word_tokenize
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

def pre_processing(X_array):
    X_train_preprocessed=copy.deepcopy(X_array)
    for i,review in enumerate(X_array):
        review_pre_processed=word_tokenize(review) #tokenize in words
        review_pre_processed=[re.sub("[/`-]"," ",word) for word in review_pre_processed]
        review_pre_processed=[word for word in review_pre_processed if word not in (string.punctuation)]
        review_pre_processed=[word for word in review_pre_processed if word not in ["''","..."]]
        review_pre_processed=[word for word in review_pre_processed if word.strip()]
        #review_pre_processed=[word.lower() for word in review_pre_processed]
        X_train_preprocessed[i]=np.array(review_pre_processed)
    return X_train_preprocessed

def get_most_common_words_among_label(X_array_preprocessed,y_array,k=300):
    most_common_words=set()
    for index,label in enumerate(["TRUTHFULPOSITIVE","TRUTHFULNEGATIVE","DECEPTIVEPOSITIVE","DECEPTIVENEGATIVE"]):
        print(label)
        examples=X_array_preprocessed[y_array==label]
        examples=np.concatenate(examples)
        word_counter=Counter(examples)
        label_most_frequent_words=word_counter.most_common(k)
        print(label_most_frequent_words)
        label_most_frequent_words=set(aux[0] for aux in label_most_frequent_words)
        if index==0:
            most_common_words=most_common_words.union(label_most_frequent_words)
        else:
            most_common_words=most_common_words.intersection(label_most_frequent_words)
    return list(most_common_words)

def remove_common_words(X_array_preprocessed,most_commons_words): #changes X_array_preprocessed
    for i,example in enumerate(X_array_preprocessed):
        X_array_preprocessed[i]=[word for word in list(example) if word not in most_commons_words]
    return X_array_preprocessed

def get_size_feature(X_array_pre_processed):
    size_of_sentence=[len(example) for example in X_array_pre_processed]
    size_of_sentence=np.array(size_of_sentence)
    return size_of_sentence.reshape(len(size_of_sentence),1)

def analyse_feature_per_class(X_array_pre_processed,y_array):
    for _,label in enumerate(["TRUTHFULPOSITIVE","TRUTHFULNEGATIVE","DECEPTIVEPOSITIVE","DECEPTIVENEGATIVE"]):
        #SIZE FEATURE
        examples=X_array_pre_processed[y_array==label]
        label_size=[len(example) for example in examples]
        #print(label,"Avg number of tokens:",np.mean(label_size),"Std deviation of tokens:",np.std(label_size))
    
        #UPPERCASE FEATURE
        label_number_of_uppers=[]
        for example in examples:
            counter=0
            for word in example:
                if word[0].isupper():
                    counter+=1
            label_number_of_uppers.append(counter/len(example))
        print(label,"Avg number uppers:",np.mean(label_number_of_uppers),"Std deviation of uppers:",np.std(label_number_of_uppers))


#Classificatio
def tf_idf_extractor(X_array,y_array):
    vectorizer = TfidfVectorizer(ngram_range=(1,1),lowercase=False)
    #train_size_feature=get_size_feature(X_array)
    X_train_tf_idf=vectorizer.fit_transform(X_array)
    clf=SVC(C=1.0,kernel='linear',class_weight='balanced',degree=1)
    clf.fit(X_train_tf_idf, y_array)
    #test_size_feature=get_size_feature(X_array)
    X_test_tf_idf=vectorizer.transform(X_array)
    pred=clf.predict(X_test_tf_idf)
    print(accuracy_score(y_array,pred))


            
        
        


