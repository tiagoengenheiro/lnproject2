import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from collections import Counter
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from utils import *
import matplotlib.pyplot as plt
import seaborn as sns

df=pd.read_csv("train.txt",sep="\t",names=["LABEL","REVIEW"])
df= df.loc[:, ::-1]
def separate_class(example):
    index=example.find("NEGATIVE")
    if index==-1: #Negative is present
        index=example.find("POSITIVE")
    class1=example[:index]
    class2=example[index:]
    return [class1,class2]

separated_classes=df["LABEL"].apply(separate_class).tolist()
class1=[a[0] for a in separated_classes]
class2=[a[1] for a in separated_classes]

df["Class1"]=class1 #0.89
df["Class2"]=class2 #0.94

random_state=42

# best_hypers={
#     0:[0,None],
#     1:[0,None],
# }
# for i,label in enumerate([class1,class2]):
#     X_train, X_test, y_train, y_test = train_test_split(np.array(df["REVIEW"]),np.array(label), test_size=0.2,random_state=random_state)
#     print(np.unique(label))
#     for max_features in np.arange(250,8000,250):
#         vectorizer = TfidfVectorizer(ngram_range=(1,1),lowercase=False,max_features=max_features)
#         X_train_tf_idf=vectorizer.fit_transform(X_train)
#         for kernel in ["linear","rbf"]:
#             for C in [1,5,10]:
#                 clf=SVC(C=C,kernel=kernel,random_state=random_state)
#                 clf.fit(X_train_tf_idf, y_train)
#                 X_test_tf_idf=vectorizer.transform(X_test)
#                 pred=clf.predict(X_test_tf_idf)
#                 acc=accuracy_score(y_test,pred)
#                 if acc>best_hypers[i][0]:
#                     best_hypers[i][0]=acc
#                     best_hypers[i][1]={
#                         "max_features":max_features,
#                         "kernel":kernel,
#                         "C":C,
#                     }
#                 print(best_hypers)
preds=[]
best_hypers={0: [0.8928571428571429, {'max_features': 2250, 'kernel': 'rbf', 'C': 1}], 1: [0.9428571428571428, {'max_features': 3000, 'kernel': 'rbf', 'C': 5}]}
for i,label in enumerate([class1,class2]):
    X_train, X_test, y_train, y_test = train_test_split(np.array(df["REVIEW"]),np.array(label), test_size=0.2,random_state=random_state)
    print(X_test[0])
    print(np.unique(label))
    vectorizer = TfidfVectorizer(ngram_range=(1,1),lowercase=False,max_features=best_hypers[i][1]["max_features"])
    
    X_train_tf_idf=vectorizer.fit_transform(X_train)
    #clf=GaussianNB()
    clf=SVC(C=best_hypers[i][1]["C"],kernel=best_hypers[i][1]["kernel"],random_state=random_state)
    clf.fit(X_train_tf_idf, y_train)
    X_test_tf_idf=vectorizer.transform(X_test)
    pred=clf.predict(X_test_tf_idf)
    acc=accuracy_score(y_test,pred)
    save_acc_per_label(y_test,pred,model_name=f'sparse_{i}')

    print(acc)
    preds.append(pred)

[pred1,pred2]=preds
final_preds=[]
for el1,el2 in zip(pred1,pred2):
    final_preds.append(el1+el2)

#0.8107
final_preds=np.array(final_preds)
_, X_test, _, y_test = train_test_split(np.array(df["REVIEW"]),np.array(df["LABEL"]), test_size=0.2,random_state=random_state)
error_index=y_test!=final_preds
print(y_test)
np.save("val_true_labels",y_test)
df_test=pd.DataFrame({"Predicted":final_preds[error_index],"True":y_test[error_index],"Reviews":X_test[error_index]})
df_test.to_csv("outputs/errors_sparse.tsv",sep="\t",index=False)

print(accuracy_score(y_test,final_preds))
save_confusion_matrices(y_test,final_preds,model_name='sparse')
save_acc_per_label(y_test,final_preds,model_name='sparse_agg')

#multilabel_confusion_matrix(y_test,final_preds))


