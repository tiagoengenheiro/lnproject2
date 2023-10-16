#https://towardsdatascience.com/word-embeddings-for-sentiment-analysis-65f42ea5d26e

import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.model_selection import train_test_split
from sentence_transformers import SentenceTransformer
import numpy as np
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

df=pd.read_csv("train.txt",sep="\t",names=["LABEL","REVIEW"])

df= df.loc[:, ::-1] #Reverse the order, so that LABEL is in the second column

X_train, X_test, y_train, y_test = train_test_split(df["REVIEW"], df["LABEL"], test_size=0.2, random_state=42)

X_train=np.array(X_train)
X_test=np.array(X_test)
y_train=np.array(y_train)
y_test=np.array(y_test)
print(X_train.shape)
#sentence-transformers/distiluse-base-multilingual-cased-v2
model = SentenceTransformer('sentence-transformers/paraphrase-mpnet-base-v2')

def get_embeddings(X_array,step=100):
    embeddings=model.encode(X_array[:step],device=device,convert_to_numpy=True)
    for i in range(step,X_array.shape[0],step):   
        embeddings=np.vstack((embeddings,model.encode(X_array[i:i+step],device=device,convert_to_numpy=True)))
    return embeddings

test_embeddings=get_embeddings(X_test,step=100)
train_embeddings=get_embeddings(X_train,step=150)


test_embeddings/=np.linalg.norm(test_embeddings,axis=1).reshape(test_embeddings.shape[0],1) #280,512
train_embeddings/=np.linalg.norm(train_embeddings,axis=1).reshape(train_embeddings.shape[0],1) #1120,512

predictions=np.dot(test_embeddings,train_embeddings.T)
predictions=np.argmax(predictions,axis=1)
predictions=y_train[predictions]

print(accuracy_score(y_test,predictions))



