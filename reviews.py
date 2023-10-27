# !pip install transformers
# !pip install datasets
# !pip install "tf-models-official==2.13.*"

import pandas as pd
from datasets import Dataset,DatasetDict
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.model_selection import train_test_split
from utils import *
import tensorflow as tf
from transformers import TFDistilBertForSequenceClassification
from transformers import DefaultDataCollator
import pandas as pd
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from datasets import Dataset
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score
from official.nlp import optimization

random_state=42
seed=812
#MODEL_NAME = 'distilbert-base-cased'
MODEL_NAME= 'distilbert-base-uncased'
tokenizer = DistilBertTokenizer.from_pretrained(MODEL_NAME)

def reverse_mapping(preds):
    mapping={
        0:"DECEPTIVENEGATIVE",
        1:"DECEPTIVEPOSITIVE",
        2:"TRUTHFULNEGATIVE",
        3:"TRUTHFULPOSITIVE",
    }
    return [mapping[pred] for pred in preds]

def label_mapping(example):
    mapping={
        "DECEPTIVENEGATIVE":0,
        "DECEPTIVEPOSITIVE":1,
        "TRUTHFULNEGATIVE":2,
        "TRUTHFULPOSITIVE":3,
    }
    return {"labels":mapping[example["labels"]]}


def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

def train_pre_processing(txt_file):
  df=pd.read_csv(txt_file,sep="\t",names=["LABEL","REVIEW"])
  X_train, X_test, y_train, y_test = train_test_split( df["REVIEW"].to_list(),df["LABEL"].to_list(), test_size=0.2, random_state=random_state)
  print(X_test[0])
  train_dataset=Dataset.from_dict({"text":X_train,"labels":y_train})
  test_dataset=Dataset.from_dict({"text":X_test,"labels":y_test})
  dataset=DatasetDict({"train":train_dataset,"test":test_dataset})
  dataset=dataset.map(label_mapping,batched=False)
  tokenized_datasets = dataset.map(tokenize_function, batched=True)
  tokenized_datasets = tokenized_datasets.remove_columns(["text"])
  return tokenized_datasets,X_test

def test_pre_processing(txt_file):
  with open(txt_file, "r") as file:
      content = file.read()
  content=content.splitlines()
  df=pd.DataFrame(content,columns=["REVIEW"])
  dataset = Dataset.from_pandas(df)
  dataset=dataset.rename_columns({"REVIEW":"text"})
  tokenized_datasets = dataset.map(tokenize_function, batched=True)
  tokenized_datasets = tokenized_datasets.remove_columns(["text"])
  return tokenized_datasets

def get_predictions(model,dataset):
  preds=model.predict(dataset)
  return np.argmax(preds.logits,axis=1)

tf.keras.utils.set_random_seed(seed)

model = TFDistilBertForSequenceClassification.from_pretrained(MODEL_NAME,num_labels=4)

batch_size=8
epochs=6

data_collator = DefaultDataCollator(return_tensors="tf")
tokenizer = DistilBertTokenizer.from_pretrained(MODEL_NAME)
tokenized_datasets,X_test=train_pre_processing("train.txt")

tf_train_dataset = tokenized_datasets["train"].to_tf_dataset(
    columns=["attention_mask", "input_ids"],
    label_cols="labels",
    shuffle=True,
    collate_fn=data_collator,
    batch_size=batch_size,
)

tf_validation_dataset = tokenized_datasets["test"].to_tf_dataset(
    columns=["attention_mask", "input_ids"],
    label_cols="labels",
    shuffle=False,
    collate_fn=data_collator,
    batch_size=batch_size,
)


steps_per_epoch = tf.data.experimental.cardinality(tf_train_dataset).numpy()
num_train_steps = steps_per_epoch * epochs
num_warmup_steps = int(0.1*num_train_steps)

init_lr = 3e-5
optimizer = optimization.create_optimizer(init_lr=init_lr,
                                          num_train_steps=num_train_steps,
                                          num_warmup_steps=num_warmup_steps,
                                          optimizer_type='adamw')

model.compile(
    #optimizer=tf.keras.optimizers.AdamW(init_lr),
    optimizer=optimizer,
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=tf.metrics.SparseCategoricalAccuracy(),
)

#optimizer_type='customized'
#optimizer_type='adamW'
#save_model_dir=os.path.join(os.getcwd(),f"model_seed_{seed}_epcs_{epochs}_bs_{batch_size}_lr_{init_lr}_opt_{optimizer_type}")
callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=2),
             #tf.keras.callbacks.ModelCheckpoint(filepath=save_model_dir, monitor='val_loss',save_best_only=True,save_weights_only=True)
             ]
H=model.fit(tf_train_dataset, validation_data=tf_validation_dataset, epochs=epochs,callbacks=callbacks)

test_tokenized=test_pre_processing("test_just_reviews.txt")

tf_test_dataset = test_tokenized.to_tf_dataset(
    columns=["attention_mask", "input_ids"],
    shuffle=False,
    collate_fn=data_collator,
    batch_size=batch_size,
)

preds=get_predictions(model,tf_test_dataset)
preds_labels=reverse_mapping(preds)

with open("test_just_reviews.txt", "r") as file:
    content = file.read()
content=content.splitlines()
df=pd.DataFrame(content,columns=["REVIEWS"])
df["LABEL"]=preds_labels
df["LABEL"].to_csv("results.txt",index=False,header=False)