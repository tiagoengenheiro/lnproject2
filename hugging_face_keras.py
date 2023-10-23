import pandas as pd
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from datasets import Dataset
from transformers import TrainingArguments, Trainer
import matplotlib.pyplot as pls
import numpy as np
import evaluate
#from official.nlp import optimization

df=pd.read_csv("train.txt",sep="\t",names=["LABEL","REVIEW"])
dataset = Dataset.from_pandas(df)
dataset=dataset.rename_columns({"LABEL":"label","REVIEW":"text"})
dataset=dataset.train_test_split(test_size=0.2,shuffle=True)

tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
def label_mapping(example):
    mapping={
        "DECEPTIVENEGATIVE":0,
        "DECEPTIVEPOSITIVE":1,
        "TRUTHFULNEGATIVE":2,
        "TRUTHFULPOSITIVE":3,
    }
    return {"label":mapping[example["label"]]}
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

dataset=dataset.map(label_mapping,batched=False)
print(dataset["train"][0])
tokenized_datasets = dataset.map(tokenize_function, batched=True)
tokenized_datasets = tokenized_datasets.remove_columns(["text"])
tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
print(tokenized_datasets["train"][0])

#tokenized_datasets["train"] = tokenized_datasets["train"].shuffle(seed=42).select(range(100))
#tokenized_datasets["test"] = tokenized_datasets["test"].shuffle(seed=42).select(range(10))

import tensorflow as tf
from transformers import TFDistilBertForSequenceClassification
model = TFDistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased",num_labels=4)
from transformers import DefaultDataCollator

data_collator = DefaultDataCollator(return_tensors="tf")
batch_size=8
epochs=6
tf_train_dataset = tokenized_datasets["train"].to_tf_dataset(
    columns=["attention_mask", "input_ids"],
    label_cols=["labels"],
    shuffle=True,
    collate_fn=data_collator,
    batch_size=batch_size,
)

tf_validation_dataset = tokenized_datasets["test"].to_tf_dataset(
    columns=["attention_mask", "input_ids"],
    label_cols=["labels"],
    shuffle=False,
    collate_fn=data_collator,
    batch_size=batch_size,
)
# steps_per_epoch = tf.data.experimental.cardinality(tf_train_dataset).numpy()
# num_train_steps = steps_per_epoch * epochs
# num_warmup_steps = int(0.1*num_train_steps)

# init_lr = 3e-5
# optimizer = optimization.create_optimizer(init_lr=init_lr,
#                                           num_train_steps=num_train_steps,
#                                           num_warmup_steps=num_warmup_steps,
#                                           optimizer_type='adamw')
model.compile(
    optimizer=tf.keras.optimizers.AdamW(learning_rate=1e-5),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=tf.metrics.SparseCategoricalAccuracy(),
)
callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=2),
             tf.keras.callbacks.ModelCheckpoint(filepath='best_model.h5', monitor='val_loss', save_best_only=True)]
H=model.fit(tf_train_dataset, validation_data=tf_validation_dataset, epochs=epochs,callbacks=callbacks)