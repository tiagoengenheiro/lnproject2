import torch
import pandas as pd
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from datasets import Dataset
from transformers import TrainingArguments, Trainer
import numpy as np
import evaluate




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
model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased",num_labels=4)

training_args = TrainingArguments(output_dir="test_trainer")

metric = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


training_args = TrainingArguments(
    output_dir="test_trainer", 
    evaluation_strategy="epoch",
    learning_rate= 1e-05,
    num_train_epochs=1,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    warmup_steps=600,
    )

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    compute_metrics=compute_metrics,
)

trainer.train()
trainer.save_model("model")
# inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
# with torch.no_grad():
#     logits = model(**inputs).logits

# predicted_class_id = logits.argmax().item()
# print(predicted_class_id)