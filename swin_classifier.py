from datasets import load_dataset 
from transformers import AutoImageProcessor
from datasets import load_metric
from datasets import load_dataset
import pandas as pd
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from transformers import AutoImageProcessor
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from torchvision.transforms import ToPILImage
from PIL import Image
from sklearn.metrics import classification_report
from operator import itemgetter
from torchvision.transforms import Resize, Compose, ToTensor, Normalize, RandomResizedCrop
from transformers import AutoFeatureExtractor
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torchvision.transforms.functional as TF
import matplotlib.ticker as ticker
from transformers import AutoFeatureExtractor
from torch.utils.data import Dataset, DataLoader
import os
from transformers import DefaultDataCollator
import evaluate
from transformers import ViTImageProcessor
from transformers import AutoModelForImageClassification, TrainingArguments, Trainer
from transformers import AutoConfig, TFAutoModelForTableQuestionAnswering
from transformers import ViTForImageClassification
from transformers import AdamW, SwinForImageClassification
from torch.utils.data import DataLoader
import torch
from torch.nn.functional import softmax
from transformers import ViTForImageClassification, ViTFeatureExtractor
from transformers import TrainingArguments, Trainer
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from transformers import DefaultDataCollator
from torchvision.transforms import CenterCrop, Compose, Normalize, RandomHorizontalFlip,RandomResizedCrop, Resize, ToTensor





dataset = load_dataset("citeseerx/ACL-fig")

train_ds = dataset['train']
val_ds = dataset['validation']
test_ds = dataset['test']

labels = dataset["train"].features["label"].names

label2id, id2label = dict(), dict()
for i, label in enumerate(labels):
    label2id[label] = i
    id2label[i] = label

metric = load_metric("accuracy")

# model_checkpoint = "microsoft/swin-tiny-patch4-window7-224" # pre-trained model from which to fine-tune
model_checkpoint = "microsoft/swin-tiny-patch4-window7-224"
image_processor  = AutoImageProcessor.from_pretrained(model_checkpoint)
model = AutoModelForImageClassification.from_pretrained(
    model_checkpoint, 
    label2id=label2id,
    id2label=id2label,
    ignore_mismatched_sizes = True, # provide this in case you're planning to fine-tune an already fine-tuned checkpoint
)

transforms_ = transforms.Compose([
    transforms.Resize([image_processor.size["height"], image_processor.size["width"]]),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize(mean=image_processor.image_mean, std=image_processor.image_std),
])

def transforms_image(examples):
    examples["pixel_values"] = [transforms_(image.convert("RGB")) for image in examples["image"]]
    return examples

train_ds.set_transform(transforms_image)
val_ds.set_transform(transforms_image)
test_ds.set_transform(transforms_image)

print(train_ds[0])


def collate_fn(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    labels = torch.tensor([example["label"] for example in examples])
    return {"pixel_values": pixel_values, "labels": labels}

train_dataloader = DataLoader(train_ds, collate_fn=collate_fn, batch_size=4)



batch = next(iter(train_dataloader))
for k,v in batch.items():
  if isinstance(v, torch.Tensor):
    print("Printing the shape of the tensor")
    print(k, v.shape)
    print("*"*150)

metric_name = "accuracy"

args = TrainingArguments(
    output_dir = "./swin_classifier_results",
    save_strategy="epoch",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=10,
    per_device_eval_batch_size=4,
    num_train_epochs=3,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model=metric_name,
    logging_dir='./swin_classifier_results/logs',
    remove_unused_columns=False,
)


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return dict(accuracy=accuracy_score(predictions, labels))


def compute_metrics2(pred):
    preds, labels = pred
    preds = np.argmax(preds, axis=1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
    acc = accuracy_score(labels, preds)
    c_report = classification_report(labels, preds)
    print(c_report)
    return {
        "accuracy": acc,
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "classification_report" : c_report
    }
 

trainer = Trainer(
    model,
    args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    data_collator=collate_fn,
    compute_metrics=compute_metrics2,
    tokenizer=image_processor,
)

trainer.train()

print("Training completed. Evaluating on validation dataset...")
validation_results = trainer.evaluate(val_ds)
print("Swin Trasnformer Validation results:", validation_results)
print("*"*150)

with torch.no_grad():
    
    print("Swin Trasnformer Evaluating on test dataset...")
    test_results = trainer.evaluate(test_ds)
    print("Test results:", test_results)
    print("*"*150)


