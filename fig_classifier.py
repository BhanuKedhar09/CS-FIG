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
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
from transformers import DefaultDataCollator
import evaluate
from transformers import AutoModelForImageClassification, TrainingArguments, Trainer
from transformers import AutoConfig, TFAutoModelForTableQuestionAnswering
from transformers import ViTForImageClassification
from transformers import AdamW
from torch.utils.data import DataLoader
import torch
from torch.nn.functional import softmax
from transformers import ViTForImageClassification, ViTFeatureExtractor
from transformers import TrainingArguments, Trainer
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from transformers import DefaultDataCollator




dataset = load_dataset("citeseerx/ACL-fig")

checkpoint = "google/vit-large-patch16-224-in21k"
image_processor = AutoImageProcessor.from_pretrained(checkpoint)
feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-large-patch16-224-in21k')
model = ViTForImageClassification.from_pretrained('google/vit-large-patch16-224-in21k')  # Load with default classifier
model.classifier = torch.nn.Linear(model.config.hidden_size, 19)

transforms_ = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.Resize([224, 224]),
    transforms.ToTensor(),
    transforms.Normalize(mean=feature_extractor.image_mean, std=feature_extractor.image_std),
])

def transforms_image(examples):
    examples["pixel_values"] = [(transforms_(image.convert("RGB"))) for image in examples["image"]]
    return examples

dat = dataset.map(transforms_image, batched=True,remove_columns=['image'])

dat["train"].set_format(type = "torch", columns=["pixel_values", "label"], output_all_columns= True)
dat["validation"].set_format(type = "torch", columns=["pixel_values", "label"], output_all_columns= True,)
dat["test"].set_format(type = "torch", columns=["pixel_values", "label"], output_all_columns= True)

data_collater = DefaultDataCollator(return_tensors="pt")

datatrain_loader = DataLoader(dat["train"], batch_size=152, shuffle=True, collate_fn=data_collater)
datavalid_loader = DataLoader(dat["validation"], batch_size=152, shuffle=True, collate_fn=data_collater)
datatest_loader = DataLoader(dat["test"], batch_size=152, shuffle=True, collate_fn=data_collater)

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
    acc = accuracy_score(labels, preds)
    return {
        "accuracy": acc,
        "f1": f1,
        "precision": precision,
        "recall": recall
    }
    
accuracy = evaluate.load("accuracy")

model = AutoModelForImageClassification.from_pretrained(
    checkpoint,
    num_labels=19,
    # id2label=id2label,
    # label2id=label2id,
)


training_args = TrainingArguments(
    output_dir="my_awesome_fig_model",
    remove_unused_columns=False,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=152,
    gradient_accumulation_steps=4,
    per_device_eval_batch_size=152,
    num_train_epochs=3,
    warmup_ratio=0.1,
    logging_steps=10,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    push_to_hub=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collater,
    train_dataset=dat["train"],
    eval_dataset=dat["validation"],
    tokenizer=image_processor,
    compute_metrics=compute_metrics,
)

trainer.train()