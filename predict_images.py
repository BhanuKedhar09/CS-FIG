from PIL import Image
from torchvision import transforms
import torch
from transformers import ViTImageProcessor
from transformers import ViTForImageClassification, AutoFeatureExtractor
import requests
import io
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
from transformers import AutoModelForImageClassification, AutoImageProcessor


swin_fine_tine = "/lstr/sahara/datalab-ml/z1974769/classifier/swin_classifier_results/checkpoint-804"
google_vit_fine_tune = "/lstr/sahara/datalab-ml/z1974769/classifier/classifier_results/checkpoint-804"
# image = Image.open("/lstr/sahara/datalab-ml/z1974769/outputs_pdffigures/real_outputs/figure_images1601.05647-Figure4-1.png")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

processor = AutoImageProcessor.from_pretrained(swin_fine_tine)
swin_model = AutoModelForImageClassification.from_pretrained(swin_fine_tine)
google_vit_model = AutoModelForImageClassification.from_pretrained(google_vit_fine_tune)

df = pd.read_json("/lstr/sahara/datalab-ml/z1974769/outputs_pdffigures/real_outputs/all_files_data_combined.json")
df["Fig_name"] = df["renderURL"].str.split("/").str[-1]




def swin_prediction(row):
    img_path = row["renderURL"]
    image = Image.open(img_path)
    inputs = processor(images=image, return_tensors="pt").to(device)
    swin_model.to(device)
    pixel_values = inputs.pixel_values
    with torch.no_grad():
        outputs = swin_model(pixel_values)
    logits = outputs.logits
    probabilities = torch.softmax(logits, dim=1)
    prediction_confidence, predicted_class = torch.max(probabilities, dim=1)
    # prediction = logits.argmax(-1)
    # return swin_model.config.id2label[prediction.item()]
    return [swin_model.config.id2label[predicted_class.item()],prediction_confidence.item()]


def google_vit_prediction(row):
    img_path = row["renderURL"]
    image = Image.open(img_path)
    inputs = processor(images=image, return_tensors="pt").to(device)
    google_vit_model.to(device)
    pixel_values = inputs.pixel_values
    with torch.no_grad():
        outputs = google_vit_model(pixel_values)
    logits = outputs.logits
    # prediction = logits.argmax(-1)
    probabilities = torch.softmax(logits, dim=1)
    prediction_confidence, predicted_class = torch.max(probabilities, dim=1)
    # return google_vit_model.config.id2label[prediction.item()]
    return [google_vit_model.config.id2label[predicted_class.item()],prediction_confidence.item()]



df[["Swin_prediction", "Swin_confidence"]] = df.apply(swin_prediction, axis=1, result_type="expand")
df[["Goolge_ViT_prediction", "Goolge_ViT_confidence"]] = df.apply(google_vit_prediction, axis=1, result_type="expand")


df.to_csv("/lstr/sahara/datalab-ml/z1974769/outputs_pdffigures/real_outputs/all_files_data_combined_with_predictions.csv", index=False)
column1 = pd.Series(df["Goolge_ViT_prediction"])
column2 = pd.Series(df["Swin_prediction"])

same_values_count = (column1 == column2).sum()

# Count rows with different values
different_values_count = (column1 != column2).sum()

print("Number of rows with same values:", same_values_count)
print("Number of rows with different values:", different_values_count)

