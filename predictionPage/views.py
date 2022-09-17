from django.shortcuts import render, redirect
from django.http import HttpResponse
import csv

from predictionPage.model import predict
from predictionPage.plot import plotImage

import sentencepiece
import transformers
from transformers import XLNetTokenizer, XLNetModel, AdamW, get_linear_schedule_with_warmup
import torch

import numpy as np
import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt
# from matplotlib import rc
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
# from collections import defaultdict
# from textwrap import wrap
# from pylab import rcParams

from torch import nn, optim
from keras.preprocessing.sequence import pad_sequences
from torch.utils.data import TensorDataset,RandomSampler,SequentialSampler
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F


# import io
# from google.colab import files

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device

from transformers import XLNetTokenizer, XLNetModel
PRE_TRAINED_MODEL_NAME = 'xlnet-base-cased'
tokenizer = XLNetTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)

#restart runtime if doesnot work
input_txt = "Nepal is my country. All Nepalese are my brothers and sisters"
encodings = tokenizer.encode_plus(input_txt, add_special_tokens=True, 
                                  max_length=16, return_tensors='pt', return_token_type_ids=False,
                                  return_attention_mask=True, pad_to_max_length=False)

attention_mask = pad_sequences(encodings['attention_mask'], maxlen=512, dtype=torch.Tensor ,truncating="post",padding="post")

attention_mask = attention_mask.astype(dtype = 'int64')
attention_mask = torch.tensor(attention_mask) 
attention_mask.flatten()

from transformers import XLNetTokenizer, XLNetModel
PRE_TRAINED_MODEL_NAME = 'xlnet-base-cased'
tokenizer = XLNetTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)
MAX_LEN = 512

tokenizer

from transformers import XLNetForSequenceClassification
model = XLNetForSequenceClassification.from_pretrained('xlnet-base-cased', num_labels = 3)
model = model.to(device)

model




SN = []
Date = []
Title = []
Sentiment = []

def csv_file():
    with open("csv/hi.csv") as file:
        reader = csv.DictReader(file)
        for row in reader:
            SN.append(row["SN"])
            Date.append(row["Date"])
            Title.append(row["Title"])
            Sentiment.append(row["Sentiment"])

def indexPage(request):
    return render(request, "index.html", {})

def resultPage(request):
    csv = request.GET['csv']
    model = request.GET['model']
    result, val = predict(csv, model)
    return render(request, 'result.html', {"result": result, "value": val})

def detailPage(request):
    images = plotImage()
    return render(request, 'detail.html', {"first_image": images[0], "second_image": images[1], "third_image": images[2], "fourth_image": images[3]})

# Create your views here.
def newsPage(request): 
    csv_file()
    return render(request, 'news.html', {"Title": Title})

def sentimentPage(request, index):
    index = int(index)
    probs = predict_sentiment(index-1)
    negative, positive, neutral = probs[0], probs[1], probs[2]

    return render(request, 'sentimentPage.html', {"negative": negative,"positive": positive,  "neutral": neutral})

def predict_sentiment(index):
    model.load_state_dict(torch.load('model/xlnet_model.bin',map_location=torch.device('cpu')))
    news_text = Title[index]

    encoded_news = tokenizer.encode_plus(
    news_text,
    max_length=MAX_LEN,
    add_special_tokens=True,
    return_token_type_ids=False,
    pad_to_max_length=False,
    return_attention_mask=True,
    return_tensors='pt',
    )

    input_ids = pad_sequences(encoded_news['input_ids'], maxlen=MAX_LEN, dtype=torch.Tensor ,truncating="post",padding="post")
    input_ids = input_ids.astype(dtype = 'int64')
    input_ids = torch.tensor(input_ids) 

    attention_mask = pad_sequences(encoded_news['attention_mask'], maxlen=MAX_LEN, dtype=torch.Tensor ,truncating="post",padding="post")
    attention_mask = attention_mask.astype(dtype = 'int64')
    attention_mask = torch.tensor(attention_mask) 

    input_ids = input_ids.reshape(1,512).to(device)
    attention_mask = attention_mask.to(device)

    outputs = model(input_ids=input_ids, attention_mask=attention_mask)

    outputs = outputs[0][0].cpu().detach()


    probs = F.softmax(outputs, dim=-1).cpu().detach().numpy().tolist()
  
    _, prediction = torch.max(outputs, dim =-1)


    # print("Positive score:", probs[1])
    # print("Negative score:", probs[0])
    # print("Neutral score:", probs[2])
    # print(f'News text: {news_text}')
    # print(f'Sentiment  : {class_names[prediction]}')
    return probs


# if __name__ == "__main__":
#     with open(r'C:\Users\acer\Desktop\stockPrediction\stockPrediction\csv\News_dataset.csv') as file:
#         reader = csv.DictReader(file)
    
#         data = []
#         for row in reader:
#             data.append(row)

#         print(data)