import gc
import librosa
import math
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import random
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
from glob import glob
from plotly.subplots import make_subplots
from scipy.signal import butter, lfilter
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm
from sklearn.model_selection import train_test_split

import config
from preprocessing_functions import *
from train import train_model
from predict import predict_imu
from model import lstm_res
from CustomData import CustomDataset

def pipeline(config, model,num_epochs, save: bool = True,save_dir:str):

  df_train = pd.read_csv(config.TRAIN_CSV)
  df_test = pd.read_csv(config.TEST_CSV)
  
  df_train["target"] = df_train["gesture"].map(config.label_to_num)
  
  demo = df_train[['sequence_id','target']].drop_duplicates()
    
  train_seq, val_seq, train_y, val_y = train_test_split(
      demo['sequence_id'],               
      demo['target'],              
      test_size=0.20,   
      train_size=None,  
      random_state=42,  
      shuffle=True,     
      stratify=demo['target']      
  )
  
  train = df_train[df_train.sequence_id.isin(train_seq)]
  val = df_train[df_train.sequence_id.isin(val_seq)]

  train_X, train_y = [], []

  for sequence_id in tqdm(train_seq):
      ds = train[train["sequence_id"] == sequence_id]
      X = ds[imu_cols].values
      y = ds.target.values[0]
      acc = standard_scale(X[:, 0:3])
      rot = X[:, 3:]
      X = np.concatenate([acc, rot], axis=1)
      X = np.where(np.isnan(X), 0, X)  # fill NaNs
      train_X.append(X)
      train_y.append(y)
  
  train_y = np.array(train_y)
  
  val_X, val_y = [], []
  
  for sequence_id in tqdm(val_seq):
      ds = val[val["sequence_id"] == sequence_id]
      X = ds[imu_cols].values
      y = ds.target.values[0]
      acc = standard_scale(X[:, 0:3])
      rot = X[:, 3:]
      X = np.concatenate([acc, rot], axis=1)
      X = np.where(np.isnan(X), 0, X)  # fill NaNs
      val_X.append(X)
      val_y.append(y)
  
  val_y = np.array(val_y)
  test_X = []
  sequence_ids_test = df_test.sequence_id.unique()
  
  for sequence_id in tqdm(sequence_ids_test):
      ds = df_test[df_test["sequence_id"] == sequence_id]
      X = ds[imu_cols].values
      acc = standard_scale(X[:, 0:3])
      rot = X[:, 3:]
      X = np.concatenate([acc, rot], axis=1)
      X = np.where(np.isnan(X), 0, X)  # fill NaNs
      test_X.append(X)

  transforms = Compose([
    OneOf([
        GaussianNoise(p=0.5, max_noise_amplitude=0.05),
        PinkNoiseSNR(p=0.5, min_snr=4.0, max_snr=20.0),
        ButterFilter(p=0.5)
    ]),
    TimeStretch(p=0.25),
    TimeShift(p=0.25)

  ])
  train_dataset = CustomDataset(config,train, train_X,  train_y, transforms, mode="train")
  val_dataset = CustomDataset(config,val, val_X,  val_y, mode="val")
  test_dataset =  CustomDataset(config,df = df_test,X = test_X,mode="test")
  
  model = model
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  model = model.to(device)
  
  # optimizer = optim.Adam(model.parameters(), lr=0.005)   
  # criterion = nn.KLDivLoss(reduction="batchmean")
  
  train_loader = DataLoader(train_dataset, batch_size=64, shuffle=False)
  val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
  test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False) 
  
  gc.collect()
  torch.cuda.empty_cache()
  train_model(model, train_loader, config.criterion, config.optimizer, num_epochs=20)

  if save == True:
    # save_dir = "/kaggle/working"
    os.makedirs(save_dir, exist_ok=True)
    model_path = os.path.join(save_dir, "lstm_res_005_64.pth")
    torch.save(model.state_dict(), model_path)

  model.eval()
  predict_val = predict_imu(model, val_loader, device=device)
  predict_test = predict_imu(model, test_loader, device=device)

  acc = accuracy_score(val_y, predict_val)

  print("########Model fitting and predicting Complete########"
  print("Accuracy:", acc)
  print(predict_test)

if __name__ == "main":
  model= lstm_res()
  pipeline(config, model,num_epochs=20, save=True,save_dir= config.save_dir)
  


 
 
