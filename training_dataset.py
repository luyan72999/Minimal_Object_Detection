import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset



class training_dataset(Dataset):
    def __init__(self):
        trainX = np.asarray(pd.read_csv('./Dataset/trainingData.csv', sep=',', header=None))
        trainY = np.asarray(pd.read_csv('./Dataset/ground-truth.csv', sep=',', header=None))

        self.features_train = torch.Tensor(trainX / 255.0) # Scaling images
        self.groundTruth_train = torch.Tensor(trainY / 100.0)  # Scaling coordinates

        self.len = len(trainX)

    def __getitem__(self, item):
        return self.features_train[item], self.groundTruth_train[item]

    def __len__(self):
        return self.len
