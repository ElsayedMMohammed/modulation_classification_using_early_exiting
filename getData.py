import numpy as np
import pickle 
import random
import torch
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import Dataset, DataLoader


# Seed for reproducibility
SEED = 25
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class myDataset(Dataset):
    def __init__(self, data_dic, transform = None):
        self.data = data_dic["data"]
        self.labels = data_dic["labels"]
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        label = self.labels[idx]
        signal = self.data[idx]

        if self.transform:
            signal = self.transform(signal)
        return signal, label


class getData():
    def __init__(self, splitting_ratio=0.2):
        self.splitting_ratio = splitting_ratio
        self.dataset, self.allXs, self.allLabels = [], [], []
        self.snrs, self.mods =0, 0
        self.devXs, self.devLabels, self.dev_idx, self.test_idx = 0, 0, 0, 0
        self.train_idx, self.valid_idx = 0, 0
        self.label_dict = {'QAM16': 0, 'QAM64': 1, '8PSK': 2, 'WBFM': 3, 'BPSK': 4, 'CPFSK': 5, 'AM-DSB': 6, 'GFSK': 7,
                'PAM4': 8, 'QPSK': 9}
        self.L = 128
        self.trainLoader, self.valLoader, self.testLoader = 0, 0, 0
        self.read_data()

    def read_data(self):
        
        with (open("./Data/RML22.pickle.01A", "rb")) as openfile:
            while True:
                try:
                    self.dataset.append(pickle.load(openfile, encoding='latin1'))
                except EOFError:
                    break
        self.dataset = self.dataset[0]

        # read the keys - snrs and mods.
        self.snrs, self.mods = map(lambda j: sorted(list(set(map(lambda x: x[j], self.dataset.keys())))), [1, 0])

        for mod in self.mods:
            for snr in self.snrs:
                self.allXs.append(self.dataset[(mod, snr)][0:2000])
                for i in range(2000):
                    self.allLabels.append((mod, snr))
        self.allXs = np.vstack(self.allXs)
        self.allLabels = np.array(self.allLabels)

        # The best wasy for train/valid/test splits 
        splitting_dev_test = StratifiedShuffleSplit(n_splits=2, test_size=self.splitting_ratio, random_state=SEED)
        self.dev_idx, self.test_idx = next(splitting_dev_test.split(self.allXs, self.allLabels))
        self.devXs, self.devLabels = self.allXs[self.dev_idx], self.allLabels[self.dev_idx]

        self.train_idx, self.valid_idx = next(splitting_dev_test.split(self.devXs, self.devLabels))

        self.trainXs, self.trainLabels = self.devXs[self.train_idx], self.devLabels[self.train_idx]
        self.valXs, self.valLabels = self.devXs[self.valid_idx], self.devLabels[self.valid_idx]
        self.testXs, self.testLabels = self.allXs[self.test_idx], self.allLabels[self.test_idx]
    
        print("Training: ", self.trainXs.shape, self.trainLabels.shape)
        print("Validation: ",self.valXs.shape, self.valLabels.shape)
        print("Testing: ",self.testXs.shape, self.testLabels.shape)

        # Convert train labels
        label_val = list(map(lambda x: self.trainLabels[x][0], range(len(self.trainLabels))))
        label = list(map(lambda x: self.label_dict[x], label_val))
        label = np.array(label)
        data = self.trainXs[:, :, 0:self.L]
        train_set = {'data': torch.tensor(data).float(), 'labels': torch.tensor(label).float()}

        # Convert validation labels
        label_val = list(map(lambda x: self.valLabels[x][0], range(len(self.valLabels))))
        label = list(map(lambda x: self.label_dict[x], label_val))
        label = np.array(label)
        data = self.valXs[:, :, 0:self.L]
        valid_set = {'data': torch.tensor(data).float(), 'labels': torch.tensor(label).float()}

        # Convert test labels
        label_val = list(map(lambda x: self.testLabels[x][0], range(len(self.testLabels))))
        label = list(map(lambda x: self.label_dict[x], label_val))
        label = np.array(label)
        data = self.testXs[:, :, 0:self.L]
        test_set = {'data': torch.tensor(data).float(), 'labels': torch.tensor(label).float()}

        trainSet = myDataset(train_set)
        valSet = myDataset(valid_set)
        testSet = myDataset(test_set)

        self.trainLoader = DataLoader(trainSet, batch_size= 500, shuffle = True, num_workers= 0)
        self.valLoader = DataLoader(valSet, batch_size = 500, shuffle = False)
        self.testLoader = DataLoader(testSet, batch_size = 500, shuffle = False)

    def get_loaders(self):
        return self.trainLoader, self.valLoader, self.testLoader
    
    def get_snrs(self):
        return self.snrs
    
    def get_Xs(self):
        return self.trainXs, self.valXs, self.testXs
    
    def get_labels(self):
        return self.trainLabels, self.valLabels, self.testLabels
    
    def get_label_dict(self):
        return self.label_dict