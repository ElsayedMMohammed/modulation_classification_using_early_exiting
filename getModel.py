import torch
import torch.nn as nn
import torch.nn.functional as func


# Seed for reproducibility
SEED = 25
torch.manual_seed(SEED)


class eeModel(nn.Module):

    def __init__(self, num_classes=10, input_dim = 2):
        super(eeModel, self).__init__()
        
        self.baseModel = nn.Sequential(
            nn.Conv1d(in_channels=input_dim, out_channels=64, kernel_size=3,padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),

            nn.MaxPool1d(kernel_size=2, stride=2,padding = 0),

            nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),

            nn.MaxPool1d(kernel_size=2, stride=2, padding = 0),

            nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),

        )
        
        self.shortBranch = nn.Sequential(
            nn.MaxPool1d(kernel_size=2, stride=2,padding = 0),
            nn.Flatten(),
            nn.Linear(in_features=1024, out_features=256),
            nn.Dropout(0.5),
            nn.Linear(in_features=256, out_features=64),
            nn.Dropout(0.5),
            nn.Linear(in_features=64, out_features=num_classes)
        )
        
        self.longBranch = nn.Sequential(
            nn.Conv1d(in_channels=64, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            
            nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            
            nn.Conv1d(in_channels=32, out_channels=16, kernel_size=3, padding=1),
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=True),

            nn.MaxPool1d(kernel_size=2, stride=2, padding = 0),
            nn.Flatten(),
            nn.Linear(in_features=256, out_features=256),
            nn.Dropout(0.5),
            nn.Linear(in_features=256, out_features=64),
            nn.Dropout(0.5),
            nn.Linear(in_features=64, out_features=num_classes)
        )
   
                
    def forward(self, X):
        X = self.baseModel(X)
        X1 = self.shortBranch(X)
        X2 = self.longBranch(X)
        
        return X1, X2


class blModel(nn.Module):

    def __init__(self, num_classes=10, input_dim = 2):
        super(blModel, self).__init__()
        
        self.baseModel = nn.Sequential(
            nn.Conv1d(in_channels=input_dim, out_channels=64, kernel_size=3,padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),

            nn.MaxPool1d(kernel_size=2, stride=2,padding = 0),

            nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),

            nn.MaxPool1d(kernel_size=2, stride=2, padding = 0),

            nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),

        )
        
        
        self.longBranch = nn.Sequential(
            nn.Conv1d(in_channels=64, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            
            nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            
            nn.Conv1d(in_channels=32, out_channels=16, kernel_size=3, padding=1),
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=True),

            nn.MaxPool1d(kernel_size=2, stride=2, padding = 0),
            nn.Flatten(),
            nn.Linear(in_features=256, out_features=256),
            nn.Dropout(0.5),
            nn.Linear(in_features=256, out_features=64),
            nn.Dropout(0.5),
            nn.Linear(in_features=64, out_features=num_classes)
        )
   
                
    def forward(self, X):
        X = self.baseModel(X)
        X = self.longBranch(X)     
        return X
