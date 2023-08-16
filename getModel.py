import torch
import torch.nn as nn
import torch.nn.functional as func
import numpy as np

# Seed for reproducibility
SEED = 25
torch.manual_seed(SEED)
np.random.seed(SEED)

class eeModel(nn.Module):

    def __init__(self, num_classes=10, input_dim = 2):
        super(eeModel, self).__init__()
        
        self.baseModel = None
        self.shortBranch = None
        self.longBranch = None
   
              
    def forward(self, X):
        X = self.baseModel(X)
        X1 = self.shortBranch(X)
        X2 = self.longBranch(X)
        
        return X1, X2
    
    def short_forward(self, X):
        X = self.baseModel(X)
        X = self.shortBranch(X)
        return X
    
    def long_forward(self, X):
        X = self.baseModel(X)
        X = self.longBranch(X)
        return X
    
    def forward_timeTest(self, X):
        selected_data = X[0: 50]
        remaining_data = X[50: X.shape[0]]
        _ = self.short_forward(selected_data)
        _ = self.long_forward(remaining_data)
    
    def print_summary(self):
        basemodel_total_params = sum(p.numel() for p in self.baseModel.parameters())
        basemodel_trainable_params = sum(p.numel() for p in self.baseModel.parameters() if p.requires_grad)

        shortBranch_total_params = sum(p.numel() for p in self.shortBranch.parameters())
        shortBranch_trainable_params = sum(p.numel() for p in self.shortBranch.parameters() if p.requires_grad)

        longBranch_total_params = sum(p.numel() for p in self.longBranch.parameters())
        longBranch_trainable_params = sum(p.numel() for p in self.longBranch.parameters() if p.requires_grad)

        print("Number of base parameters: {}".format(basemodel_total_params))
        print("Number of short branch parameters: {}".format(shortBranch_total_params))
        print("Number of long branch parameters: {}".format(longBranch_total_params))
        print("Difference = {}".format(longBranch_total_params-shortBranch_total_params))
    
    def compute_size(self):
        param_size = 0
        for param in self.baseModel.parameters():
            param_size += param.nelement() * param.element_size()
        for param in self.shortBranch.parameters():
            param_size += param.nelement() * param.element_size()
        for param in self.longBranch.parameters():
            param_size += param.nelement() * param.element_size()

        buffer_size = 0
        for buffer in self.baseModel.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        for buffer in self.shortBranch.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        for buffer in self.longBranch.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()

        size_all_mb = (param_size + buffer_size) / 1024**2
        return size_all_mb
    
    def compute_short_branch_size(self):
        param_size = 0
        for param in self.baseModel.parameters():
            param_size += param.nelement() * param.element_size()
        for param in self.shortBranch.parameters():
            param_size += param.nelement() * param.element_size()

        buffer_size = 0
        for buffer in self.baseModel.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        for buffer in self.shortBranch.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()

        size_all_mb = (param_size + buffer_size) / 1024**2
        return size_all_mb
    
    def compute_long_branch_size(self):
        param_size = 0
        for param in self.baseModel.parameters():
            param_size += param.nelement() * param.element_size()
        for param in self.longBranch.parameters():
            param_size += param.nelement() * param.element_size()

        buffer_size = 0
        for buffer in self.baseModel.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        for buffer in self.longBranch.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()

        size_all_mb = (param_size + buffer_size) / 1024**2
        return size_all_mb


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
    




class eeModel_V0(eeModel):

    """
        A class for the model in which the short branch is the closest to the baseModel compared to the other models.
    """
    def __init__(self, num_classes=10, input_dim = 2):
        super(eeModel_V0, self).__init__()

        self.baseModel = nn.Sequential(
            nn.Conv1d(in_channels=input_dim, out_channels=64, kernel_size=3,padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True), 
            nn.MaxPool1d(kernel_size=2, stride=2,padding = 0),         
        )
        
        self.shortBranch = nn.Sequential(
            nn.MaxPool1d(kernel_size=4, stride=4,padding = 0),
            nn.Flatten(),
            nn.Linear(in_features=1024, out_features=64),
            nn.Dropout(0.5),
            nn.Linear(in_features=64, out_features=32),
            nn.Dropout(0.5),
            nn.Linear(in_features=32, out_features=num_classes)
        )
        
        self.longBranch = nn.Sequential(
            nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),

            nn.MaxPool1d(kernel_size=2, stride=2, padding = 0),

            nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            
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
   
        self.print_summary()

    def forward(self, X):
        X = self.baseModel(X)
        X1 = self.shortBranch(X)
        X2 = self.longBranch(X)
        
        return X1, X2
    

class eeModel_V1(eeModel):

    def __init__(self, num_classes=10, input_dim = 2):
        super(eeModel_V1, self).__init__()
        
        self.baseModel = nn.Sequential(
            nn.Conv1d(in_channels=input_dim, out_channels=64, kernel_size=3,padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),

            nn.MaxPool1d(kernel_size=2, stride=2,padding = 0),

            nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),


        )
        
        self.shortBranch = nn.Sequential(
            nn.MaxPool1d(kernel_size=4, stride=4,padding = 0),
            nn.Flatten(),
            nn.Linear(in_features=1024, out_features=64),
            nn.Dropout(0.5),
            nn.Linear(in_features=64, out_features=32),
            nn.Dropout(0.5),
            nn.Linear(in_features=32, out_features=num_classes)
        )
        
        self.longBranch = nn.Sequential(
            nn.MaxPool1d(kernel_size=2, stride=2, padding = 0),

            nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            
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
   
        self.print_summary()

    def forward(self, X):
        X = self.baseModel(X)
        X1 = self.shortBranch(X)
        X2 = self.longBranch(X)
        
        return X1, X2


class eeModel_V3(eeModel):

    """
        A class for the model in which the short branch is the closest to the long branch compared to the other models.
    """

    def __init__(self, num_classes=10, input_dim = 2):
        super(eeModel_V3, self).__init__()
        
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
            
            nn.Conv1d(in_channels=64, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            
            nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),

        )
        
        self.shortBranch = nn.Sequential(
            #nn.MaxPool1d(kernel_size=4, stride=4,padding = 0),
            nn.Flatten(),
            nn.Linear(in_features=1024, out_features=64),
            nn.Dropout(0.5),
            nn.Linear(in_features=64, out_features=32),
            nn.Dropout(0.5),
            nn.Linear(in_features=32, out_features=num_classes)
        )
        
        self.longBranch = nn.Sequential(
            
            
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
   
        self.print_summary()

    def forward(self, X):
        X = self.baseModel(X)
        X1 = self.shortBranch(X)
        X2 = self.longBranch(X)
        
        return X1, X2
    

class eeModel_V2(eeModel):

    """
        A class for the model in which the short branch is the closest to the long branch compared to the other models.
    """

    def __init__(self, num_classes=10, input_dim = 2):
        super(eeModel_V2, self).__init__()
        
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
            nn.Linear(in_features=1024, out_features=64),
            nn.Dropout(0.5),
            nn.Linear(in_features=64, out_features=32),
            nn.Dropout(0.5),
            nn.Linear(in_features=32, out_features=num_classes)
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
   
        self.print_summary()

    def forward(self, X):
        X = self.baseModel(X)
        X1 = self.shortBranch(X)
        X2 = self.longBranch(X)
        
        return X1, X2
    

"""
class eeModel_V3(eeModel):

        #A class for the model in which the short branch is the closest to the long branch compared to the other models.
  

    def __init__(self, num_classes=10, input_dim = 2):
        super(eeModel_V3, self).__init__()
        
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
            
            nn.Conv1d(in_channels=64, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            
            nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            
        )
        
        self.shortBranch = nn.Sequential(
            #nn.MaxPool1d(kernel_size=4, stride=4,padding = 0),
            nn.Flatten(),
            nn.Linear(in_features=1024, out_features=64),
            nn.Dropout(0.5),
            nn.Linear(in_features=64, out_features=32),
            nn.Dropout(0.5),
            nn.Linear(in_features=32, out_features=num_classes)
        )
        
        self.longBranch = nn.Sequential(
             
            ##########
            
            nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
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
   
        self.print_summary()

    def forward(self, X):
        X = self.baseModel(X)
        X1 = self.shortBranch(X)
        X2 = self.longBranch(X)
        
        return X1, X2
    
"""