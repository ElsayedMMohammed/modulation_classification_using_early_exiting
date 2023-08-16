
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as func
import torch.optim as optim
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedShuffleSplit
from scipy.stats import entropy
from getData import * 

# Seed for reproducibility
SEED = 25
torch.manual_seed(SEED)
np.random.seed(SEED)

class eeHandler():
    def __init__(self, net, criterion, optimizer, device, nBranches=2, scheduler=None, num_epochs=50, bestPath="./models/best_0420.pth"):
        self.net = net
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.NUM_EPOCHS = num_epochs
        self.device = device
        self.bestPath = bestPath
        self.nBranches = nBranches

        self.history = {"1": {"train": {"loss": [], "accuracy": []}, "validation":{"loss": [], "accuracy": []}},
              "2": {"train": {"loss": [], "accuracy": []}, "validation":{"loss": [], "accuracy": []}},
              "T": {"train": {"loss": [], "accuracy": []}, "validation":{"loss": [], "accuracy": []}}}
        
        
    def train(self, tLoader, vLoader=None):
        eStopThreshold, eStopCounter = 8, 0 
        best_loss, preValLoss = 100, 100

        for epoch in range(self.NUM_EPOCHS):
            loss1Total, loss2Total, totalLoss = 0, 0, 0
            acc1Total, acc2Total, totalAcc = 0, 0, 0
            loss1Total_v, loss2Total_v, valLoss = 0, 0, 0
            acc1Total_v, acc2Total_v, valAcc = 0, 0, 0

            self.net.train()
            for i, data in enumerate(tLoader, 0):
                inputs, labels = data[0].to(self.device), data[1].to(self.device)

                self.optimizer.zero_grad()
                output1, output2 = self.net(inputs)
                loss1, loss2 = self.criterion(output1, labels.long()), self.criterion(output2, labels.long())
                
                loss1Total += loss1.item()
                loss2Total += loss2.item()
                totalLoss += 0.5*loss1.item() + 0.5*loss2.item()
                
                loss1.backward(inputs=list(self.net.baseModel.parameters())+list(self.net.shortBranch.parameters()), retain_graph=True)
                loss2.backward(inputs=list(self.net.longBranch.parameters()))

                _, predicted1 = torch.max(output1, 1)
                _, predicted2 = torch.max(output2, 1)

                acc1 = accuracy_score(labels.detach().cpu().numpy(), predicted1.detach().cpu().numpy())
                acc2 = accuracy_score(labels.detach().cpu().numpy(), predicted2.detach().cpu().numpy())
                
                acc1Total += acc1
                acc2Total += acc2
                totalAcc += 0.5*acc1+0.5*acc2
                
                self.optimizer.step()
            if self.scheduler:
                self.scheduler.step()

            loss1Total = loss1Total/len(tLoader)
            loss2Total = loss2Total/len(tLoader)
            totalLoss = totalLoss/len(tLoader)
            acc1Total = acc1Total/len(tLoader)
            acc2Total = acc2Total/len(tLoader)
            totalAcc = totalAcc/len(tLoader)
            
            self.history["1"]["train"]["loss"].append(loss1Total)
            self.history["1"]["train"]["accuracy"].append(acc1Total)
            self.history["2"]["train"]["loss"].append(loss2Total)
            self.history["2"]["train"]["accuracy"].append(acc2Total)
            self.history["T"]["train"]["loss"].append(totalLoss)
            self.history["T"]["train"]["accuracy"].append(totalAcc)
            
            print("epoch {} --> trainLoss: {:0.3f}, trainAcc: {:0.3f}"
              .format(epoch+1, totalLoss, totalAcc), end="")
            
            if vLoader:
                with torch.no_grad():
                    self.net.eval()
                    for i, data in enumerate(vLoader, 0):
                        inputs, labels = data[0].to(self.device), data[1].to(self.device)
                        output1, output2 = self.net(inputs)

                        loss1, loss2 = self.criterion(output1, labels.long()), self.criterion(output2, labels.long())
                        loss1Total_v += loss1.item()
                        loss2Total_v += loss2.item()
                        valLoss += 0.5*loss1.item() + 0.5*loss2.item()

                        _, predicted1 = torch.max(output1, 1)
                        _, predicted2 = torch.max(output2, 1)
                        acc1 = accuracy_score(labels.detach().cpu().numpy(), predicted1.detach().cpu().numpy())
                        acc2 = accuracy_score(labels.detach().cpu().numpy(), predicted2.detach().cpu().numpy())

                        acc1Total_v += acc1
                        acc2Total_v += acc2
                        valAcc += 0.5*acc1 + 0.5*acc2

                    loss1Total_v = loss1Total_v/len(vLoader)
                    loss2Total_v = loss2Total_v/len(vLoader)
                    valLoss = valLoss/len(vLoader)
                    acc1Total_v = acc1Total_v/len(vLoader)
                    acc2Total_v = acc2Total_v/len(vLoader)
                    valAcc = valAcc/len(vLoader)

                    self.history["1"]["validation"]["loss"].append(loss1Total_v)
                    self.history["1"]["validation"]["accuracy"].append(acc1Total_v)
                    self.history["2"]["validation"]["loss"].append(loss2Total_v)
                    self.history["2"]["validation"]["accuracy"].append(acc2Total_v)
                    self.history["T"]["validation"]["loss"].append(valLoss)
                    self.history["T"]["validation"]["accuracy"].append(valAcc)
            
                print(", validLoss: {:0.3f}, validAcc: {:0.3f}"
                  .format(valLoss, valAcc))
                
                if valLoss <= best_loss:
                    # Save the model with the lowest validation loss.
                    best_loss = valLoss
                    torch.save(self.net.state_dict(), self.bestPath)
                    print("Model Saved!")
        
                if valLoss >= preValLoss:
                    eStopCounter += 1
                    if eStopCounter >= eStopThreshold:
                        print("Training Stopped!")
                        break;
                else:
                    eStopCounter = 0
                preValLoss = valLoss
        
            else:
                print("")
        
        return self.net, self.history

    def infer(self, sLoader, threshold=0.05, verbose=False):
        """
            @Inference: we compare the output confidence (entropy) at a branch with a certain threshold.
        """
        softmaxLayer = nn.Softmax(dim=1)
        acc = 0
        predicted = []
        recorder = {x: [] for x in range(self.nBranches)}
        self.net.eval()
        with torch.no_grad():
            for inputs, gTruth in sLoader:
                inputs, gTruth = inputs.to(self.device), gTruth.to(self.device)
                x = self.net.baseModel(inputs)
                for iSample in range(x.shape[0]): # a sample by sample
                    out1 = self.net.shortBranch(x[iSample:iSample+1])
                    y = softmaxLayer(out1)
                    e = entropy(y.detach().cpu().numpy().squeeze(), base=10)
                    if e <= threshold:
                        if verbose:
                            print(e)
                        _, label = torch.max(out1, 1)
                        predicted.append(label)
                        if label == gTruth[iSample].item():
                            recorder[0].append(1)
                            acc+=1
                        else:
                            recorder[0].append(0)
                        continue
                    out2 = self.net.longBranch(x[iSample:iSample+1])
                    _, label = torch.max(out2, 1)
                    predicted.append(label)
                    if label == gTruth[iSample].item():
                        acc+=1
                        recorder[1].append(1)
                    else:
                        recorder[1].append(0) 
            
            acc = acc / sum([len(recorder[x]) for x in range(self.nBranches)])

        return recorder, torch.FloatTensor(predicted), acc
    
    def forward_timeTest(self, sLoader, ratio=0.1):
        self.net.eval()
        num_samples_to_select = int(ratio * 500)
        with torch.no_grad():
            for inputs, _ in sLoader:
                inputs = inputs.to(self.device)
                selected_data = inputs[0: num_samples_to_select]
                remaining_data = inputs[num_samples_to_select: inputs.shape[0]]
                _ = self.net.short_forward(selected_data)
                _ = self.net.long_forward(remaining_data)

    def testingSummary(self, recorder, nBranches=2, overall=True):
        print('Summary')
        print("======================")
        overallAcc, acc = 0, 0
        overallCount = sum([len(recorder[x]) for x in range(nBranches)])
        for i in range(nBranches):
            countSamples = len(recorder[i])
            if countSamples != 0:
                acc = recorder[i].count(1)/len(recorder[i])
                print("Branch {}: Accuracy {:.2f}% with {:.2f}% of the samples".format(i+1, acc*100, countSamples/overallCount*100))
            else:
                print("Branch {}: Got 0% of the samples".format(i+1))
            overallAcc += acc*countSamples
        if overall:
            print("Overall Weighted Accuracy: {:.2f}%".format(overallAcc/overallCount*100))   



class blHandler():
    def __init__(self, net, criterion, optimizer, device, scheduler=None, num_epochs=50, bestPath="./models/best_0420.pth"):
        self.net = net
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.NUM_EPOCHS = num_epochs
        self.device = device
        self.bestPath = bestPath
      
        self.history = {
              "train": {"loss": [], "accuracy": []}, 
              "validation":{"loss": [], "accuracy": []},
              }
        
    def train(self, tLoader, vLoader=None):
        eStopThreshold, eStopCounter = 8, 0 
        best_loss, preValLoss = 100, 100
        
        for epoch in range(self.NUM_EPOCHS):
            totalLoss, totalAcc = 0, 0
            valLoss, valAcc = 0, 0

            self.net.train()
            for i, data in enumerate(tLoader, 0):
                inputs, labels = data[0].to(self.device), data[1].to(self.device)

                self.optimizer.zero_grad()
                output = self.net(inputs)
                loss = self.criterion(output, labels.long())
                totalLoss += loss.item()
                loss.backward()
    
                _, predicted = torch.max(output, 1)
                acc = accuracy_score(labels.detach().cpu().numpy(), predicted.detach().cpu().numpy())
                totalAcc += acc
                
                self.optimizer.step()
            if self.scheduler:
                self.scheduler.step() 
            totalLoss = totalLoss/len(tLoader)
            totalAcc = totalAcc/len(tLoader)
            
            self.history["train"]["loss"].append(totalLoss)
            self.history["train"]["accuracy"].append(totalAcc)
            
            print("epoch {} --> trainLoss: {:0.3f}, trainAcc: {:0.3f}"
              .format(epoch+1, totalLoss, totalAcc), end="")
            
            if vLoader:
                with torch.no_grad():
                    self.net.eval()
                    for i, data in enumerate(vLoader, 0):
                        inputs, labels = data[0].to(self.device), data[1].to(self.device)
                        output = self.net(inputs)
                        loss = self.criterion(output, labels.long())

                        valLoss += loss.item()

                        _, predicted = torch.max(output, 1)
                        acc = accuracy_score(labels.detach().cpu().numpy(), predicted.detach().cpu().numpy())

                        valAcc += acc 
         
                    valLoss = valLoss/len(vLoader)
                    valAcc = valAcc/len(vLoader)

                    self.history["validation"]["loss"].append(valLoss)
                    self.history["validation"]["accuracy"].append(valAcc)
            
                print(", validLoss: {:0.3f}, validAcc: {:0.3f}"
                  .format(valLoss, valAcc))
                
                if valLoss <= best_loss:
                    # Save the model with the lowest validation loss.
                    best_loss = valLoss
                    torch.save(self.net.state_dict(), self.bestPath)
                    print("Model Saved!")
        
                if valLoss >= preValLoss:
                    eStopCounter += 1
                    if eStopCounter >= eStopThreshold:
                        print("Training Stopped!")
                        break;
                else:
                    eStopCounter = 0
                preValLoss = valLoss
        
            else:
                print("")

        return self.net, self.history

    def infer(self, sLoader):
        """
            @Inference
        """
        
        acc, tLoss = 0, 0
        predicted = []
        with torch.no_grad():
            self.net.eval()
            for inputs, gTruth in sLoader:
                inputs, gTruth = inputs.to(self.device), gTruth.to(self.device)
                outputs = self.net(inputs)

                _, preds = torch.max(outputs, 1)
                predicted.append(preds)
                loss = self.criterion(outputs, gTruth.long())
                tLoss += loss.item()
                acc += accuracy_score(gTruth.detach().cpu().numpy(), preds.detach().cpu().numpy())

            acc = acc/len(sLoader)
            tLoss = tLoss/len(sLoader)
        return predicted, acc
    

    def late_inference(self, sLoader, threshold=0.05, verbose=False):
        """
            @Inference: we compare the output confidence (entropy) at a branch with a certain threshold.
        """
        softmaxLayer = nn.Softmax(dim=1)
        acc = 0
        predicted = []
        recorder = {x: [] for x in range(2)}
        self.net.eval()
        with torch.no_grad():
            for inputs, gTruth in sLoader:
                inputs, gTruth = inputs.to(self.device), gTruth.to(self.device)
                x = self.net(inputs)
                for iSample in range(x.shape[0]): # a sample by sample
                    out1 = self.net(x[iSample:iSample+1])
                    y = softmaxLayer(out1)
                    e = entropy(y.detach().cpu().numpy().squeeze(), base=10)
                    if e <= threshold:
                        if verbose:
                            print(e)
                        _, label = torch.max(out1, 1)
                        predicted.append(label)
                        if label == gTruth[iSample].item():
                            recorder[0].append(1)
                            acc+=1
                        else:
                            recorder[0].append(0)
                        continue
                    out2 = self.net(x[iSample:iSample+1])
                    _, label = torch.max(out2, 1)
                    predicted.append(label)
                    if label == gTruth[iSample].item():
                        acc+=1
                        recorder[1].append(1)
                    else:
                        recorder[1].append(0) 
            
            acc = acc / sum([len(recorder[x]) for x in range(2)])

        return recorder, torch.FloatTensor(predicted), acc
