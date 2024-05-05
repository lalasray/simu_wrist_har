import torch
import torch.nn as nn
import torch.nn.functional as F
import config

classifer_type = config.classifer_type

if classifer_type == "cnn":
    
    print("ToDo") 

elif classifer_type == "res":
    
    print("ToDo") 

elif classifer_type == "lstm":
    
    print("ToDo") 

elif classifer_type == "spatiotemporl":

    print("ToDo") 

else:
    class ClassificationHead(nn.Module):
        def __init__(self, input_dim, num_classes):
            super(ClassificationHead, self).__init__()
            self.fc = nn.Linear(input_dim, num_classes)

        def forward(self, x):
            x = self.fc(x)
            return x
