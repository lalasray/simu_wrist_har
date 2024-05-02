import torch
import torch.nn as nn
import torch.nn.functional as F

imu_encoder_type = "cnn"

if imu_encoder_type == "cnn":

    class ImuEncoder(nn.Module):
        def __init__(self, input_dim=(12, 60), embedding_dim=512):
            super(ImuEncoder, self).__init__()
            self.conv1 = nn.Conv1d(in_channels=input_dim[0], out_channels=32, kernel_size=3, padding=1)
            self.conv2 = nn.Conv1d(in_channels=32, out_channels=128, kernel_size=3, padding=1)
            self.conv3 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
            self.pool = nn.MaxPool1d(kernel_size=3)
            self.fc1 = nn.Linear(256 * (input_dim[1] // 27), embedding_dim)  # Calculate the output size dynamically
            self.fc2 = nn.Linear(embedding_dim, embedding_dim)

        def forward(self, x):
            x = self.conv1(x)
            x = torch.relu(x)
            x = self.pool(x)
            x = self.conv2(x)
            x = torch.relu(x)
            x = self.pool(x)
            x = self.conv3(x)
            x = torch.relu(x)
            x = self.pool(x)
            x = torch.flatten(x, start_dim=1)
            x = self.fc1(x)
            x = torch.relu(x)
            x = self.fc2(x)
            
            return x

elif imu_encoder_type == "lstm":

    class HARModel(nn.Module):
    
        def __init__(self, n_hidden=128, n_layers=1, n_filters=64, 
                    n_classes=18, filter_size=5, drop_prob=0.5):
            super(HARModel, self).__init__()
            self.drop_prob = drop_prob
            self.n_layers = n_layers
            self.n_hidden = n_hidden
            self.n_filters = n_filters
            self.n_classes = n_classes
            self.filter_size = filter_size
                
            self.conv1 = nn.Conv1d(12, n_filters, filter_size)
            self.conv2 = nn.Conv1d(n_filters, n_filters, filter_size)
            self.conv3 = nn.Conv1d(n_filters, n_filters, filter_size)
            self.conv4 = nn.Conv1d(n_filters, n_filters, filter_size)
            
            self.lstm1  = nn.LSTM(n_filters, n_hidden, n_layers)
            self.lstm2  = nn.LSTM(n_hidden, n_hidden, n_layers)
            
            self.fc = nn.Linear(n_hidden, n_classes)

            self.dropout = nn.Dropout(drop_prob)
        
        def forward(self, x, hidden, batch_size):
            
            x = x.view(-1, 12, 60)
            x = F.relu(self.conv1(x))
            x = F.relu(self.conv2(x))
            x = F.relu(self.conv3(x))
            x = F.relu(self.conv4(x))
            
            x = x.view(8, -1, self.n_filters)
            x, hidden = self.lstm1(x, hidden)
            x, hidden = self.lstm2(x, hidden)
            
            x = x.contiguous().view(-1, self.n_hidden)
            x = self.dropout(x)
            x = self.fc(x)
            
            out = x.view(batch_size, -1, self.n_classes)[:,-1,:]
            
            return out, hidden
        
        def init_hidden(self, batch_size):
            ''' Initializes hidden state '''
            # Create two new tensors with sizes n_layers x batch_size x n_hidden,
            # initialized to zero, for hidden state and cell state of LSTM
            weight = next(self.parameters()).data
            
            if (torch.cuda.is_available()):
                hidden = (weight.new(self.n_layers, batch_size, self.n_hidden).zero_().cuda(),
                    weight.new(self.n_layers, batch_size, self.n_hidden).zero_().cuda())
            else:
                hidden = (weight.new(self.n_layers, batch_size, self.n_hidden).zero_(),
                        weight.new(self.n_layers, batch_size, self.n_hidden).zero_())
            
            return hidden

else:       

    class ImuEncoder(nn.Module):
        def __init__(self, input_dim= 60*12, embedding_dim = 512):
            super(ImuEncoder, self).__init__()
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, embedding_dim*2),
                nn.ReLU(),
                nn.Linear(embedding_dim*2, embedding_dim)
            )

        def forward(self, x):
            batch_size = x.size(0)
            return self.encoder(x.view(batch_size, -1))