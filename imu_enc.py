import torch
import torch.nn as nn
import torch.nn.functional as F

imu_encoder_type = "fc"

if imu_encoder_type == "cnn":

    print("ToDo")

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
        def __init__(self, input_dim, embedding_dim):
            super(ImuEncoder, self).__init__()
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, embedding_dim),
                nn.ReLU(),
                nn.Linear(embedding_dim, embedding_dim)
            )

        def forward(self, x):
            return self.encoder(x)
