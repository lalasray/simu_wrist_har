import torch
import torch.nn as nn
import torch.nn.functional as F

imu_encoder_type = "cnn"

if imu_encoder_type == "cnn":

    class ImuEncoder(nn.Module):
        def __init__(self, input_dim=(3, 60), embedding_dim=512):
            super(ImuEncoder, self).__init__()
            self.conv1 = nn.Conv1d(in_channels=input_dim[0], out_channels=32, kernel_size=3, padding=1)
            self.conv2 = nn.Conv1d(in_channels=32, out_channels=128, kernel_size=3, padding=1)
            self.conv3 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
            self.pool = nn.MaxPool1d(kernel_size=3)
            self.fc1 = nn.Linear(256 * (input_dim[1] // 27), embedding_dim)
            self.fc2 = nn.Linear(embedding_dim*4, embedding_dim*2)
            self.fc3 = nn.Linear(embedding_dim*2, embedding_dim)


        def forward(self, x):
            slices = [x[:,0:3,:],x[:,3:6,:],x[:,6:9,:],x[:,9:12,:]]
            outputs = []
    
            for slice in slices:
                out = self.conv1(slice)
                out = torch.relu(out)
                out = self.pool(out)
                out = self.conv2(out)
                out = torch.relu(out)
                out = self.pool(out)
                out = self.conv3(out)
                out = torch.relu(out)
                out = self.pool(out)
                out = torch.flatten(out, start_dim=1)
                out = self.fc1(out)
                outputs.append(out)
            x = torch.cat(outputs, dim=1)
            x = self.fc2(x)
            x = torch.relu(x)
            x = self.fc3(x)
                        
            return x

elif imu_encoder_type == "lstm":

    class ImuEncoder(nn.Module):
        def __init__(self, in_ch = 12, embedding_dim = 512):
            super(ImuEncoder, self).__init__()
            num_conv_layers = 4
            num_conv_filter = 64
            ks = 5
            blocks = []
            for i in range(num_conv_layers):
                in_ch_ = in_ch if i == 0 else 64
                blocks.append(
                    nn.Sequential(
                        nn.Conv2d(in_ch_, 64, kernel_size=(5, 1), padding=(2, 0)),
                        nn.BatchNorm2d(64),
                        nn.ReLU(),
                    )
                )
            self.conv_blocks = nn.ModuleList(blocks)

            hidden_units = 128
            self.lstm6 = nn.LSTM(num_conv_filter, hidden_units, batch_first=True, bidirectional=True)
            self.lstm7 = nn.LSTM(hidden_units * 2, hidden_units, batch_first=True, bidirectional=True)
            self.dropout6 = nn.Dropout(p=0.3)
            self.dropout7 = nn.Dropout(p=0.3)

            self.out8 = nn.Linear(hidden_units * 2, embedding_dim)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            
            for block in self.conv_blocks:
                x = block(x)

            x = x.squeeze(3).transpose(1, 2)

            x, _ = self.lstm6(x)
            x = self.dropout6(x)
            x, _ = self.lstm7(x)
            x = self.dropout7(x)

            x = torch.mean(x, dim=1) 

            x = self.out8(x)

            return x

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
        

batch_size = 16
input_tensor = torch.randn(batch_size, 12, 60)
model = ImuEncoder(embedding_dim = 1024)
#input_tensor = input_tensor.unsqueeze(3) # only for lstm
print("input shape:", input_tensor.shape)
output = model(input_tensor)
print("Output shape:", output.shape)