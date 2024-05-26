import torch
import torch.nn as nn
import torch.nn.functional as F
import config

imu_encoder_type = config.imu_encoder_type

if imu_encoder_type == "cnn":

    class ImuEncoder(nn.Module):
        def __init__(self, input_dim=(3, 60), embedding_dim=512):
            super(ImuEncoder, self).__init__()
            self.conv1 = nn.Conv1d(in_channels=input_dim[0], out_channels=32, kernel_size=3, padding='same')
            self.conv2 = nn.Conv1d(in_channels=32, out_channels=128, kernel_size=3, padding='same')
            self.conv3 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, padding='same')
            self.fc1 = nn.Linear(256 * (input_dim[1] // 27), embedding_dim)
            self.fc2 = nn.Linear(embedding_dim*4, embedding_dim*2)
            self.fc3 = nn.Linear(embedding_dim*2, embedding_dim)
            self.dropout = nn.Dropout(p=0.3)  

        def forward(self, x):

            x = x.view(-1, x.size(2), x.size(1))
            slices = [x[:,0:3,:],x[:,3:6,:],x[:,6:9,:],x[:,9:12,:]]
            outputs = []
        
            for slice in slices:
                out = self.conv1(slice)
                out = nn.LeakyReLU(out)
                out = self.dropout(out)  
                out = self.conv2(out)
                out = nn.LeakyReLU(out)
                out = self.dropout(out)  
                out = self.conv3(out)
                out = nn.LeakyReLU(out)
                out = self.dropout(out)  
                out = torch.flatten(out, start_dim=1)
                out = self.fc1(out)
                outputs.append(out)
            x = torch.cat(outputs, dim=1)
            x = self.fc2(x)
            x = nn.LeakyReLU(x)
            x = self.dropout(x) 
            x = self.fc3(x)
                            
            return x
        
elif imu_encoder_type == "res":

    class ImuEncoder(nn.Module):
        def __init__(self, input_dim=(3, 60), embedding_dim=512):
            super(ImuEncoder, self).__init__()
            self.conv1 = nn.Conv1d(in_channels=input_dim[0], out_channels=9, kernel_size=3, padding='same')
            self.conv2 = nn.Conv1d(in_channels=9, out_channels=30, kernel_size=3, padding='same')
            self.conv3 = nn.Conv1d(in_channels=30, out_channels=90, kernel_size=3, padding='same')
            self.fc1 = nn.Linear(input_dim[0]*input_dim[1], embedding_dim)
            self.fc2 = nn.Linear(embedding_dim*4, embedding_dim*2)
            self.fc3 = nn.Linear(embedding_dim*2, embedding_dim)
            self.dropout = nn.Dropout(p=0.3)  

        def forward(self, x):
            x = x.view(-1, x.size(2), x.size(1))
            slices = [x[:,0:3,:],x[:,3:6,:],x[:,6:9,:],x[:,9:12,:]]
            outputs = []

            for slice in slices:
                residual = slice
                out = self.conv1(slice)
                out = nn.LeakyReLU(out)
                out = self.dropout(out)
                out = out + residual.view(out.shape[0], out.shape[1], -1)
                residual = out
                out = self.conv2(out)
                out = nn.LeakyReLU(out)
                out = self.dropout(out)  
                out = out + residual.view(out.shape[0], out.shape[1], -1)
                residual = out
                out = self.conv3(out)
                out = nn.LeakyReLU(out)
                out = self.dropout(out)  
                out = out + residual.view(out.shape[0], out.shape[1], -1)
                out = torch.flatten(out, start_dim=1) 
                out = self.fc1(out)
                outputs.append(out)

            x = torch.cat(outputs, dim=1)
            x = self.fc2(x)
            x = nn.LeakyReLU(x)
            x = self.dropout(x) 
            x = self.fc3(x)
                            
            return x

elif imu_encoder_type == "f_spatiotemporal":

    class ImuEncoder(nn.Module):
        def __init__(self, input_dim=(6, 60), embedding_dim=1024, num_heads=8):
            super(ImuEncoder, self).__init__()
            self.embedding_dim = embedding_dim
            self.upsample = nn.Linear(input_dim[1], embedding_dim)
            self.pos_encoding = PositionalEncoding(embedding_dim, input_dim[1])
            self.self_attention = nn.MultiheadAttention(embedding_dim, num_heads, dropout=0.1)
            self.layer_norm = nn.LayerNorm(embedding_dim)
            self.fc1 = nn.Linear(embedding_dim * input_dim[0], embedding_dim * 6)
            self.fc2 = nn.Linear(embedding_dim * 6, embedding_dim * 3)
            self.activation = nn.LeakyReLU()

        def forward(self, x):
            # Assume x is of shape (batch_size, channels, sequence_length)
            x = x.view(x.size(0), x.size(2), x.size(1))  # (batch_size, sequence_length, channels)
            x = self.upsample(x)  # (batch_size, sequence_length, embedding_dim)
            x = self.pos_encoding(x)  # (batch_size, sequence_length, embedding_dim)
            x = x.permute(1, 0, 2)  # (sequence_length, batch_size, embedding_dim)
            attn_output, _ = self.self_attention(x, x, x)  # (sequence_length, batch_size, embedding_dim)
            attn_output = attn_output.permute(1, 0, 2)  # (batch_size, sequence_length, embedding_dim)
            attn_output = self.layer_norm(attn_output)  # (batch_size, sequence_length, embedding_dim)
            attn_output = attn_output.view(attn_output.shape[0], -1)  # (batch_size, sequence_length * embedding_dim)
            attn_output = self.fc1(attn_output)  # (batch_size, embedding_dim * 6)
            print(attn_output.shape[1]/256)
            attn_output = self.activation(attn_output)  # Apply activation
            attn_output = self.fc2(attn_output)  # (batch_size, embedding_dim * 3)
            attn_output = attn_output.view(attn_output.shape[0], -1, self.embedding_dim)  # (batch_size, ?, embedding_dim)
            context = torch.mean(attn_output, dim=1)  # (batch_size, embedding_dim)
            return context

    class PositionalEncoding(nn.Module):
        def __init__(self, d_model, max_len=1000):
            super(PositionalEncoding, self).__init__()
            self.dropout = nn.Dropout(p=0.3)
            self.embedding_dim = d_model

            position = torch.arange(0, max_len).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, d_model, 2) * (-torch.log(torch.tensor(10000.0)) / d_model))
            pe = torch.zeros(max_len, d_model)
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            self.register_buffer('pe', pe)

        def forward(self, x):
            x = x + self.pe[:x.size(1), :].unsqueeze(0)  # Add positional encoding
            x = self.dropout(x)
            return x


elif imu_encoder_type == "i_spatiotemporal":

    class ImuEncoder(nn.Module):
        def __init__(self, input_dim=(6, 60), embedding_dim=1024, num_heads=8):
            super(ImuEncoder, self).__init__()
            self.embedding_dim = embedding_dim
            self.upsample = nn.Linear(input_dim[1], embedding_dim)
            self.pos_encoding = PositionalEncoding(embedding_dim, input_dim[1])
            self.self_attention = nn.MultiheadAttention(embedding_dim, num_heads, dropout=0.1)
            self.layer_norm = nn.LayerNorm(embedding_dim)
            self.fc1 = nn.Linear(embedding_dim*12, embedding_dim*6)
            self.fc2 = nn.Linear(embedding_dim*6, embedding_dim*3)
                    
        def forward(self, x):
            x = x.view(-1, x.size(2), x.size(1))
            x = self.upsample(x)
            x = self.pos_encoding(x)
            x = x.permute(1, 0, 2)
            attn_output, _ = self.self_attention(x, x, x)
            attn_output = attn_output.permute(1, 0, 2) 
            attn_output = self.layer_norm(attn_output)
            attn_output = attn_output.view(attn_output.shape[0], -1)
            attn_output = self.fc1(attn_output)
            x = nn.LeakyReLU(x)
            attn_output = self.fc2(attn_output)
            attn_output= attn_output.reshape(attn_output.shape[0], -1, self.embedding_dim)
            context = torch.mean(attn_output, dim=1)
            return context

    class PositionalEncoding(nn.Module):
        def __init__(self, d_model, max_len=1000):
            super(PositionalEncoding, self).__init__()
            self.dropout = nn.Dropout(p=0.3)
            self.embedding_dim = d_model
                    
            position = torch.arange(0, max_len).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, d_model, 2) * (-torch.log(torch.tensor(10000.0)) / d_model))
            pe = torch.zeros(max_len, d_model)
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            self.register_buffer('pe', pe)

        def forward(self, x):
            x = x + self.pe[:x.size(1), :].unsqueeze(0) 
            x = self.dropout(x)
            return x

elif imu_encoder_type == "spatiotemporal":

    class ImuEncoder(nn.Module):
        def __init__(self, input_dim=(12, 60), embedding_dim=1024, num_heads=8):
            super(ImuEncoder, self).__init__()
            self.embedding_dim = embedding_dim
            self.upsample = nn.Linear(input_dim[1], embedding_dim)
            self.pos_encoding = PositionalEncoding(embedding_dim, input_dim[1])
            self.self_attention = nn.MultiheadAttention(embedding_dim, num_heads, dropout=0.1)
            self.layer_norm = nn.LayerNorm(embedding_dim)
                    
        def forward(self, x):
            x = x.view(-1, x.size(2), x.size(1))
            x = self.upsample(x)
            x = self.pos_encoding(x)
            x = x.permute(1, 0, 2)
            attn_output, _ = self.self_attention(x, x, x)
            attn_output = attn_output.permute(1, 0, 2) 
            attn_output = self.layer_norm(attn_output)
            context = torch.mean(attn_output, dim=1)
            return context

    class PositionalEncoding(nn.Module):
        def __init__(self, d_model, max_len=1000):
            super(PositionalEncoding, self).__init__()
            self.dropout = nn.Dropout(p=0.3)
            self.embedding_dim = d_model
                    
            position = torch.arange(0, max_len).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, d_model, 2) * (-torch.log(torch.tensor(10000.0)) / d_model))
            pe = torch.zeros(max_len, d_model)
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            self.register_buffer('pe', pe)

        def forward(self, x):
            x = x + self.pe[:x.size(1), :].unsqueeze(0) 
            x = self.dropout(x)
            return x


elif imu_encoder_type == "hybrid_st":

    class ImuEncoder(nn.Module):
        def __init__(self, input_dim=(12, 60), embedding_dim=1024, num_heads=8):
            super(ImuEncoder, self).__init__()
            self.embedding_dim = embedding_dim
            self.upsample = nn.Linear(input_dim[1], embedding_dim)
            self.pos_encoding = PositionalEncoding(embedding_dim, input_dim[1])
            self.self_attention = nn.MultiheadAttention(embedding_dim, num_heads, dropout=0.1)
            self.layer_norm = nn.LayerNorm(embedding_dim)
            self.conv1 = nn.Conv1d(input_dim[0], 16, kernel_size=3, padding='same')
            self.conv2 = nn.Conv1d(16, 32, kernel_size=3, padding='same')
            self.fc1 = nn.Linear(embedding_dim, embedding_dim // 2)
            self.fc2 = nn.Linear(embedding_dim // 2, embedding_dim)

        def forward(self, x):
            x = x.view(-1, x.size(2), x.size(1)) 
            x = self.conv1(x)
            x = F.leaky_relu(x)
            x = self.conv2(x)
            x = F.leaky_relu(x)
            print(x.shape)
            x = x.reshape(x.shape[0], 12, -1) 
            print(x.shape)
            x = self.upsample(x)
            x = self.pos_encoding(x)
            x = x.permute(1, 0, 2)  
            attn_output, _ = self.self_attention(x, x, x)
            attn_output = attn_output.permute(1, 0, 2)
            attn_output = self.layer_norm(attn_output)
            context = torch.mean(attn_output, dim=1)
            context = self.fc1(context)
            context = F.leaky_relu(context)
            context = self.fc2(context)
            
            return context

    class PositionalEncoding(nn.Module):
        def __init__(self, d_model, max_len=1000):
            super(PositionalEncoding, self).__init__()
            self.dropout = nn.Dropout(p=0.3)
            self.embedding_dim = d_model

            position = torch.arange(0, max_len).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, d_model, 2) * (-torch.log(torch.tensor(10000.0)) / d_model))
            pe = torch.zeros(max_len, d_model)
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            self.register_buffer('pe', pe)

        def forward(self, x):
            x = x + self.pe[:x.size(1), :].unsqueeze(0)
            x = self.dropout(x)
            return x

elif imu_encoder_type == "lstm":

    class ImuEncoder(nn.Module):
        def __init__(self, in_ch=12, embedding_dim=512):
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
                        nn.LeakyReLU(),
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
            
            x = x.view(-1, x.size(2), x.size(1))
            x = x.unsqueeze(3)
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
        def __init__(self, input_dim=60*12, embedding_dim=512):
            super(ImuEncoder, self).__init__()
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, embedding_dim*2),
                nn.LeakyReLU(),
                nn.Dropout(p=0.3),  
                nn.Linear(embedding_dim*2, embedding_dim),
                nn.Dropout(p=0.3) 
            )

        def forward(self, x):
            batch_size = x.size(0)
            return self.encoder(x.view(batch_size, -1))
            
if __name__ == '__main__':
    batch_size = config.batch_size
    input_tensor = torch.randn(batch_size, 60, 6)
    model = ImuEncoder(embedding_dim = config.embedding_dim)
    print("input shape:", input_tensor.shape)
    output = model(input_tensor)
    print("Output shape:", output.shape)