import torch.nn as nn
import torch
import config

pose_type = config.pose_type
pose_encoder_type = config.pose_encoder_type

if pose_type == "pose":

    if pose_encoder_type == "i_spatiotemporal":

        class PoseEncoder(nn.Module):
            def __init__(self, input_dim=(30, 156), embedding_dim=1024, num_heads=8):
                super(PoseEncoder, self).__init__()
                self.embedding_dim = embedding_dim
                self.upsample = nn.Linear(input_dim[1], embedding_dim)
                self.pos_encoding = PositionalEncoding(embedding_dim, input_dim[1])
                self.self_attention = nn.MultiheadAttention(embedding_dim, num_heads, dropout=0.1)
                self.layer_norm = nn.LayerNorm(embedding_dim)
                self.fc1 = nn.Linear(embedding_dim*30, embedding_dim*10)
                self.fc2 = nn.Linear(embedding_dim*10, embedding_dim*5)
                        
            def forward(self, x):
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

    elif pose_encoder_type == "spatiotemporal":

        class PoseEncoder(nn.Module):
            def __init__(self, input_dim=(30, 156), embedding_dim=1024, num_heads=8):
                super(PoseEncoder, self).__init__()
                self.embedding_dim = embedding_dim
                self.upsample = nn.Linear(input_dim[1], embedding_dim)
                self.pos_encoding = PositionalEncoding(embedding_dim, input_dim[1])
                self.self_attention = nn.MultiheadAttention(embedding_dim, num_heads, dropout=0.1)
                self.layer_norm = nn.LayerNorm(embedding_dim)
                        
            def forward(self, x):
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



    elif pose_encoder_type == "cnn":

        class PoseEncoder(nn.Module):
            def __init__(self, input_dim=(30,156), embedding_dim=512):
                super(PoseEncoder, self).__init__()
                self.conv1_1 = nn.Conv1d(in_channels=3, out_channels=32, kernel_size=3, padding='same')
                self.conv1_2 = nn.Conv1d(in_channels=63, out_channels=32, kernel_size=3, padding='same')
                self.conv1_3 = nn.Conv1d(in_channels=45, out_channels=32, kernel_size=3, padding='same')
                self.conv2 = nn.Conv1d(in_channels=32, out_channels=128, kernel_size=3, padding='same')
                self.conv3 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, padding='same')
                self.fc1 = nn.Linear(256 , embedding_dim) 
                self.fc2 = nn.Linear(embedding_dim*4, embedding_dim*2)
                self.fc3 = nn.Linear(embedding_dim*2, embedding_dim)
                
                self.dropout = nn.Dropout(p=0.3)  

            def forward(self, x):
                x = x.view(-1, x.size(2), x.size(1))
                slices = [x[:,0:3,:],x[:,3:66,:],x[:,66:111,:],x[:,111:156,:]]
                outputs = []

                for x in slices:
                    
                    if x.shape[1]== 3:
                        x = self.conv1_1(x)
                    elif x.shape[1]== 63:
                        x = self.conv1_2(x)
                    else:
                        x = self.conv1_3(x)
                    x = nn.LeakyReLU(x)
                    x = self.dropout(x)  
                    x = self.conv2(x)
                    x = nn.LeakyReLU(x)
                    x = self.dropout(x)  
                    x = self.conv3(x)
                    x = nn.LeakyReLU(x)
                    x = self.dropout(x)  
                    x = torch.flatten(x, start_dim=1)
                    x = self.fc1(x)
                    outputs.append(x)
            
                x = torch.cat(outputs, dim=1)
                x = self.fc2(x)
                x = nn.LeakyReLU(x)
                x = self.dropout(x) 
                x = self.fc3(x)
                        
                return x
            
    elif pose_encoder_type == "res":

        class PoseEncoder(nn.Module):
            def __init__(self, input_dim=(30,156), embedding_dim=512):
                super(PoseEncoder, self).__init__()
                self.conv1_1 = nn.Conv1d(in_channels=3, out_channels=9, kernel_size=3, padding='same')
                self.conv1_2 = nn.Conv1d(in_channels=63, out_channels=63*3, kernel_size=3, padding='same')
                self.conv1_3 = nn.Conv1d(in_channels=45, out_channels=45*3, kernel_size=3, padding='same')
                self.conv2_1 = nn.Conv1d(in_channels=9, out_channels=30, kernel_size=3, padding='same')
                self.conv2_2 = nn.Conv1d(in_channels=63*3, out_channels=630, kernel_size=3, padding='same')
                self.conv2_3 = nn.Conv1d(in_channels=45*3, out_channels=450, kernel_size=3, padding='same')
                self.conv3_1 = nn.Conv1d(in_channels=30, out_channels=90, kernel_size=3, padding='same')
                self.conv3_2 = nn.Conv1d(in_channels=630, out_channels=90, kernel_size=3, padding='same')
                self.conv3_3 = nn.Conv1d(in_channels=450, out_channels=90, kernel_size=3, padding='same')
                self.fc1_1 = nn.Linear(90 , embedding_dim) 
                self.fc1_2 = nn.Linear(1890 , embedding_dim) 
                self.fc1_3 = nn.Linear(1350 , embedding_dim) 
                self.fc2 = nn.Linear(embedding_dim*4, embedding_dim*2)
                self.fc3 = nn.Linear(embedding_dim*2, embedding_dim)
                
                self.dropout = nn.Dropout(p=0.3)  

            def forward(self, x):
                x = x.view(-1, x.size(2), x.size(1))
                slices = [x[:,0:3,:], x[:,3:66,:], x[:,66:111,:], x[:,111:156,:]]
                outputs = []

                for slice in slices:
                    residual = slice.clone()
                    if slice.shape[1] == 3:
                        out = self.conv1_1(slice)
                        out = nn.LeakyReLU(out)
                        out = self.dropout(out)  
                        out = out + residual.view(out.shape[0], out.shape[1], -1)
                        residual = out
                        out = self.conv2_1(out)
                        out = nn.LeakyReLU(out)
                        out = self.dropout(out)  
                        out = out + residual.view(out.shape[0], out.shape[1], -1)
                        residual = out
                        out = self.conv3_1(out)
                        out = nn.LeakyReLU(out)
                        out = self.dropout(out)  
                        out = out + residual.view(out.shape[0], out.shape[1], -1)
                        out = torch.flatten(out, start_dim=1)
                        out = self.fc1_1(out)
                        outputs.append(out)

                    elif slice.shape[1] == 63:
                        out = self.conv1_2(slice)
                        out = nn.LeakyReLU(out)
                        out = self.dropout(out)  
                        out = out + residual.view(out.shape[0], out.shape[1], -1)
                        residual = out
                        out = self.conv2_2(out)
                        out = nn.LeakyReLU(out)
                        out = self.dropout(out)  
                        out = out + residual.view(out.shape[0], out.shape[1], -1)
                        residual = out
                        out = self.conv3_2(out)
                        out = nn.LeakyReLU(out)
                        out = self.dropout(out)  
                        out = out + residual.view(out.shape[0], out.shape[1], -1)
                        out = torch.flatten(out, start_dim=1)
                        out = self.fc1_2(out)
                        outputs.append(out)

                    else:
                        out = self.conv1_3(slice)
                        out = nn.LeakyReLU(out)
                        out = self.dropout(out)  
                        out = out + residual.view(out.shape[0], out.shape[1], -1)
                        residual = out
                        out = self.conv2_3(out)
                        out = nn.LeakyReLU(out)
                        out = self.dropout(out)  
                        out = out + residual.view(out.shape[0], out.shape[1], -1)
                        residual = out
                        out = self.conv3_3(out)
                        out = nn.LeakyReLU(out)
                        out = self.dropout(out)  
                        out = out + residual.view(out.shape[0], out.shape[1], -1)
                        out = torch.flatten(out, start_dim=1)
                        out = self.fc1_3(out)
                        outputs.append(out)

                x = torch.cat(outputs, dim=1)
                x = self.fc2(x)
                x = nn.LeakyReLU(x)
                x = self.dropout(x) 
                x = self.fc3(x)
                        
                return x

    else:

        class PoseEncoder(nn.Module):
            def __init__(self, input_dim=30*156, embedding_dim=512):
                super(PoseEncoder, self).__init__()
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
                    
elif pose_type == "embedding":

    if pose_encoder_type == "attention":

        print("ToDo")

    elif pose_encoder_type == "cnn":

        print("ToDo")
  
    else:

        print("ToDo")

if __name__ == '__main__':
    batch_size = config.batch_size
    input_tensor = torch.randn(batch_size, 30, 156)
    model = PoseEncoder(embedding_dim = config.embedding_dim)
    print("input shape:", input_tensor.shape)
    output = model(input_tensor)
    print("Output shape:", output.shape)