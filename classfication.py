import torch
import torch.nn as nn
from model import TriModalModel
from text_enc import TextEncoder
from imu_enc import ImuEncoder
from pose_enc import PoseEncoder
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from dataloader import TriDataset, get_data_files
from torch.utils.data import ConcatDataset, DataLoader
import config

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

embedding_dim = config.embedding_dim
num_classes = config.classes 

parent = config.parent
val_path = parent + '/data/how2sign/val/tensors'

dataset_val = TriDataset_class(get_data_files(val_path)) #dataloader with classes

loader = DataLoader(dataset_val, batch_size=config.batch_size_class, shuffle=False)

class ClassificationHead(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(ClassificationHead, self).__init__()
        self.fc = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        x = self.fc(x)
        return x

model = TriModalModel(TextEncoder(embedding_dim=embedding_dim).to(device),
                      ImuEncoder(embedding_dim=embedding_dim).to(device),
                      PoseEncoder(embedding_dim=embedding_dim).to(device)).to(device)
model.load_state_dict(torch.load('best_model.pth'))
model.eval()


classification_head = ClassificationHead(input_dim=embedding_dim * 3, num_classes=num_classes).to(device)


criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(classification_head.parameters(), lr=0.001) 

num_epochs = config.num_epochs_class 

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    
    for pose, imu, text, labels in loader: 
        pose, imu, text, labels = pose.to(device), imu.to(device), text.to(device), labels.to(device)
        

        text_embedding, imu_embedding, pose_embedding = model(text, imu, pose)
        concatenated_embedding = torch.cat((text_embedding, imu_embedding, pose_embedding), dim=1)
        classification_output = classification_head(concatenated_embedding)
        
        loss = criterion(classification_output, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * text.size(0)
    
    epoch_loss = running_loss / len(dataset_val)
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}')

print('Training finished!')
