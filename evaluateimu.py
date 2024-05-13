import torch
from model import TriModalModel
from text_enc import TextEncoder
from imu_enc import ImuEncoder
from pose_enc import PoseEncoder
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from dataloader import TriDataset, get_data_files
from torch.utils.data import DataLoader
import config
from classifier import ClassifierDecoder  
from dataloader_class import TriDataset, get_data_files
import torch.nn as nn
import torch.optim as optim

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
embedding_dim = config.embedding_dim

model = TriModalModel(TextEncoder(embedding_dim=embedding_dim).to(device),
                      ImuEncoder(embedding_dim=embedding_dim).to(device),
                      PoseEncoder(embedding_dim=embedding_dim).to(device)).to(device)

model.load_state_dict(torch.load('best_model.pth'))
model.eval()
imu_encoder = model.imu_encoder
parent = config.parent
#val_path = parent + '/data/how2sign/val/tensors'

#dataset_val = TriDataset(get_data_files(val_path))

#loader = DataLoader(dataset_val, batch_size=config.batch_size, shuffle=False)

#imu_embeddings = []

#with torch.no_grad():
#    for pose, imu, text in loader:
#        imu_embedding= imu_encoder(imu)
#        imu_embeddings.append(imu_embedding)
#        print(imu_embedding.shape, imu.dtype)

train_path = parent + 'data/openpack_uni/tensors' 
classifier_decoder = ClassifierDecoder(input_size=embedding_dim, num_classes=config.classes).to(device)

class FineTunedModel(nn.Module):
    def __init__(self, pretrained_model, classifier_decoder):
        super(FineTunedModel, self).__init__()
        self.pretrained_model = pretrained_model
        self.classifier_decoder = classifier_decoder
        
    def forward(self, imu_input):
        imu_output = self.pretrained_model(imu_input)
        classification_logits = self.classifier_decoder(imu_output)
        return classification_logits

    
fine_tuned_model = FineTunedModel(imu_encoder, classifier_decoder).to(device)
train_dataset = TriDataset(get_data_files(train_path))
train_loader = DataLoader(train_dataset, batch_size=config.batch_size_class, shuffle=True)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(classifier_decoder.parameters(), lr=0.001)


num_epochs = config.num_epochs_class

for epoch in range(num_epochs):
    fine_tuned_model.train()
    total_train_loss = 0.0
    for imu, label_data in train_loader:
        optimizer.zero_grad()
        aclass_pred = fine_tuned_model(imu.to(device))
        loss = criterion(aclass_pred, label_data.to(device))  
        loss.backward()
        optimizer.step()
        total_train_loss += loss.item()

    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {total_train_loss / len(train_loader)}')
    torch.save(classifier_decoder.state_dict(), f'classifier_decoder_.pth')