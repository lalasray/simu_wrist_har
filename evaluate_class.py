import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from model import TriModalModel
from text_enc import TextEncoder
from imu_enc import ImuEncoder
from pose_enc import PoseEncoder
from classifier import ClassifierDecoder  
from dataloader_class import TriDataset, get_data_files
import config
from sklearn.metrics import f1_score, confusion_matrix
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
embedding_dim = config.embedding_dim
encoder = TriModalModel(TextEncoder(embedding_dim=embedding_dim).to(device),
                        ImuEncoder(embedding_dim=embedding_dim).to(device),
                        PoseEncoder(embedding_dim=embedding_dim).to(device))

imu_encoder = encoder.imu_encoder

classifier_decoder = ClassifierDecoder(input_size=embedding_dim, num_classes=11).to(device)

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
fine_tuned_model.load_state_dict(torch.load('0_no_fc_fcclassifier_decoder_.pth'))

parent = config.parent
val_path = parent + 'data/openpack/u1/'
val_dataset = TriDataset(get_data_files(val_path))
val_loader = DataLoader(val_dataset, batch_size=config.batch_size_class, shuffle=False, num_workers=10,  drop_last=False, pin_memory=True)

criterion = nn.CrossEntropyLoss().to(device)  
optimizer = optim.Adam(classifier_decoder.parameters(), lr=0.001)

num_epochs = config.num_epochs_class

fine_tuned_model.eval()
all_predictions = []
all_labels = []

with torch.no_grad():
    for imu, label_data in val_loader:
        imu = imu.to(device)
        imu_np = imu.detach().cpu().numpy()
        imu_double = torch.tensor(imu_np, dtype=torch.float32).to(device)
        
        output = fine_tuned_model(imu_double)
        _, predicted = torch.max(output, 1)
        
        all_predictions.extend(predicted.cpu().numpy())
        all_labels.extend(label_data.flatten().cpu().numpy())


f1 = f1_score(all_labels, all_predictions, average='weighted')
conf_matrix = confusion_matrix(all_labels, all_predictions)
conf_matrix_trimmed = conf_matrix[1:, 1:]
f1_trimmed = f1_score(all_labels, all_predictions, average='weighted', labels=np.unique(all_predictions))
print(f1_trimmed)