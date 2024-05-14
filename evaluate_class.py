import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from model import TriModalModel
from text_enc import TextEncoder
from imu_enc import ImuEncoder
from pose_enc import PoseEncoder
from classifier import ClassifierDecoder  
from dataloader_class import TriDataset, get_data_files
import config
from sklearn.metrics import f1_score, accuracy_score

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
embedding_dim = config.embedding_dim
encoder = TriModalModel(TextEncoder(embedding_dim=embedding_dim).to(device),
                        ImuEncoder(embedding_dim=embedding_dim).to(device),
                        PoseEncoder(embedding_dim=embedding_dim).to(device)).to(device)
encoder.load_state_dict(torch.load('best_model.pth'))
encoder.eval()
imu_encoder = encoder.imu_encoder

num_classes = config.classes
classifier_decoder = ClassifierDecoder(input_size=embedding_dim, num_classes=num_classes).to(device)

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

parent = config.parent
val_path = parent + 'data/openpack_uni/val/tensors'

val_dataset = TriDataset(get_data_files(val_path))
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

criterion = nn.MultiLabelSoftMarginLoss().to(device)

# Evaluation
fine_tuned_model.eval()
total_val_loss = 0.0
predictions = []
targets = []

with torch.no_grad():
    for imu, label_data in val_loader:
        imu = imu.to(device)
        imu_np = imu.detach().cpu().numpy()
        imu_double = torch.tensor(imu_np, dtype=torch.float32).to(device)  
        aclass_pred = fine_tuned_model(imu_double)
        loss = criterion(aclass_pred, label_data.float().to(device))
        total_val_loss += loss.item()
        
        # Convert logits to predictions by thresholding at 0.5
        predicted = (aclass_pred > 0).float()  # Assuming logits, no thresholding
        predictions.extend(predicted.cpu().numpy())
        targets.extend(label_data.cpu().numpy())
        #print(label_data, label_data.argmax(dim=1))
        break

# Convert one-hot encoded targets and predictions to binary labels
targets_binary = torch.tensor(targets).argmax(dim=1)
predictions_binary = torch.tensor(predictions).argmax(dim=1)

# Calculate accuracy
accuracy = accuracy_score(targets_binary, predictions_binary)

# Calculate F1 score
f1 = f1_score(targets_binary, predictions_binary, average='micro')

print(f'Validation Loss: {total_val_loss / len(val_loader)}')
print(f'Validation Accuracy: {accuracy * 100}%')
print(f'F1 Score: {f1}')
