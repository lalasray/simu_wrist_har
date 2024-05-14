import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, f1_score
from model import TriModalModel
from text_enc import TextEncoder
from imu_enc import ImuEncoder
from pose_enc import PoseEncoder
from classifier import ClassifierDecoder  
from dataloader_class import TriDataset, get_data_files
import config

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
embedding_dim = config.embedding_dim

encoder = TriModalModel(TextEncoder(embedding_dim=embedding_dim).to(device),
                        ImuEncoder(embedding_dim=embedding_dim).to(device),
                        PoseEncoder(embedding_dim=embedding_dim).to(device)).to(device)

#encoder.load_state_dict(torch.load('best_model.pth'))
imu_encoder = encoder.imu_encoder

classifier_decoder = ClassifierDecoder(input_size=embedding_dim, num_classes=1).to(device)

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
test_path = parent + 'data/openpack_uni/val/tensors'

test_dataset = TriDataset(get_data_files(test_path))
test_loader = DataLoader(test_dataset, batch_size=config.batch_size_class, shuffle=False)

criterion = nn.CrossEntropyLoss().to(device)  

# Load fine-tuned model weights
fine_tuned_model.load_state_dict(torch.load('classifier_decoder_.pth'))

fine_tuned_model.eval()
total_test_loss = 0.0
predictions = []
true_labels = []

with torch.no_grad():
    for imu, label_data in test_loader:
        imu = imu.to(device)
        imu_np = imu.detach().cpu().numpy()
        imu_double = torch.tensor(imu_np, dtype=torch.float32).to(device)
        aclass_pred = fine_tuned_model(imu_double)
        loss = criterion(aclass_pred, label_data.long().flatten().to(device))
        total_test_loss += loss.item()
        predictions.extend(torch.sigmoid(aclass_pred).cpu().numpy())
        true_labels.extend(label_data.numpy())

print(f'Test Loss: {total_test_loss / len(test_loader)}')

predictions = [1 if pred > 0.5 else 0 for pred in predictions]

# Calculate confusion matrix
conf_matrix = confusion_matrix(true_labels, predictions)
print("Confusion Matrix:")
print(conf_matrix)

# Calculate F1 score
f1 = f1_score(true_labels, predictions)
print("F1 Score:", f1)
