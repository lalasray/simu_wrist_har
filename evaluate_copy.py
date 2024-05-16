import torch
from model import TriModalModel
from text_enc import TextEncoder
from imu_enc import ImuEncoder
from pose_enc import PoseEncoder
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from dataloader_class import TriDataset, get_data_files
from torch.utils.data import DataLoader
import config
from classifier import ClassifierDecoder  
import torch.nn as nn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
embedding_dim = config.embedding_dim

model = TriModalModel(TextEncoder(embedding_dim=embedding_dim).to(device),
                      ImuEncoder(embedding_dim=embedding_dim).to(device),
                      PoseEncoder(embedding_dim=embedding_dim).to(device)).to(device)
                      
model.load_state_dict(torch.load('best_model_fc.pth'))
model.eval()

parent = config.parent
val_path = parent + 'data/openpack/u1/'
val_dataset = TriDataset(get_data_files(val_path))
loader = DataLoader(val_dataset, batch_size=config.batch_size_class, shuffle=False, num_workers=10,  drop_last=False, pin_memory=True)

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

fine_tuned_model = FineTunedModel(model.imu_encoder, classifier_decoder).to(device)
fine_tuned_model.load_state_dict(torch.load('0_no_fc_fcclassifier_decoder_.pth'))


text_embeddings = []
imu_embeddings = []
raw_embeddings = []
final_embeddings = []

with torch.no_grad():
    for imu, label_data in loader:
        imu = imu.to(device)
        label_data = label_data.to(device)
        out = model.imu_encoder(imu.float())
        text_embeddings.append(label_data)
        imu_embeddings.append(out)
        raw_embeddings.append(imu.float())
        output = fine_tuned_model(imu.float())
        final_embeddings.append(output)

text_embeddings = torch.cat(text_embeddings, dim=0)
imu_embeddings = torch.cat(imu_embeddings, dim=0)
raw_embeddings = torch.cat(raw_embeddings, dim=0)
final_embeddings = torch.cat(final_embeddings, dim=0)
new_size = (raw_embeddings.size(0), raw_embeddings.size(1) * raw_embeddings.size(2))
raw_embeddings = raw_embeddings.view(new_size)
tsne = TSNE(n_components=2, random_state=42, metric='cosine')
imu_tsne_1 = tsne.fit_transform(raw_embeddings.cpu())
imu_tsne_2 = tsne.fit_transform(imu_embeddings.cpu())
imu_tsne_3 = tsne.fit_transform(final_embeddings.cpu())
text_labels_color = text_embeddings.cpu().numpy()



plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.title('Raw IMU Modality')
plt.scatter(imu_tsne_1[:, 0], imu_tsne_1[:, 1],c=text_labels_color, cmap='tab10', s=5)


plt.subplot(1, 3, 2)
plt.title('IMU Modality Frozen')
plt.scatter(imu_tsne_2[:, 0], imu_tsne_2[:, 1],c=text_labels_color, cmap='tab10', s=5)

plt.subplot(1, 3, 3)
plt.title('IMU Modality Finetuned')
plt.scatter(imu_tsne_3[:, 0], imu_tsne_3[:, 1],c=text_labels_color, cmap='tab10', s=5)

plt.show()