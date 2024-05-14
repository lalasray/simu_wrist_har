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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
embedding_dim = config.embedding_dim

model = TriModalModel(TextEncoder(embedding_dim=embedding_dim).to(device),
                      ImuEncoder(embedding_dim=embedding_dim).to(device),
                      PoseEncoder(embedding_dim=embedding_dim).to(device)).to(device)
                      
model.load_state_dict(torch.load('best_model.pth'))
model.eval()

parent = config.parent
val_path = parent + 'data/openpack_uni/test/tensors'
val_dataset = TriDataset(get_data_files(val_path))
loader = DataLoader(val_dataset, batch_size=config.batch_size_class, shuffle=False, num_workers=10,  drop_last=False, pin_memory=True)


text_embeddings = []
imu_embeddings = []

with torch.no_grad():
    for imu, label_data in loader:
        imu = imu.to(device)
        label_data = label_data.to(device)
        out = model.imu_encoder(imu.float())
        text_embeddings.append(label_data)
        imu_embeddings.append(out)

text_embeddings = torch.cat(text_embeddings, dim=0)
imu_embeddings = torch.cat(imu_embeddings, dim=0)

#all_embeddings = torch.cat( [text_embeddings,imu_embeddings,pose_embeddings])
tsne = TSNE(n_components=2, random_state=42, metric='cosine')

#text_tsne = tsne.fit_transform(text_embeddings.cpu())
imu_tsne = tsne.fit_transform(imu_embeddings.cpu())
#pose_tsne = tsne.fit_transform(pose_embeddings.cpu())
#all_tsne = tsne.fit_transform(all_embeddings.cpu())
plt.figure(figsize=(15, 5))

plt.subplot(1, 4, 1)
plt.title('Text Modality')
plt.scatter(text_tsne[:, 0], text_tsne[:, 1], s=5)

plt.subplot(1, 4, 2)
plt.title('IMU Modality')
plt.scatter(imu_tsne[:, 0], imu_tsne[:, 1], s=5)

plt.subplot(1, 4, 3)
plt.title('Pose Modality')
plt.scatter(pose_tsne[:, 0], pose_tsne[:, 1], s=5)

plt.subplot(1, 4, 4)
plt.title('All Modality')
s = len(text_embeddings)
plt.scatter(all_tsne[0: s, 0], all_tsne[0: s, 1], s=1, color='b')
plt.scatter(all_tsne[s: 2* s , 0], all_tsne[s:2*s, 1], s=1, color='g')
plt.scatter(all_tsne[2*s:, 0], all_tsne[2*s:, 1], s=1, color='r')

plt.show()