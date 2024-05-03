import torch
from model import TriModalModel
from text_enc import TextEncoder
from imu_enc import ImuEncoder
from pose_enc import PoseEncoder
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from dataloader import TriDataset, get_data_files
from torch.utils.data import ConcatDataset, DataLoader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
embedding_dim = 512

model = TriModalModel(TextEncoder(embedding_dim=embedding_dim).to(device),
                      ImuEncoder(embedding_dim=embedding_dim).to(device),
                      PoseEncoder(embedding_dim=embedding_dim).to(device)).to(device)
model.load_state_dict(torch.load('best_model.pth'))
model.eval()

parent = "c:/Users/lalas/Documents/GitHub/simu_wrist_har/"
val_path = parent + '/data/how2sign/val/tensors'

dataset_val = TriDataset(get_data_files(val_path))

loader = DataLoader(dataset_val, batch_size=256, shuffle=False)

text_embeddings = []
imu_embeddings = []
pose_embeddings = []

with torch.no_grad():
    for pose, imu, text in loader:
        imu = imu.permute(0, 2, 1)
        #imu = imu.unsqueeze(3) # only for lstm
        text_embedding, imu_embedding, pose_embedding = model(text, imu, pose)
        text_embeddings.append(text_embedding)
        imu_embeddings.append(imu_embedding)
        pose_embeddings.append(pose_embedding)

text_embeddings = torch.cat(text_embeddings, dim=0)
imu_embeddings = torch.cat(imu_embeddings, dim=0)
pose_embeddings = torch.cat(pose_embeddings, dim=0)

tsne = TSNE(n_components=2, random_state=42)

text_tsne = tsne.fit_transform(text_embeddings.cpu())
imu_tsne = tsne.fit_transform(imu_embeddings.cpu())
pose_tsne = tsne.fit_transform(pose_embeddings.cpu())

plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.title('Text Modality')
plt.scatter(text_tsne[:, 0], text_tsne[:, 1], s=5)

plt.subplot(1, 3, 2)
plt.title('IMU Modality')
plt.scatter(imu_tsne[:, 0], imu_tsne[:, 1], s=5)

plt.subplot(1, 3, 3)
plt.title('Pose Modality')
plt.scatter(pose_tsne[:, 0], pose_tsne[:, 1], s=5)

plt.show()
