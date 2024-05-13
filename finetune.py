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


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

embedding_dim = config.embedding_dim
encoder = TriModalModel(TextEncoder(embedding_dim=embedding_dim).to(device),
                        ImuEncoder(embedding_dim=embedding_dim).to(device),
                        PoseEncoder(embedding_dim=embedding_dim).to(device)).to(device)
encoder.load_state_dict(torch.load('best_model.pth'))
encoder.eval()

# Define classifier decoder
num_classes = 10  
classifier_decoder = ClassifierDecoder(input_size=embedding_dim, num_classes=num_classes).to(device)

parent = config.parent
train_path = parent + '/data/openpack/train/tensors' 
val_path = parent + '/data/openpack/val/tensors'

#train_dataset = TriDataset(get_data_files(train_path))
#val_dataset = TriDataset(get_data_files(val_path))
train_dataset = TriDataset(None)  # Pass None for random data
val_dataset = TriDataset(None)  # Pass None for random data

train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(classifier_decoder.parameters(), lr=0.001)


num_epochs = 10 
for epoch in range(num_epochs):
    classifier_decoder.train()
    running_loss = 0.0
    for _ in range(len(train_loader)):
        # Generate random data
        pose = torch.rand(config.batch_size, config.pose_input_dim).to(device)
        imu = torch.rand(config.batch_size, config.imu_input_dim).to(device)
        text = torch.randint(0, config.vocab_size, (config.batch_size, config.max_seq_length)).to(device)
        labels = torch.randint(0, num_classes, (config.batch_size,)).to(device)
        
        # Forward pass
        with torch.no_grad():
            text_embedding, imu_embedding, pose_embedding = encoder(text, imu, pose)
        outputs = classifier_decoder(imu_embedding)
        
        # Calculate loss
        loss = criterion(outputs, labels)
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    
    # Print statistics
    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(train_loader)}')

    # Save model after each epoch
    torch.save(classifier_decoder.state_dict(), f'classifier_decoder_epoch_{epoch + 1}.pth')

# Evaluation
classifier_decoder.eval()
correct = 0
total = 0
with torch.no_grad():
    for _ in range(len(val_loader)):
        # Generate random data for validation
        pose = torch.rand(config.batch_size, config.pose_input_dim).to(device)
        imu = torch.rand(config.batch_size, config.imu_input_dim).to(device)
        text = torch.randint(0, config.vocab_size, (config.batch_size, config.max_seq_length)).to(device)
        labels = torch.randint(0, num_classes, (config.batch_size,)).to(device)
        
        # Forward pass
        _, imu_embedding, _ = encoder(text, imu, pose)
        outputs = classifier_decoder(imu_embedding)
        
        # Calculate accuracy
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy on validation set: {100 * correct / total}%')
