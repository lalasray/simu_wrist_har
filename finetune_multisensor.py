import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from model import TriModalModel,QuadModalModel
from text_enc import TextEncoder
from imu_enc import ImuEncoder
from pose_enc import PoseEncoder
from classifier import ClassifierDecoder  
from dataloader_class import TriDataset, get_data_files
import config
from sklearn.metrics import f1_score
from torch.optim.lr_scheduler import ReduceLROnPlateau


for it in range(1):
        
    def calculate_f1_score(y_true, y_pred):
        return f1_score(y_true, y_pred, average='macro')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    embedding_dim = config.embedding_dim
    encoder = QuadModalModel(TextEncoder(embedding_dim=embedding_dim).to(device),
                            ImuEncoder(embedding_dim=embedding_dim).to(device),
                            ImuEncoder(embedding_dim=embedding_dim).to(device),
                            PoseEncoder(embedding_dim=embedding_dim).to(device)).to(device)
    #encoder.load_state_dict(torch.load('best_model_fc.pth'))
    imu_encoder = encoder.imu_encoderL

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
    print(fine_tuned_model)
    parent = config.parent
    #train_path = parent + 'data/openpack/train' 
    train_path = parent + 'data/openpack_uni/tensors'
    val_path = parent + 'data/openpack/u9'
    train_dataset = TriDataset(get_data_files(train_path))
    val_dataset = TriDataset(get_data_files(val_path))

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size_class, shuffle=True, drop_last=False, pin_memory=True, num_workers=10)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size_class, shuffle=False, drop_last=False, pin_memory=True, num_workers=10)

    criterion = nn.CrossEntropyLoss().to(device)  
    optimizer = optim.Adam(fine_tuned_model.parameters(), lr=0.001)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', patience=(config.patience))
    num_epochs = config.num_epochs_class

    best_f1_score = 0.0
    early_stopping_counter = 0
    early_stopping_criteria = 0.01
    previous_best_f1_score = 0.0
    patience = config.patience

    for epoch in range(num_epochs):
        fine_tuned_model.train()
        total_train_loss = 0.0
        
        for imu, label_data in train_loader:
            imu = imu[:, :, 0:6].to(device)
            label_data = label_data.to(device)
            
            optimizer.zero_grad()
            aclass_pred = fine_tuned_model(imu.float())
            loss = criterion(aclass_pred, label_data.long().flatten())  
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
        
        fine_tuned_model.eval()
        with torch.no_grad():
            all_predictions = []
            all_labels = []
            
            for imu, label_data in val_loader:
                imu = imu.to(device)
                label_data = label_data.to(device)
                
                aclass_pred = fine_tuned_model(imu.float())
                _, predicted = torch.max(aclass_pred, 1)
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(label_data.cpu().numpy())
            
            val_f1_score = calculate_f1_score(all_labels, all_predictions)
            
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {total_train_loss / len(train_loader)}, Validation F1 Score: {val_f1_score}')
            scheduler.step(val_f1_score)

            if val_f1_score > previous_best_f1_score:
                improvement = val_f1_score - previous_best_f1_score
                early_stopping_criteria = max(early_stopping_criteria, improvement)
                previous_best_f1_score = val_f1_score
                early_stopping_counter = 0
                torch.save(fine_tuned_model.state_dict(), str(it)+'_no_'+config.imu_encoder_type+'_'+config.classifer_type+'classifier_decoder_.pth')
            else:
                early_stopping_counter += 1
                if early_stopping_counter >= patience:
                    print("Early stopping...")
                    break