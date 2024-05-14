from loss import InfonceLoss,ContrastiveLoss

imu_encoder_type = "hybrid_st" #fc #cnn #res #lstm #spatiotemporal #hybrid_st
text_encoder_type = "res" #fc #cnn #res #spatial
pose_type = "pose" #embedding 
pose_encoder_type = "spatiotemporal"  #fc #cnn #res #spatiotemporal

batch_size = 16
embedding_dim = 256
num_epochs = 300
patience = 10

classifer_type = "multihead" #fc

classes = 11 #openpack11
num_epochs_class = 300
batch_size_class = 64

#parent = "c:/Users/lalas/Documents/GitHub/simu_wrist_har/"
parent = "/home/lala/other/Repos/git/simu_wrist_har/"

loss = InfonceLoss() #ContrastiveLoss() #InfonceLoss()