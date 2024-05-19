from loss import InfonceLoss,ContrastiveLoss

imu_encoder_type = "i_spatiotemporal" #fc #cnn #res #lstm #spatiotemporal #hybrid_st #i_spatiotemporal
text_encoder_type = "fc" #fc #cnn #res #spatial
pose_type = "pose" #embedding 
pose_encoder_type = "i_spatiotemporal"  #fc #cnn #res #spatiotemporal #i_spatiotemporal

batch_size = 32
embedding_dim = 256
num_epochs = 300
patience = 30

classifer_type = 'fc' #multihead #fc #i_multihead

classes = 11 #openpack11
num_epochs_class = 300
batch_size_class = 32

parent = "c:/Users/lalas/Documents/GitHub/simu_wrist_har/"
#parent = "/home/lala/other/Repos/git/simu_wrist_har/"

loss = InfonceLoss() #ContrastiveLoss() #InfonceLoss()