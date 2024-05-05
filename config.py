from loss import InfonceLoss,InfonceLossForClustering

imu_encoder_type = "spatiotemporal" #fc #cnn #res #lstm #spatiotemporal
text_encoder_type = "res" #fc #cnn #res #spatial
pose_type = "pose" #embedding 
pose_encoder_type = "spatiotemporal"  #fc #cnn #res #spatiotemporal

batch_size = 32
embedding_dim = 2048
num_epochs = 100
patience = 10

classes = 10
num_epochs_class = 100
batch_size_class = 32

parent = "c:/Users/lalas/Documents/GitHub/simu_wrist_har/"
#parent = "/home/lala/other/Repos/git/simu_wrist_har/"

loss = InfonceLossForClustering() #InfonceLossForClustering() #InfonceLoss()