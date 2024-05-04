from tensorboard import notebook

batch_size = 32
embedding_size = 1024
log_dir = './logs_embedding_dim_'+str(1024)+'_batch_size_'+str(32)
notebook.start("--logdir " + log_dir)
