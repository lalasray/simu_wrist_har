from tensorboard import notebook

log_dir = 'C:\Users\lalas\Documents\GitHub\simu_wrist_har\local_logs\multimodal_experiment\version_0\events.out.tfevents.1714786213.port-3128.22968.0'
notebook.start("--logdir " + log_dir)
