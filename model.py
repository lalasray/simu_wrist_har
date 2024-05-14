import torch.nn as nn
import torch.nn.functional as F

class TriModalModel(nn.Module):
    def __init__(self, text_encoder, imu_encoder, pose_encoder):
        super(TriModalModel, self).__init__()
        self.text_encoder = text_encoder
        self.imu_encoder = imu_encoder
        self.pose_encoder = pose_encoder

    def forward(self, text_input, imu_input, pose_input):
        text_output = self.text_encoder(text_input)
        imu_output = self.imu_encoder(imu_input)
        pose_output = self.pose_encoder(pose_input)
        #text_output = F.normalize(text_output, p=2, dim=1)
        #imu_output = F.normalize(imu_output, p=2, dim=1)
        #pose_output = F.normalize(pose_output, p=2, dim=1)
        
        return text_output, imu_output, pose_output

class QuadModalModel(nn.Module):
    def __init__(self, text_encoder, imu_encoder, pose_encoder, video_encoder):
        super(QuadModalModel, self).__init__()
        self.text_encoder = text_encoder
        self.imu_encoder = imu_encoder
        self.pose_encoder = pose_encoder
        self.video_encoder = video_encoder 

    def forward(self, text_input, imu_input, video_input, pose_input):
        text_output = self.text_encoder(text_input)
        imu_output = self.imu_encoder(imu_input)
        pose_output = self.pose_encoder(pose_input)
        video_output = self.video_encoder(video_input) 

        return text_output, imu_output, pose_output, video_output 