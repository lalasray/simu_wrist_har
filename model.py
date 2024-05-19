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
    def __init__(self, text_encoder, imu_encoderL, imu_encoderR, pose_encoder):
        super(QuadModalModel, self).__init__()
        self.text_encoder = text_encoder
        self.imu_encoderL = imu_encoderL
        self.pose_encoder = pose_encoder
        self.imu_encoderR = imu_encoderR 

    def forward(self, text_input, imu_input, pose_input):
        text_output = self.text_encoder(text_input)
        imu_outputL = self.imu_encoderL(imu_input[:,:,0:6])
        pose_output = self.pose_encoder(pose_input)
        imu_outputR = self.imu_encoderR(imu_input[:,:,7:12]) 

        return text_output, imu_outputL,imu_outputR, pose_output 