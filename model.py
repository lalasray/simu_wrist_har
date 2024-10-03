import torch.nn as nn
import torch.nn.functional as F
import torch

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

    def forward(self, text_input, imu_inputL, imu_inputR, pose_input):
        text_output = self.text_encoder(text_input)
        imu_outputL = self.imu_encoderL(imu_inputL)
        pose_output = self.pose_encoder(pose_input)
        imu_outputR = self.imu_encoderR(imu_inputR) 

        return text_output, imu_outputL,imu_outputR, pose_output 
    
class TriModalDecModel(nn.Module):
    def __init__(self, text_encoder, imu_encoder, pose_encoder, imu_decoder_i, imu_decoder_p):
        super(TriModalDecModel, self).__init__()
        self.text_encoder = text_encoder
        self.imu_encoder = imu_encoder
        self.pose_encoder = pose_encoder
        self.imu_decoder_i = imu_decoder_i
        self.imu_decoder_p = imu_decoder_p

    def forward(self, text_input, imu_input, pose_input):
        text_output = self.text_encoder(text_input)
        imu_output = self.imu_encoder(imu_input)
        pose_output = self.pose_encoder(pose_input)
        imu_output_i = self.imu_decoder_i(imu_output)
        imu_output_p = self.imu_decoder_p(pose_output)
        #text_output = F.normalize(text_output, p=2, dim=1)
        #imu_output = F.normalize(imu_output, p=2, dim=1)
        #pose_output = F.normalize(pose_output, p=2, dim=1)
        
        return text_output, imu_output, pose_output, imu_output_i, imu_output_p

class QuadModalDecModel(nn.Module):
    def __init__(self, text_encoder, imu_encoderL, imu_encoderR, pose_encoder, imu_decoder_i, imu_decoder_p):
        super(QuadModalDecModel, self).__init__()
        self.text_encoder = text_encoder
        self.imu_encoderL = imu_encoderL
        self.pose_encoder = pose_encoder
        self.imu_encoderR = imu_encoderR 
        self.imu_decoder_i = imu_decoder_i
        self.imu_decoder_p = imu_decoder_p
        self.linear_layer = nn.Linear(512, 256)

    def forward(self, text_input, imu_inputL, imu_inputR, pose_input):
        text_output = self.text_encoder(text_input)
        imu_outputL = self.imu_encoderL(imu_inputL)
        pose_output = self.pose_encoder(pose_input)
        imu_outputR = self.imu_encoderR(imu_inputR) 
        imu_output = torch.cat((imu_outputL, imu_outputR), dim=1)
        imu_output = self.linear_layer(imu_output)
        imu_output_i = self.imu_decoder_i(imu_output)
        imu_output_p = self.imu_decoder_p(pose_output)
        #text_output = F.normalize(text_output, p=2, dim=1)
        #imu_output = F.normalize(imu_output, p=2, dim=1)
        #pose_output = F.normalize(pose_output, p=2, dim=1)

        return text_output, imu_outputL,imu_outputR, pose_output, imu_output_i, imu_output_p
