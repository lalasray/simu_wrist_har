import torch.nn as nn

class MultiModalModel(nn.Module):
    def __init__(self, text_encoder, imu_encoder, video_encoder, pose_encoder):
        super(MultiModalModel, self).__init__()
        self.text_encoder = text_encoder
        self.imu_encoder = imu_encoder
        self.video_encoder = video_encoder
        self.pose_encoder = pose_encoder

    def forward(self, text_input, imu_input, video_input, pose_input):
        text_output = self.text_encoder(text_input)
        imu_output = self.imu_encoder(imu_input)
        video_output = self.video_encoder(video_input)
        pose_output = self.pose_encoder(pose_input)
        return text_output, imu_output, video_output, pose_output