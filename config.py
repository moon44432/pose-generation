import os

class Config:
    def __init__(self):
        
        self.alexnet_fc7_dim = 4096
        self.vgg16_fc_dim = 4096
        self.pose_dim = 30

        self.fc_dim = 512
        self.latent_dim = 30
        self.sd_dim = 34
        self.class_weight = [0.1882, 1.01616, 0.39219, 0.41017, 1.03255, 0.24806, 0.66632, 0.45527, 2.21929, 3.7488, 1.38706, 0.87419, 2.96168, 2.4992, 0.35841, 3.43898, 20.29837, 3.80015, 3.07097, 9.90754, 0.55855, 2.18434, 9.35094, 25.21919, 9.24704, 6.45142, 21.33932, 7.70586, 41.61167, 52.01458]

        self.target_point_method = 'center' # ['mean', 'center', keypoint_index]

        self.num_epochs = 100
        self.validation_term = 1
        self.variational_beta = 1
        self.learning_rate = 2e-4
        self.adam_beta1 = 0.5
        self.adam_beta2 = 0.999
        self.weight_decay = 2e-5
        self.CLIP = 1
        self.batch_size =  256
        
        self.pck_threshold = 0.2

        self.backbone_freeze = False

        self.checkpoint_dir = 'checkpoints/experiment20'
        if not os.path.exists('checkpoints'):
            os.mkdir('checkpoints')
        if not os.path.exists(self.checkpoint_dir):
            os.mkdir(self.checkpoint_dir)