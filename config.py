import os

class Config:
    def __init__(self):
        
        self.alexnet_fc7_dim = 4096
        self.pose_dim = 30

        self.fc_dim = 512
        self.latent_dim = 30
        self.sd_dim = 34

        self.num_epochs = 100
        self.validation_term = 1
        self.variational_beta = 0.99
        self.learning_rate = 2e-4
        self.adam_beta1 = 0.5
        self.adam_beta2 = 0.999
        self.CLIP = 1
        self.batch_size =  256
        
        self.pck_threshold = 0.2

        self.backbone_freeze = True

        self.checkpoint_dir = 'checkpoints/experiment4'
        if not os.path.exists('checkpoints'):
            os.mkdir('checkpoints')
        if not os.path.exists(self.checkpoint_dir):
            os.mkdir(self.checkpoint_dir)