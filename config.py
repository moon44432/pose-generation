class Config:
    def __init__(self):
        
        self.alexnet_fc7_dim = 4096
        self.pose_dim = 30

        self.fc_dim = 512
        self.latent_dim = 30
        self.sd_dim = 34

        self.num_epochs = 10
        self.validation_term = 1
        self.variational_beta = 0.99
        self.learning_rate = 2e-4
        self.adam_beta1 = 0.5
        self.adam_beta2 = 0.999
        self.CLIP = 1
        self.batch_size = 32        