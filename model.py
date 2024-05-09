import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from config import Config

'''
class AlexNetFc7(nn.Module):
    def __init__(self, freeze):
        super(AlexNetFc7, self).__init__()
        model = models.alexnet(weights='IMAGENET1K_V1')
        modules = list(model.children())
        modules.insert(-1, nn.Flatten())
        modules[-1] = modules[-1][:-1]

        self.alexnet = nn.Sequential(*modules)
        self.freeze = freeze

    def forward(self, x):
        if self.freeze:
            with torch.no_grad():
                x = self.alexnet(x)
        else:
            x = self.alexnet(x)
        return x

'''
    
class VGG16_fc(nn.Module):
    def __init__(self, freeze):
        super(VGG16_fc, self).__init__()
        model = models.vgg16(weights='IMAGENET1K_V1')
        modules = list(model.children())
        modules.insert(-1, nn.Flatten())
        modules[-1] = modules[-1][:-3]

        self.vgg16 = nn.Sequential(*modules)
        print(self.vgg16)
        self.freeze = freeze

    def forward(self, x):
        if self.freeze:
            with torch.no_grad():
                x = self.vgg16(x)
        else:
            x = self.vgg16(x)
        return x
    
class PoseClassifier(nn.Module):
    def __init__(self, cfg):
        super(PoseClassifier, self).__init__()
        self.fc7 = VGG16_fc(cfg.backbone_freeze)
        self.fc = nn.Linear(3*cfg.vgg16_fc_dim, cfg.pose_dim)

    def forward(self, x, x_crop, x_zoom):
        x = self.fc7(x)
        x_crop = self.fc7(x_crop)
        x_zoom = self.fc7(x_zoom)

        x = torch.cat((x, x_crop, x_zoom), dim=1)
        x = self.fc(x)
        return x
    
class ConditionEncoder(nn.Module):
    def __init__(self, cfg):
        super(ConditionEncoder, self).__init__()

        self.alexnet = VGG16_fc(cfg.backbone_freeze)
        self.fc1 = nn.Linear(in_features=cfg.pose_dim, out_features=cfg.fc_dim)
        self.fc2 = nn.Linear(in_features=cfg.fc_dim + 3*cfg.vgg16_fc_dim, out_features=cfg.fc_dim)

    def forward(self, pose, img, img_crop, img_zoom):
        pose = F.relu(self.fc1(pose))
        img = self.alexnet(img)
        img_crop = self.alexnet(img_crop)
        img_zoom = self.alexnet(img_zoom)

        output = torch.cat((pose, img, img_crop, img_zoom), dim=1)
        output = F.relu(self.fc2(output))

        return output

class Encoder(nn.Module):
    def __init__(self, cfg, condition_encoder):
        super(Encoder, self).__init__()
        self.fc1 = nn.Linear(in_features=cfg.sd_dim, out_features=cfg.fc_dim)
        self.fc2 = nn.Linear(in_features=cfg.fc_dim, out_features=cfg.fc_dim)

        self.condition_encoder = condition_encoder
        self.fc_mu = nn.Linear(in_features=2 * cfg.fc_dim, out_features=cfg.latent_dim)
        self.fc_logvar = nn.Linear(in_features=2 * cfg.fc_dim, out_features=cfg.latent_dim)

    def forward(self, x, pose, img, img_crop, img_zoom):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        c = self.condition_encoder(pose, img, img_crop, img_zoom)

        x = torch.cat((x, c), dim=1)
        x_mu = self.fc_mu(x)
        x_logvar = self.fc_logvar(x)

        return x_mu, x_logvar
    
class Decoder(nn.Module):
    def __init__(self, cfg, condition_encoder):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(in_features=cfg.latent_dim, out_features=cfg.fc_dim)
        self.fc2 = nn.Linear(in_features=cfg.fc_dim, out_features=cfg.fc_dim)

        self.condition_encoder = condition_encoder

        self.fc3 = nn.Linear(in_features=cfg.fc_dim, out_features=cfg.fc_dim)
        self.fc4 = nn.Linear(in_features=2 * cfg.fc_dim, out_features=cfg.fc_dim)
        self.fc5 = nn.Linear(in_features=cfg.fc_dim, out_features=cfg.fc_dim)
        self.fc6 = nn.Linear(in_features=cfg.fc_dim, out_features=cfg.sd_dim)

    def forward(self, x, pose, img, img_crop, img_zoom):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        c = self.condition_encoder(pose, img, img_crop, img_zoom)
        c = F.relu(self.fc3(c))

        x = torch.cat((x, c), dim=1)
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        x = self.fc6(x)

        return x
    
class VariationalAutoencoder(nn.Module):
    def __init__(self, cfg):
        super(VariationalAutoencoder, self).__init__()
        self.condition_encoder = ConditionEncoder(cfg)
        self.encoder = Encoder(cfg, self.condition_encoder)
        self.decoder = Decoder(cfg, self.condition_encoder)

    def forward(self, x, pose, img, img_crop, img_zoom):
        latent_mu, latent_logvar = self.encoder(x, pose, img, img_crop, img_zoom)
        latent = self.latent_sample(latent_mu, latent_logvar)
        x_recon = self.decoder(latent, pose, img, img_crop, img_zoom)
        
        return x_recon, latent_mu, latent_logvar

    def latent_sample(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = torch.empty_like(std).normal_()
            return eps.mul(std).add_(mu)
        else:
            return mu
        
if __name__ == '__main__':
    cfg = Config()
    vae = VariationalAutoencoder(cfg)
    x = torch.randn((1, cfg.sd_dim))
    pose = torch.randn((1, cfg.pose_dim))
    img1 = torch.randn((1, 3, 224, 224))
    img2 = torch.randn((1, 3, 224, 224))
    img3 = torch.randn((1, 3, 224, 224))

    x_recon, latent_mu, latent_logvar = vae(x, pose, img1, img2, img3)
    print("===VAE OUTPUT===")
    print(x_recon.shape)
    print(latent_mu.shape)
    print(latent_logvar.shape)

    print("===DECODER OUTPUT===")
    latent = torch.randn((1, cfg.latent_dim))
    x_recon = vae.decoder(latent, pose, img1, img2, img3)
    print(x_recon.shape)

    print("===Parmeter Count===")
    num_params = sum(p.numel() for p in vae.parameters() if p.requires_grad)
    print(num_params)