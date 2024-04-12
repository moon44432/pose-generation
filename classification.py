import torch
import torch.nn as nn
import torchvision.models

class PoseClassificationNetwork(nn.Module):
    def __init__(self, num_classes):
        super(PoseClassificationNetwork, self).__init__()

        self.backbone = self.build_backbone()
        self.classifier = nn.Linear(4096 * 3, num_classes)

    def forward(self, image1, image2, image3):

        """
        feature = self.backbone(torch.concat((image1, image2, image3), dim=0))
        feature = feature.view(3, feature.size(0) // 3, -1)
        feature = torch.concat((feature[0], feature[1], feature[2]), dim=1)
        """

        feature1 = self.backbone(image1)
        feature2 = self.backbone(image2)
        feature3 = self.backbone(image3)
        feature = torch.concat((feature1, feature2, feature3), dim=1)
        
        output = self.classifier(feature)
        return output
    
    def build_backbone(self):
        
        model = torchvision.models.alexnet(weights='IMAGENET1K_V1')
        modules = list(model.children())
        
        # add Flatten before fc layer
        modules.insert(-1, nn.Flatten())
        
        # remove fc8
        modules[-1] = modules[-1][:-1]

        model = nn.Sequential(*modules)
        return model

if __name__ == '__main__':
    
    model = PoseClassificationNetwork(num_classes=30)
    model.eval()
    x = torch.randn((1, 3, 224, 224))
    y = torch.randn((1, 3, 224, 224))
    z = torch.randn((1, 3, 224, 224))
    output = model(x, y, z)
    print(output.shape)