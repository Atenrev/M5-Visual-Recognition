import torch
import torchvision

# https://open.spotify.com/track/5NSR9uH0YeDo67gMiEv13n?si=1885da5af0694dfd


class VGG19(torch.nn.Module):
    def __init__(self, norm=None, batchnorm=True, pretrained='imagenet'):
        super(VGG19, self).__init__()
        if pretrained and batchnorm: pretrained = torchvision.models.VGG19_BN_Weights.IMAGENET1K_V1
        elif pretrained: pretrained = torchvision.models.VGG19_Weights.IMAGENET1K_V1
        else: pretrained = None

        model = torchvision.models.vgg19(
            weights=pretrained) if not batchnorm else torchvision.models.vgg19_bn(weights=pretrained)
        self.model = torch.nn.Sequential(*(list(model.children())[:-1]))
        self.norm = norm

    def infer(self, image):
        # Single Numpy Array inference
        with torch.no_grad():
            return self(torch.from_numpy(image.transpose(2, 0, 1)).float().unsqueeze(0)).cpu().squeeze().view(-1).numpy()

    def __str__(self):
        return str(self.model)

    def forward(self, batch):
        h = torch.nn.functional.adaptive_max_pool2d(self.model(batch), (1, 1))

        if self.norm is not None:
            h = torch.nn.functional.normalize(h, p=self.norm, dim=1)

        return h
