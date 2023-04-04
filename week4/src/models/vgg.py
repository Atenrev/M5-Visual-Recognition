import torch
import torchvision

# https://open.spotify.com/track/5NSR9uH0YeDo67gMiEv13n?si=1885da5af0694dfd


class VGG19(torch.nn.Module):
    def __init__(self, norm=2, batchnorm=True, pretrained='imagenet'):
        super(VGG19, self).__init__()
        model = torchvision.models.vgg19(
            pretrained=pretrained) if not batchnorm else torchvision.models.vgg19_bn(pretrained=pretrained)
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
