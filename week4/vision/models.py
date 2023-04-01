import torch
import torchvision

# https://open.spotify.com/track/5NSR9uH0YeDo67gMiEv13n?si=1885da5af0694dfd

class Resnet(torch.nn.Module):
    def __init__(self, norm = 2, resnet = '152', pretrained = 'imagenet'):
        super(Resnet, self).__init__()

        if resnet == '152': self.resnet = torchvision.models.resnet152(pretrained = pretrained)
        elif resnet == '101': self.resnet =  torchvision.models.resnet101(pretrained = pretrained)
        elif resnet == '50': self.resnet =  torchvision.models.resnet50(pretrained = pretrained)
        elif resnet == '34': self.resnet =  torchvision.models.resnet34(pretrained = pretrained)
        elif resnet == '18': self.resnet =  torchvision.models.resnet18(pretrained = pretrained)
        else: raise NotImplementedError
        
        self.resnet = torch.nn.Sequential(*(list(self.resnet.children())[:-1]))
        self.norm = norm

    def infer(self, image):
        # Single Numpy Array inference
        with torch.no_grad():

            return self(torch.from_numpy(image.transpose(2, 0, 1)).float().unsqueeze(0)).cpu().squeeze().view(-1).numpy()
        
    def __str__(self):
        return str(self.resnet)


    def forward(self, batch):

        h = self.resnet(batch)
        if self.norm is not None: h =  torch.nn.functional.normalize(h, p = self.norm, dim = 1)
        return h
    
