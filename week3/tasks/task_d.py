import json
from pathlib import Path
import numpy as np
import pandas as pd
import cv2
from PIL import Image
import torch

from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer

MODELS = {
    "mask": 'COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml',
    "faster": 'COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml',
}

MODEL = MODELS['mask']
device = 'cuda'

def fgsm_attack(image, epsilon, data_grad):
    # Collect the element-wise sign of the data gradient
    #sign_data_grad = data_grad.sign()

    # Create the perturbed image by adjusting each pixel of the input image
    perturbed_image = image + epsilon*data_grad
    # Adding clipping to maintain [0,1] range
    perturbed_image = torch.clamp(perturbed_image, 0, 255)
    # Return the perturbed image
    return perturbed_image

def task_d(*args, attacked_image = './data/weird/el_bone.jpg', steps = 100, imsize = 1024):

    npimage = cv2.imread(attacked_image)
    h, w, c = npimage.shape
    data = torch.from_numpy(npimage.transpose(2, 0, 1)).float().to(device) 

    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(MODEL))
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(MODEL)
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5

    # Normalize the input image    

    predictor = DefaultPredictor(cfg)
    model = predictor.model
    loss = torch.nn.BCELoss()
    perturbation = torch.rand_like(data)
    perturbation.requires_grad = True
    for step in range(steps):

        model.zero_grad()

        input_data = torch.clamp(data + perturbation * 127, 0, 255)
        output = model([{'image': input_data}])[0]
        scores = output['instances'].scores
        fake_scores = torch.zeros_like(scores)
        
        loss_value = loss(scores.unsqueeze(0), fake_scores.unsqueeze(0)) - torch.sum((perturbation**2), dim = -1) / (h * w)
        print(loss_value.item())

        loss_value.backward()
        data_grad = perturbation.grad
        #perturbed_data = fgsm_attack(data, 150, data_grad)
        perturbation = perturbation + data_grad * 0.1
        
        perturbed_data = perturbation.detach().cpu().numpy()
        perturbation = torch.from_numpy(perturbed_data).to(device)
        perturbation.requires_grad = True
        #perturbed_data = torch.from_numpy(perturbed_data)
                
    #### VISUALIZER ZONE #####
    adversarial_image = np.clip(data.cpu().detach().numpy() + perturbation.cpu().detach().numpy() * 127, 0, 255)
    print('writting image shape', adversarial_image.shape)
    adversarial_image = adversarial_image.transpose(1, 2, 0)
    outs = predictor(adversarial_image)
    viz = Visualizer(
        adversarial_image[:, :, ::-1],
        MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2
    )
    out = viz.draw_instance_predictions(outs["instances"].to("cpu"))
    print(out.get_image().shape)

    cv2.imwrite(
        'tmp.png',
        out.get_image()[:, :, ::-1],
    )


if __name__ == '__main__': 
    task_d()