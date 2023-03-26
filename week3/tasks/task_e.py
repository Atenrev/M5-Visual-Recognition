import os
import cv2
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from matplotlib import gridspec

import torch
import tensorflow as tf
import tensorflow_hub as hub

from detectron2.data import MetadataCatalog
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer, ColorMode


class StyleTransfer:
    def __init__(self, cfg):
        self._cfg = cfg
        self._data_dir = os.getcwd()
        self._device = None
        self._resnet50 = None
        self._utils = None
        self._hub_model = None

        self._set_device()
        self._set_utils()
        self._set_resnet50()
        self._set_fusion_model()
        self._set_predictor()

    def _set_predictor(self):
        self._cfg.MODEL.DEVICE = 'cpu'
        self._predictor = DefaultPredictor(self._cfg)

    def set_dataset(self, dataset: str):
        self._data_dir = os.path.join(self._data_dir, dataset)

    def _set_device(self):
        self._device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    def _set_resnet50(self):
        self._resnet50 = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_resnet50', pretrained=True)
        self._resnet50.eval().to(self._device)

    def _set_utils(self):
        self._utils = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_convnets_processing_utils')

    def _set_fusion_model(self):
        self.fusion_model = hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2')

    def classify_resnet50(self, paths_img):
        batch = torch.cat([self._utils.prepare_input_from_uri(uri) for uri in paths_img]).to(self._device)

        with torch.no_grad():
            output = torch.nn.functional.softmax(self._resnet50(batch), dim=1)

        results = self._utils.pick_n_best(predictions=output, n=5)
        return results

    @staticmethod
    def tensor_to_image(tensor):
        tensor = tensor * 255
        tensor = np.array(tensor, dtype=np.uint8)
        if np.ndim(tensor) > 3:
            assert tensor.shape[0] == 1
            tensor = tensor[0]
        return Image.fromarray(tensor)

    @staticmethod
    def load_image(path_to_img):
        max_dim = 512
        img = tf.io.read_file(path_to_img)
        img = tf.image.decode_image(img, channels=3)
        img = tf.image.convert_image_dtype(img, tf.float32)

        shape = tf.cast(tf.shape(img)[:-1], tf.float32)
        long_dim = max(shape)
        scale = max_dim / long_dim

        new_shape = tf.cast(shape * scale, tf.int32)

        img = tf.image.resize(img, new_shape)
        img = img[tf.newaxis, :]
        return img

    @staticmethod
    def imshow(image, title=None):
        if len(image.shape) > 3:
            image = tf.squeeze(image, axis=0)

        plt.imshow(image)
        if title:
            plt.title(title)
        plt.show()

    @staticmethod
    def show_n(images, titles=('',), classes=('',), save='summary.jpg'):
        n = len(images)
        image_sizes = [image.shape[1] for image in images]
        w = (image_sizes[0] * 6) // 320
        plt.figure(figsize=(w * n, w))
        gs = gridspec.GridSpec(1, n, width_ratios=image_sizes)
        for i in range(n):
            plt.subplot(gs[i])
            plt.imshow(images[i][0], aspect='equal')
            plt.axis('off')
            class_info = '\n'.join([f"{cls[0]}    {cls[1]}" for cls in classes[i][:3]])
            plt.title(titles[i] + '\n' + class_info if len(titles) > i else '', fontdict={'fontsize': 18})
        plt.savefig(save)
        plt.close()
        StyleTransfer.rgba2rgb(save)

    def evaluate_classification(self, content_path, style_path):
        content_path = os.path.join(self._data_dir, content_path)
        style_path = os.path.join(self._data_dir, style_path)
        content_image = self.load_image(content_path)
        style_image = self.load_image(style_path)
        stylized_image = self.fusion_model(tf.constant(content_image), tf.constant(style_image))[0]

        im = StyleTransfer.tensor_to_image(stylized_image)
        stylized_path = os.path.join(self._data_dir, f"{os.path.basename(content_path).split('.')[0]}"
                                                     f"_{os.path.basename(style_path).split('.')[0]}.jpg")
        im.save(stylized_path)

        images = [content_image, style_image, stylized_image]
        titles = ['Original content image', 'Style image', 'Stylized image']
        paths = [content_path, style_path, stylized_path]
        classes = self.classify_resnet50(paths)

        pred_path = os.path.join(self._data_dir, f"pred_{os.path.basename(content_path).split('.')[0]}"
                                                     f"_{os.path.basename(style_path).split('.')[0]}.jpg")
        self.show_n(images, titles, classes, save=pred_path)

    @staticmethod
    def rgba2rgb(img_path):
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
        cv2.imwrite(img_path, img)

    def evaluate_object_detection(self, image_path):
        img = cv2.imread(os.path.join(self._data_dir, image_path))
        outputs = self._predictor(img)
        v = Visualizer(
            img[:, :, ::-1],
            MetadataCatalog.get(self._cfg.DATASETS.TRAIN[0]),
            scale=1.2,
            instance_mode=ColorMode.SEGMENTATION,
        )
        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        cv2.imwrite(
            os.path.join(self._data_dir, f'detection_{image_path}'),
            out.get_image()[:, :, ::-1],
            [int(cv2.IMWRITE_PNG_COMPRESSION), 9]
        )


def run(cfg, args):
    """
    Run the style transfer
    """
    style_transfer = StyleTransfer(cfg)
    style_transfer.set_dataset('data/style_transfer')

    style_transfer.evaluate_classification('hair_dryer.jpg', 'texture_sofa.jpg')
    style_transfer.evaluate_object_detection('hair_dryer.jpg')
    style_transfer.evaluate_object_detection('texture_sofa.jpg')
    style_transfer.evaluate_object_detection('hair_dryer_texture_sofa.jpg')
