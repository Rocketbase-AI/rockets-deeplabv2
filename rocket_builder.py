import os
from .model import DeepLabv2_MSC
import types
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from PIL import ImageDraw
from torchvision import transforms
# from .utils import *
import numpy as np
import json
import matplotlib.pyplot as plt


def build(config_path: str = '') -> nn.Module:
    """Builds a pytorch compatible deep learning model

    The model can be used as any other pytorch model. Additional methods
    for `preprocessing`, `postprocessing`, `label_to_class` have been added to ease handling of the model
    and simplify interchangeability of different models.
    """
    # Load Config file
    if not config_path: # If no config path then load default one
        config_path = os.path.join(os.path.realpath(os.path.dirname(__file__)), "config.json")

    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Load the classes
    classes_path = os.path.join(os.path.realpath(os.path.dirname(__file__)), config['classes_path'])
    
    with open(classes_path, 'r') as f:
        classes =  json.load(f)

    # Set up model
    model = DeepLabv2_MSC(classes, config['n_blocks'], config['atrous_rates'], config['scales'] )
    weights_path = os.path.join(os.path.realpath(os.path.dirname(__file__)), config['weights_path'])
    model.load_state_dict(torch.load(weights_path), strict=True)

    
    model.postprocess = types.MethodType(postprocess, model)
    model.preprocess = types.MethodType(preprocess, model)
    setattr(model, 'classes', classes)
    setattr(model, 'config', config)

    return model

def preprocess(self, img: Image):
    """Converts PIL Image into pytorch tensor specific to this model

    Handles all the necessary steps for preprocessing such as resizing, normalization.

    Args:
        img (PIL.Image): input image
    """

    # Resize: for PIL image Image.size --> (width, height)
    scale = self.config['input_size'][0] / max(img.size[:2])
    
    input_width = min([int(img.size[0] * scale), self.config['input_size'][0]])
    input_height = min([int(img.size[1] * scale), self.config['input_size'][0]])
    
    input_img = img.resize((input_width, input_height), resample=Image.BILINEAR)
    input_img = np.array(input_img.convert("RGB"))
    
    # image = cv2.resize(image, dsize=None, fx=scale, fy=scale)
    # raw_image = image.astype(np.uint8)

    # Original image was loaded with cv2 --> BGR
    input_img =  input_img[:,:,::-1]

    # Subtract mean values
    input_img = input_img.astype(np.float32)
    input_img -= np.array(
        [
            float(self.config['mean_RGB'][2]),
            float(self.config['mean_RGB'][1]),
            float(self.config['mean_RGB'][0]),
        ]
    )

    # Convert to torch.Tensor and add "batch" axis
    input_img = torch.from_numpy(input_img.transpose(2, 0, 1)).float().unsqueeze(0)

    return input_img

def postprocess(self, detections: torch.Tensor, img: Image, visualize: bool = False):
    """Converts pytorch tensor into interpretable format

    Handles all the steps for postprocessing of the raw output of the model.
    Depending on the rocket family there might be additional options.

    Args:
        detections (Tensor): Output Tensor to postprocess
        input_img (PIL.Image): Original input image which has not been preprocessed yet
        visualize (bool): If True outputs image with annotations else a list of bounding boxes
    """
    # Get the size of the input_img
    scale = self.config['input_size'][0] / max(img.size[:2])
    
    input_width = min([int(img.size[0] * scale), self.config['input_size'][0]])
    input_height = min([int(img.size[1] * scale), self.config['input_size'][0]])
    
    input_img = img.resize((input_width, input_height), resample=Image.BILINEAR)
    input_img = np.array(input_img.convert("RGB"))
    raw_image = input_img.astype(np.uint8)

    H = input_height
    W = input_width

    # Image -> Probability map
    logits = F.interpolate(detections, size=(H, W), mode="bilinear", align_corners=False)
    probs = F.softmax(logits, dim=1)[0]
    probs = probs.cpu().numpy()

    # Refine the prob map with CRF
    # if postprocessor and raw_image is not None:
    #     probs = postprocessor(raw_image, probs)

    labelmap = np.argmax(probs, axis=0)
    
    labels = np.unique(labelmap)
    
    dict_mask = {}

    for i, label in enumerate(labels):
        mask = labelmap == label
        dict_mask[self.classes[str(label)]] = mask.astype(np.float32)

    if visualize:
        visible_labels = dict_mask.keys()
        
        # Show result for each class
        rows = np.floor(np.sqrt(len(visible_labels) + 1))
        cols = np.ceil((len(visible_labels) + 1) / rows)

        plt.figure(figsize=(10, 10))
        ax = plt.subplot(rows, cols, 1)
        ax.set_title("Input image")
        ax.imshow(raw_image)
        ax.axis("off")

        for i, label in enumerate(visible_labels):
            mask = dict_mask[label]
            ax = plt.subplot(rows, cols, i + 2)
            ax.set_title(label)
            ax.imshow(raw_image)
            ax.imshow(mask, 'jet', alpha=0.5)
            ax.axis("off")

        plt.tight_layout()
        plt.show()

        # Convert the figure to a PIL image
        # # draw the renderer
        # fig.canvas.draw ( )
    
        # # Get the RGBA buffer from the figure
        # w,h = fig.canvas.get_width_height()
        # buf = numpy.fromstring ( fig.canvas.tostring_argb(), dtype=numpy.uint8 )
        # buf.shape = ( w, h,4 )
    
        # # canvas.tostring_argb give pixmap in ARGB mode. Roll the ALPHA channel to have it in RGBA mode
        # buf = numpy.roll ( buf, 3, axis = 2 )




        # line_width = 2
        # img_out = input_img
        # ctx = ImageDraw.Draw(img_out, 'RGBA')
        # for detection in list_detections:
        #     # Extract information from the detection
        #     topLeft = (detection['topLeft_x'], detection['topLeft_y'])
        #     bottomRight = (detection['topLeft_x'] + detection['width'] - line_width, detection['topLeft_y'] + detection['height']- line_width)
        #     class_name = detection['class_name']
        #     bbox_confidence = detection['bbox_confidence']
        #     class_confidence = detection['class_confidence']

        #     # Draw the bounding boxes and the information related to it
        #     ctx.rectangle([topLeft, bottomRight], outline=(255, 0, 0, 255), width=line_width)
        #     ctx.text((topLeft[0] + 5, topLeft[1] + 10), text="{}, {:.2f}, {:.2f}".format(class_name, bbox_confidence, class_confidence))

        # del ctx
        # return img_out

    return dict_mask

