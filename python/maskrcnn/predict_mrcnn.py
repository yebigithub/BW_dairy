import sys
import tensorflow.keras
import pandas as pd
import sklearn as sk
import tensorflow as tf
import platform
import os
import random
import math
import re
import time
import numpy as np
import cv2
import matplotlib
import matplotlib.pyplot as plt
import yaml
from PIL import Image
import imgaug  # https://github.com/aleju/imgaug (pip3 install imgaug)
import skimage.io as io

sys.path.append("./maskrcnn/MaskRCNN_Train")
from mrcnn.config import Config
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
from mrcnn.model import log 
import skimage.io as io #to read in images and show predicted images

class ShapesConfig(Config):
        NAME = "cow"
        GPU_COUNT = 1
        IMAGES_PER_GPU = 1
        BACKBONE = "resnet50"
        NUM_CLASSES = 1 + 1 
        STEPS_PER_EPOCH = 300
        DETECTION_MIN_CONFIDENCE = 0.9
        IMAGE_MIN_DIM = 400
        USE_MINI_MASK = False

def get_ax(rows=1, cols=1, size=8):
    _, ax = plt.subplots(rows, cols, figsize=(size * cols, size * rows))
    return ax

# set configurations for inference.
class InferenceConfig(ShapesConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    DETECTION_MIN_CONFIDENCE = 0.3
    USE_MINI_MASK = False


#################################################################
############# Now Predict #######################################
#################################################################

def predict_mrcnn(image):
    ROOT_DIR = os.path.abspath("./maskrcnn/MaskRCNN_Train/Mask_RCNN")

    # ROOT_DIR = os.path.abspath("/Users/yebi/Library/CloudStorage/OneDrive-VirginiaTech/Research/Codes/research/BCS/BodyWeight/BW_diary/python/mrcnn/MaskRCNN_Train/Mask_RCNN")
    MODEL_DIR = os.path.join(ROOT_DIR, "logs")
    sys.path.append(ROOT_DIR)  # To find local version of the library
  
    inference_config = InferenceConfig()

    # Recreate the model in inference mode
    model = modellib.MaskRCNN(mode="inference", 
                              config=inference_config,
                              model_dir=MODEL_DIR)

    model_path = "/Users/yebi/Library/CloudStorage/OneDrive-VirginiaTech/Research/Codes/research/BCS/BodyWeight/MaskRCNN_training/MaskRCNN_Code/mask_rcnn_cow.h5"

    # Load trained weights
    tf.keras.Model.load_weights(model.keras_model, model_path, by_name=True)

    # Predict with mrcnn
    results = model.detect([image], verbose=0)
    r = results[0] # keep the results from the largest IoU
    c2 = np.argwhere(r['masks'][:,:,0])
    mask2 = np.zeros(image[:,:,0].shape, dtype = image.dtype)
    fill_img2 = cv2.drawContours(mask2, [np.flip(c2, axis = 1)], 0, (255), 0) # Draw cow contour out
    
    print("\n ####################################### mrcnn predict Done ######################################## \n")
    return fill_img2




