import sys
import tensorflow.keras
import pandas as pd
import sklearn as sk
import tensorflow as tf
import platform
import os
import sys
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
from mrcnn.config import Config
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
from mrcnn.model import log 
import skimage.io as io #to read in images and show predicted images

class ShapesConfig(Config):
        NAME = "pig"
        GPU_COUNT = 1
        IMAGES_PER_GPU = 1
        BACKBONE = "resnet50"
        NUM_CLASSES = 1 + 1 
        STEPS_PER_EPOCH = 300
        DETECTION_MIN_CONFIDENCE = 0.9
        IMAGE_MIN_DIM = 400
        USE_MINI_MASK = False

class PigDataset(utils.Dataset):
        # get the number of objects
        def get_obj_index(self, image):
            n = np.max(image)
            return n

        # interpret the yaml file 
        def from_yaml_get_class(self, image_id):
            info = self.image_info[image_id]
            with open(info['yaml_path']) as f:
                temp = yaml.full_load(f.read())
                # temp = yaml.load(f.read(), Loader = yaml.FullLoader)
                labels = temp['label_names'] #change into "label_names"
                # print(labels)
                del labels[0]
            return labels

        def draw_mask(self, num_obj, mask, image,image_id):
            info = self.image_info[image_id]
            for index in range(num_obj):
                for i in range(info['width']):
                    for j in range(info['height']):
                        at_pixel = image.getpixel((i, j))
                        if at_pixel == index + 1:
                            mask[j, i, index] = 1
            return mask


        def load_shapes(self, img_floder, mask_floder, dataset_root_path):
            """Generate the requested number of synthetic images.
            count: number of images to generate.
            height, width: the size of the generated images.
            """
            # Add classes
            self.add_class("shapes", 1, "cow")
            imglist = os.listdir(img_floder)

            for i in range(0, len(imglist)):
                filestr = imglist[i].split(".")[0]
                if len(filestr)==0:
                    pass
                else:
                    mask_path = mask_floder + "/" + filestr + "_json/" + "label.png"
                    yaml_path = mask_floder + "/" + filestr + "_json/info.yaml"
                    cv_img = cv2.imread(img_floder + "/" + filestr + ".png")
                    print('cv_img: ', i, img_floder + "/" + filestr + ".png")
                    
                    self.add_image("shapes", image_id=i, path=img_floder + "/" + imglist[i],
                                    width=cv_img.shape[1], height=cv_img.shape[0], mask_path=mask_path, yaml_path=yaml_path)

        def load_mask(self, image_id):
            """Generate instance masks for shapes of the given image ID.
            """
            global iter_num
            print("image_id",image_id)
            info = self.image_info[image_id]
            count = 1  # number of object
            img = Image.open(info['mask_path'])
            num_obj = self.get_obj_index(img)
            mask = np.zeros([info['height'], info['width'], num_obj], dtype=np.uint8)
            mask = self.draw_mask(num_obj, mask, img,image_id)
            occlusion = np.logical_not(mask[:, :, -1]).astype(np.uint8)
            for i in range(count - 2, -1, -1):
                mask[:, :, i] = mask[:, :, i] * occlusion

                occlusion = np.logical_and(occlusion, np.logical_not(mask[:, :, i]))
            labels = []
            labels = self.from_yaml_get_class(image_id)
            labels_form = []
            for i in range(len(labels)):
                if labels[i].find("cow") != -1:
                    labels_form.append("cow")
            class_ids = np.array([self.class_names.index(s) for s in labels_form])
            return mask, class_ids.astype(np.int32)

def get_ax(rows=1, cols=1, size=8):
    _, ax = plt.subplots(rows, cols, figsize=(size * cols, size * rows))
    return ax

# set configurations 
class InferenceConfig(ShapesConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    DETECTION_MIN_CONFIDENCE = 0.3
    USE_MINI_MASK = False


#################################################################
############# Predict ###########################################
#################################################################


def predict_mrcnn(image):
    ROOT_DIR = os.path.abspath("/Users/yebi/Library/CloudStorage/OneDrive-VirginiaTech/Research/Codes/research/BCS/BodyWeight/MaskRCNN/")
    MODEL_DIR = os.path.join(ROOT_DIR, "logs")
    sys.path.append(ROOT_DIR)  # To find local version of the library
   
        
    config = ShapesConfig()
    # config.display()

  
    inference_config = InferenceConfig()

    # Recreate the model in inference mode
    model = modellib.MaskRCNN(mode="inference", 
                              config=inference_config,
                              model_dir=MODEL_DIR)

    model_path = "/Users/yebi/Library/CloudStorage/OneDrive-VirginiaTech/Research/Codes/research/BCS/BodyWeight/BW_github_beta/python/mrcnn/MaskRCNN_Train/mask_rcnn_cow.h5"

    # Load trained weights
    tf.keras.Model.load_weights(model.keras_model, model_path, by_name=True)



    results = model.detect([image], verbose=1)
    # ax = get_ax(1)
    # r = results[0]
    # # visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'], 
    # #                             ['BG', 'cow'], r['scores'], ax=ax,
    # #                             title="Predictions")

    # # predict
    # c2 = np.argwhere(r['masks'][:,:,0])
    # mask2 = np.zeros(image[:,:,0].shape, dtype = image.dtype) 
    # fill_img2 = cv2.drawContours(mask2, [np.flip(c2, axis = 1)], 0, (255), 0) 
    # # cv2.imwrite("binary.png", fill_img2)
    # # fill_img2
    

    print("\n ####################################### mrcnn predict Done ######################################## \n")
    return model




