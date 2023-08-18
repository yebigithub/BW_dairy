import os
import argparse
import sys
from scipy import stats # summarize data
import csv
from scipy.spatial import distance as dist
# from imutils import perspective
import numpy as np
import cv2
import pandas as pd

sys.path.append("./maskrcnn")
import predict_mrcnn
import skimage.io as io

# os.chdir('/Users/yebi/Library/CloudStorage/OneDrive-VirginiaTech/Research/Codes/research/BCS/BodyWeight') 
os.chdir('./')
parser = argparse.ArgumentParser(description = 'Extracting image descriptors from image')
parser.add_argument('day', help = 'day info.')
args = parser.parse_args()

#setting depth images and CSV files location.
# rootdir = "/Volumes/MyPassport1"
rootdir = "./Sample_files/Depth/"
dep_folder = rootdir + args.day + "/depth/"
csv_folder = rootdir + args.day + "/CSV/"
day_folder = "./outputs/" + args.day + "/" + args.day + "_"
# img_out = "./outputs/imgs/D1/" #to check if image analysis works well.

if not os.path.exists('./outputs/' + args.day):
  os.mkdir("./outputs/" + args.day)
  print("Directory created")

########################################
########Functions###############
########################################
def wd_len_getting(fill_img):
  '''
  Input: 
  fill_image: image after Mask RCNN prediction
  Outputs:
  width: image parameter, in pixel
  length: image parameter, in pixel
  cma: maximun contour in image, we need this for the following steps.
  '''
  cnts, _ = cv2.findContours(fill_img.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE) 
  cmax = max(cnts, key=cv2.contourArea) 
  rect = cv2.minAreaRect(cmax)
  box = cv2.boxPoints(rect)
  (A, B, C, D) = np.int0(box)
  d0 = dist.euclidean(B, C)
  d1 = dist.euclidean(A, B)
  width = min(d0, d1)
  length = max(d0, d1)
  return width, length, cmax

def height0_getting(cmax, dfcsv):
    '''
    Input: 
    cmax: maximun contour in image
    Output:
    heights0: centroid height
    '''
    M = cv2.moments(cmax)
    row_centroid = int(M["m01"] / M["m00"])
    col_centroid  = int(M["m10"] / M["m00"])
    height0 = 2.94 - dfcsv.iloc[row_centroid , col_centroid]
    return height0

def height1_getting(fill_img, dfcsv):
    '''
    Input: 
    fill_img: binary image after thresholding and neck removal.
    dfcsv_crop: depth csv dataframe after cropping.
    Output:
    height1: average height.
    df: depth csv dataframe after removing outliers.
    '''
    pixel = np.argwhere(fill_img == 255) #find pixels for white part
    dfcsv_rows = [] #combine pixel and distance 
    for row, col in pixel:
      dfcsv_rows.append([row, col, dfcsv.iloc[row, col]])
    df = pd.DataFrame(dfcsv_rows, columns = ['row', 'col', 'dist'])
    df.dist.replace(to_replace=0, value = df.dist.mean(), inplace=True) #replace 0 with average distance
    height1 = 2.94 - df.dist.mean()
    return height1, df

def volume_getting(df, camera_height=2.94):
    '''
    Input:
    df: depth csv dataframe after removing outliers.
    camera_height: height of your depth camera.
    Output:
    volume: Add all heights from all pixels together.
    '''
    df["height"] = camera_height - df["dist"] #build new column named height
    volume = sum(df.height)
    return volume

###########################################################################
############################ Run MRCNN method #############################
###########################################################################
i = 1
for cowid in os.listdir(dep_folder):
  summ = os.path.join(day_folder+cowid+".csv")
  if os.path.isfile(summ):
    print("already there, please move these files to another folder.")
    continue
  else:
    depthdir = dep_folder + cowid + "/"
    csvdir = csv_folder + cowid + "/"
    with open(summ, "w", newline = "") as output:
        writer = csv.writer(output)
        writer.writerow(["Day", "ID", "Frame", "Width", "Length", "Height_Centroid", "Height_average", "Volume"])
        for root, dirs, files in os.walk(depthdir):
          Day = root.split("/")[3]
          ID = root.split("/")[5]
          for j in np.arange(3, len(files), 15): # one pic per 15 frames
              file = files[j]
             
              #Initialize summ file.
              frame = os.path.splitext(file)[0]
              width = np.nan
              length = np.nan
              height0 = np.nan 
              height1 = np.nan 
              volume = np.nan
              
              #Reading depth images and csv files.
              file_path = root + file
              if file_path.split("/")[6].split("_")[0] == ".":    #remove irregular symbols in filenames.
                continue
              print("Now is running: ", file_path)
              
              img = io.imread(file_path)  # Read in images.
              csv_filename = os.path.splitext(file)[0]+".csv" # Read in distance csv files
              csv_path = os.path.join(csvdir, csv_filename)
              dfcsv = pd.read_csv(csv_path, header = None) #read in depth csv file

              ##part1: Using mrcnn to predict contour (result is binary image with size 848*480) 
              fill_img = predict_mrcnn.predict_mrcnn(img) #bianry image after mrcnn 
              
              ##part2: calculate width and length
              width, length, cmax = wd_len_getting(fill_img)

              ##part3: calculated height: centroid method
              height0 = height0_getting(cmax, dfcsv)

              ##part4: calculate height: average method
              height1, df = height1_getting(fill_img, dfcsv)

              ##part5: calculate volume
              volume = volume_getting(df, camera_height=2.94)

              ##part6: write all the image parameters into csv file.
              writer.writerow([Day, ID, frame, width, length, height0, height1, volume])
              print("####################### Done %d ############################" %i) 
              i = i + 1
                      
