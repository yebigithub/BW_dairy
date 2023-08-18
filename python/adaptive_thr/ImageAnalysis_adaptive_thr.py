import os
import argparse
import sys
from scipy import stats # summarize data
import csv
from scipy.spatial import distance as dist
from imutils import perspective
import numpy as np
import cv2
import pandas as pd

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

def threshold_selecting(thr_max, hue):
  '''
  Input:
  thr_max: max of threshold you prefer to try. Default is 45
  hue: Hue degree from HSV model of depth image.
  Output:
  fill_img: binary image after thresholding
  '''
  for thresh in range(np.min(hue), thr_max):
      ##part00. finding hue threshold.
    thresh, thresh_img = cv2.threshold(hue, thresh, 255, cv2.THRESH_BINARY)
    cnts, _ = cv2.findContours(thresh_img.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE) 
    cmax = max(cnts, key=cv2.contourArea) 
    mask = np.zeros(thresh_img.shape, dtype = thresh_img.dtype) 
    fill_img = cv2.drawContours(mask, [cmax], 0, (255), -1) 
    x,y,w,hh = cv2.boundingRect(cmax)
        # QC for touching boundires
    if (y <= 5 or y+hh >= 245 or x <= 5): 
        if (thresh < thr_max-1):
          continue
        else:
           print('Cannot find the best threshold for this image \n')
           qc = False
           return qc, fill_img
    else: #use this hue threshold
      print("The best threshold is: ", thresh)
      print("x, y, y+h are:", x, y, y+hh)
      qc = True
      return qc, fill_img
                    
def neck_removal(fill_img, neck_threshold):
    '''
    Input:
    fill_img: binary image after thresholding
    neck_threshold: default = 0.3 # Neck threshold (cols with this number or less white pixels is classified as neck)
    Output:
    fill_img: image after thresholding and neck removal
    '''
    neck_threshold = neck_threshold
    white_part = np.sum(fill_img, axis=0) / 255
    neck_cols = (white_part/np.max(white_part) < neck_threshold) # all columns where less then some threshold of white pixels
    deleted_cols = np.where(neck_cols)[0]
    deleted_mask = ~(deleted_cols > fill_img.shape[1] / 2) # make sure that we keep the cols on the butt end
    deleted_cols = np.delete(deleted_cols, deleted_mask) # remove those cols from deletion
    
    if deleted_cols.shape[0] != 0:
        print(f"Neck will be removed with ratio {neck_threshold} \n")
        fill_img[:, deleted_cols[0]+1:] = np.zeros_like(fill_img[:, deleted_cols[0]+1:])
    else: 
      print("Neck will be kept \n")
      fill_img = fill_img
    
    return fill_img


def wd_len_getting(fill_img):
    '''
    Input:
    fill_img: image after thresholding and neck removal.
    Output: 
    width: image parameter, in pixel.
    length: image parameter, in pixel.
    cmax: maximun contour in image, we need this for the following steps.
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

def height0_getting(cmax, dfcsv_crop):
    '''
    Input: 
    cmax: maximun contour in image
    Output:
    heights0: centroid height
    '''
    M = cv2.moments(cmax)
    row_centroid = int(M["m01"] / M["m00"])
    col_centroid  = int(M["m10"] / M["m00"])
    height0 = 2.94 - dfcsv_crop.iloc[row_centroid , col_centroid]
    return height0


def height1_getting(fill_img, dfcsv_crop):
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
      dfcsv_rows.append([row, col, dfcsv_crop.iloc[row, col]])
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
############### Run Adpative threshold method #############################
###########################################################################

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
            for file in files:
                
                #Initialize summ file.
                frame = os.path.splitext(file)[0]                   
                width = np.nan
                length = np.nan
                height0 = np.nan 
                height1 = np.nan 
                volume = np.nan

                #Reading depth images and csv files.
                file_path = os.path.join(root, file)
                print("Now is running: ", file_path)
                if file_path.split("/")[6].split("_")[0] == ".":    #remove irregular symbols in filenames.
                  continue
                img = cv2.imread(file_path)                         # read in one depth image.
                csv_filename = os.path.splitext(file)[0]+".csv"     # find corresponding csv file name.
                csv_path = os.path.join(csvdir, csv_filename)       # find csv file path
                dfcsv = pd.read_csv(csv_path, header = None)        # read in depth csv file.
                dfcsv_crop = dfcsv.iloc[140:390, 120:750]           # crop csv depending on ur environment.
                img_crop = img[140:390, 120:750]                    # crop image depending on ur environment.

                #Convert RGB into HSV
                hsv = cv2.cvtColor(img_crop, cv2.COLOR_BGR2HSV)     # convert RGB into HSV
                hue = hsv[:, :, 0]                                  # extract Hue out of HSV
                
                ##part1. Convert into binary image with best threshold.
                qc, fill_img = threshold_selecting(thr_max=46, hue=hue)
                
                if qc:

                  ##part2. Neck removing
                  fill_img = neck_removal(fill_img, neck_threshold=0.3)

                  ##part3: calculate width and length
                  width, length, cmax = wd_len_getting(fill_img)

                  ##part4: calculated height: centroid method
                  height0 = height0_getting(cmax, dfcsv_crop)

                  ##part5: calculate height: average method
                  height1, df = height1_getting(fill_img, dfcsv_crop)

                  ##part6: calculate volume
                  volume = volume_getting(df, camera_height=2.94)

                  ##part7: write all the image parameters into csv file.
                  writer.writerow([Day, ID, frame, width, length, height0, height1, volume])