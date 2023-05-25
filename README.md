# Depth video data-enabled predictions of longitudinal dairy cow body weight using thresholding and Mask R-CNN algorithms
This reporsitory includes:
1. Thresholding methods
    * 1.1 Single threshold method.
    * 1.2 Adpative threshold method.
2. Mask RCNN method
    * 2.1 LabelMe.
    * 2.2 Mask-RCNN method
        * 2.2.1 Training Mask-RCNN model
        * 2.2.2 Predict depth images by trained Mask-RCNN model.
3. Build Body Weight Regression Model
    * 3.1 Data Preprocessing
    * 3.2 Goodness of fit
    * 3.3 Cross-Validation 1
    * 3.4 Cross-Validation 2

## 1. Thresholding methods
### 1.1 Single threshold method.
- [.py file](https://github.com/yebigithub/BW_github_beta/blob/main/python/single_thr/ImageAnalysis_single_thr.py)

### 1.2 Adpative threshold method.
- [.py file](https://github.com/yebigithub/BW_github_beta/blob/main/python/adpative_thr/ImageAnalysis_adpative_thr.py)
<img src='https://github.com/yebigithub/BW_github_beta/blob/main/picts/Thresholding.png' width='70%' height='70%'>

#### Main steps
<img src='https://github.com/yebigithub/BW_github_beta/blob/main/picts/Figure1.png?raw=true' width='70%' height='70%'>
<img src='https://github.com/yebigithub/BW_github_beta/blob/main/picts/Figure2.png?raw=true' width='70%' height='70%'>

![alt text](https://github.com/yebigithub/BW_github_beta/blob/main/picts/volume.gif)

## 2. Mask RCNN method
### 2.1 LabelMe.
- [How to install LabelMe](https://github.com/yebigithub/BW_github_beta/blob/main/python/mrcnn/LabelMe/Install-StandAlone-labelme.txt)
- How to use LabelMe to label image

```
## How to used our customized LabelMe.
## Code used to start label png into json files.
labelme ./depth/cow.png -O ./outputs/cow.json
```
![alt text](https://github.com/yebigithub/BW_github_beta/blob/main/python/mrcnn/LabelMe/HowToLabelMe1.gif)

```
# Code used to transfer json files into folders.
labelme_json_to_dataset ./outputs/cow.json -o ./outputs/cow_json
```
### 2.2 Mask-RCNN method
#### 2.2.1 Training Mask-RCNN model
- [.ipynb file](https://github.com/yebigithub/BW_github_beta/blob/main/python/mrcnn/MaskRCNN_Train/YB_train_cow_TF2_8.ipynb) 
- [.h5 file]()
#### 2.2.2 Predict depth images by trained Mask-RCNN model.
- [.py file](https://github.com/yebigithub/BW_github_beta/blob/main/python/mrcnn/ImageAnalysis_mrcnn.py)
<img src='https://github.com/yebigithub/BW_github_beta/blob/main/picts/MRCNN.png?raw=true' width='70%' height='70%'>

## 3. Build Body Weight Regression Model
### 3.1 Data Preprocessing
- [.Rmd file](https://github.com/yebigithub/BW_github_beta/blob/main/Rcodes/Section01_DataPreprocessing.Rmd)
### 3.2 Goodness of fit
- [.Rmd file](https://github.com/yebigithub/BW_github_beta/blob/main/Rcodes/Section02_BW_Prediction_CV0.Rmd)
### 3.3 Cross-Validation 1
- [.Rmd file](https://github.com/yebigithub/BW_github_beta/blob/main/Rcodes/Section03_BW_Prediction_CV1.Rmd)
### 3.4 Cross-Validation 2
- [.Rmd file](https://github.com/yebigithub/BW_github_beta/blob/main/Rcodes/Section04_BW_Prediction_CV2.Rmd)

<img src='https://github.com/yebigithub/BW_github_beta/blob/main/picts/CV_design.png?raw=true' width='70%' height='70%'>