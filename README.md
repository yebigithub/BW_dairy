# Depth video data-enabled predictions of longitudinal dairy cow body weight using thresholding and Mask R-CNN algorithms

**Preprint on arXiv: https://arxiv.org/abs/2307.01383**  


This repository includes:  
1. Quick sample code.
2. Thresholding methods
    * 2.1 Single threshold method.
    * 2.2 Adaptive threshold method.
3. Mask RCNN method
    * 3.1 LabelMe.
    * 3.2 Mask-RCNN method
        * 3.2.1 Training Mask-RCNN model
        * 3.2.2 Predict depth images by trained Mask-RCNN model.
4. Build Body Weight Regression Model
    * 4.1 Data Preprocessing
    * 4.2 Goodness of fit
    * 4.3 Cross-Validation 1
    * 4.4 Cross-Validation 2

## 1. Quick sample code.
### 1.1 Establish your environment. 
- Download [python folder](https://github.com/yebigithub/BW_dairy/tree/main/python) from this github into your local postion.
- Install all the packages you will use in [requirement.txt](https://github.com/yebigithub/BW_dairy/blob/main/python/requirements.txt). Recreate one [conda environment](https://conda.io/projects/conda/en/latest/index.html) follow the lines.
```
# $ conda create --name <env> --file <requirement.txt>
```
### 1.2 Run image analysis.  
#### Step1. Create outputs directory.
- Build one empty folder named ```outputs``` within your local folder ```python``` which you downloaded from this github repository.
- If you want to run for your own figures, please make sure your files following our [```Sample_files```](https://github.com/yebigithub/BW_dairy/tree/main/python/Sample_files) folder structure.  
<p align="center">
<img src='https://github.com/yebigithub/BW_dairy/blob/main/picts/sample_file_structure.png' width='30%' height='30%'>
</p>
- Please make sure the ```outputs``` folder is empty before running each image analysis method. 

#### Step2. Choose one image analysis method. 
Select one image analysis method below and run the related code block
- Single threshold method:
```
python single_thr/ImageAnalysis_single_thr.py D1
```
- Adaptive threshold method:
```
python adaptive_thr/ImageAnalysis_adaptive_thr.py D1
```
- Mask R-CNN method:
```
python maskrcnn/ImageAnalysis_mrcnn.py D1
```
#### Step3. Check your outputs. 
After running, each approach will generate one csv file in ```outputs/D1```folder similar to the following.   
<p align="center">
<img src='https://github.com/yebigithub/BW_dairy/blob/main/picts/outputs_csv.png' width='70%' height='70%'>
</p>

## 2. Thresholding methods
### 2.1 Single threshold method.
- [.py file](https://github.com/yebigithub/BW_dairy/blob/main/python/single_thr/ImageAnalysis_single_thr.py)

### 2.2 Adaptive threshold method.
- [.py file](https://github.com/yebigithub/BW_dairy/blob/main/python/adaptive_thr/ImageAnalysis_adaptive_thr.py)
<img src='https://github.com/yebigithub/BW_dairy/blob/main/picts/Thresholding.png' width='70%' height='70%'>

#### Main steps
<img src='https://github.com/yebigithub/BW_dairy/blob/main/picts/Figure1.png?raw=true' width='70%' height='70%'>
<img src='https://github.com/yebigithub/BW_dairy/blob/main/picts/Figure2.png?raw=true' width='70%' height='70%'>

![alt text](https://github.com/yebigithub/BW_dairy/blob/main/picts/volume.gif)

## 3. Mask RCNN method
### 3.1 LabelMe.
- The code was forked from https://github.com/uf-aiaos/ShinyAnimalCV
- [How to install LabelMe](https://github.com/yebigithub/BW_dairy/blob/main/python/maskrcnn/LabelMe/Install-StandAlone-labelme.txt)
- How to use LabelMe to label image

```
## How to used our customized LabelMe.
## Code used to start label png into json files.
labelme ./depth/cow.png -O ./outputs/cow.json
```
![alt text](https://github.com/yebigithub/BW_dairy/blob/main/python/maskrcnn/LabelMe/HowToLabelMe1.gif)

```
# Code used to transfer json files into folders.
labelme_json_to_dataset ./outputs/cow.json -o ./outputs/cow_json
```
### 3.2 Mask-RCNN method
#### 3.2.1 Training Mask-RCNN model
- The code was forked from https://github.com/uf-aiaos/ShinyAnimalCV
- [.ipynb file](https://github.com/yebigithub/BW_dairy/blob/main/python/maskrcnn/MaskRCNN_Train/YB_train_cow_TF2_8.ipynb) 
- [.h5 file]()

![alt text](https://github.com/yebigithub/BW_dairy/blob/main/python/maskrcnn/mrcnn.gif)

#### 3.2.2 Predict depth images by trained Mask-RCNN model.
- [.py file](https://github.com/yebigithub/BW_dairy/blob/main/python/maskrcnn/ImageAnalysis_mrcnn.py)
<img src='https://github.com/yebigithub/BW_dairy/blob/main/picts/MRCNN.png?raw=true' width='70%' height='70%'>

## 4. Build Body Weight Regression Model
### 4.1 Data Preprocessing
- [.Rmd file](https://github.com/yebigithub/BW_dairy/blob/main/Rcodes/Section01_DataPreprocessing.Rmd)
### 4.2 Goodness of fit
- [.Rmd file](https://github.com/yebigithub/BW_dairy/blob/main/Rcodes/Section02_BW_Prediction_CV0.Rmd)
### 4.3 Cross-Validation 1
- [.Rmd file](https://github.com/yebigithub/BW_dairy/blob/main/Rcodes/Section03_BW_Prediction_CV1.Rmd)
### 4.4 Cross-Validation 2
- [.Rmd file](https://github.com/yebigithub/BW_dairy/blob/main/Rcodes/Section04_BW_Prediction_CV2.Rmd)

<img src='https://github.com/yebigithub/BW_dairy/blob/main/picts/CV_design.png?raw=true' width='70%' height='70%'>
