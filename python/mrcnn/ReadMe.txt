Codes:
1. ImageAnalysis_mrcnn.py
	I used this to run repeatedly for the depth images for all days.
2. predict_mrcnn.py
	this is the function I used in ImageAnalysis_mrcnn.py, so make sure they are in the same folder.

Folders:
3. folder MaskRCNN_Train is the script that I used to train the mrcnn model. Code is YB_train_cow_TF2_8.ipynb, ran it in google colab with GPU.
4. folder mrcnn is the package folder, customized one by updating newest functions and models. 05-23-2023. This may change again when there are new versions again.
5. folder LabelMe includes how to use labelme to manually label depth images.
