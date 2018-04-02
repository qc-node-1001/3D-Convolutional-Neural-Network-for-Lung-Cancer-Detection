# 3D Convolutional Neural Network for Lung Cancer Detection

So I did some Medical Image Computing in my undergraduate engineering course and adored it! Since I'm interested in computer vision (and intelligence)
I wanted to see how AI, namely: deep learning can be applied in this Medical computer vision. Hence, motivated by some of the work done by
Nvidia as shown in GTC 2018 and deepmind medical research I wanted to apply 3D CNNs for cancer detection.

Therefore in this repository I apply 3D CNN for lung cancer detection on the Kaggle datascience bowl 2017 dataset: 
https://www.kaggle.com/c/data-science-bowl-2017/data

The data is not that large! Hence, a good idea might be to combine other data sources such as LUNA16 and then run the 3D CNNs. Unfortunately,
I am limited to my memory space, computing power and time. Hence, running a large network on a huge dataset for me during my studies was difficult, however I hope you find this helpful.

# Work Done

The repository comprises of 4 files:
1. Data Visualisation: this is where I simply import and examine the data aswell as apply other techniques such as 3D visualisation of the 
   lung CT scan, Watershed segmentation, applying edge filters like sobel operators and etc.
2. Model: this is a script of the 3D CNN model using 3 different libraries: Keras, TensorFlow.
3. Train_Test: this is where I train and test the model (as name implies)
4. Inference: this is where I deploy the model to see how it works.
