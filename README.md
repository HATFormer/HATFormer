# HATFormer
Official implementation of HATFormer
# 
## Dataset
The SMARS dataset used for evaluation is from [A 2D/3D multimodal data simulation approach with applications on urban semantic segmentation, building extraction and change detection](https://www.sciencedirect.com/science/article/pii/S092427162300254X)
We clip and split the large tile into small images, the preprocessed dataset can be download here().
# Train
For training, you can modify parameters in configs/xx.xml, or just keep the default and:
```
python main.py --config='the name of config file'
```
e.g.
```
python main.py --config='the name of config file' 
```
# Test
Please download the trained model weights here, and place them in the result directory.
The evaluation and visulization are simple
```
# For evaluation in SParis 
python train.py
# For evaluation in SVenice

```
