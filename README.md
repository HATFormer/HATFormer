# HATFormer
Official implementation of HATFormer
# 
## Dataset
The SMARS dataset used for evaluation is from [A 2D/3D multimodal data simulation approach with applications on urban semantic segmentation, building extraction and change detection](https://www.sciencedirect.com/science/article/pii/S092427162300254X)
We clip and split the large tile into small images, the preprocessed dataset can be download [here](https://pan.baidu.com/s/1TGbuuy1PAMKI4DegVE6QZA?pwd=7vux). 
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
Please download the trained model weights [here](https://pan.baidu.com/s/1NOyULHy3zBHb4ef4skuMVQ?pwd=z929 ), and place them in the result directory.
The evaluation and visulization are simple
```
# For evaluation in SParis 
--config=CD_hatformer_baseline_and_BHE_FME_AFA_sparis --eval --ckpt_version bsize4_AFA --save_img
# For evaluation in SVenice
--config=CD_hatformer_baseline_and_BHE_FME_AFA_svenice --eval --ckpt_version bsize4_AFA --save_img
```
