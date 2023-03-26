Vision Transformer Encoder from scratch using only convolution layer instead of linear layers

# TSAI - EVA8 Session 10 Assignment

## Problem Statement
---------------------

Check out this [network](https://github.com/kentaroy47/vision-transformers-cifar10/blob/main/models/vit.py):

1. Re-write this network such that it is similar to the network we wrote in the class  
2. All parameters are the same as the network we wrote  
3. Proceed to submit the assignment:  
    1. Share the model code and link to the model cost  
    2. Share the training logs  
    ---------
    ```ViT: Epoch: 0 | Train Acc: 0.1780, Test Acc: 0.2883, Time: 71.9, lr: 0.001000
ViT: Epoch: 1 | Train Acc: 0.2807, Test Acc: 0.3714, Time: 65.8, lr: 0.002000
ViT: Epoch: 2 | Train Acc: 0.3436, Test Acc: 0.4328, Time: 64.1, lr: 0.003000
ViT: Epoch: 3 | Train Acc: 0.3933, Test Acc: 0.4798, Time: 63.1, lr: 0.004000
ViT: Epoch: 4 | Train Acc: 0.4330, Test Acc: 0.5002, Time: 65.6, lr: 0.005000
ViT: Epoch: 5 | Train Acc: 0.4564, Test Acc: 0.5178, Time: 64.3, lr: 0.006000
ViT: Epoch: 6 | Train Acc: 0.4816, Test Acc: 0.5498, Time: 64.9, lr: 0.007000
ViT: Epoch: 7 | Train Acc: 0.4980, Test Acc: 0.5481, Time: 66.2, lr: 0.008000
ViT: Epoch: 8 | Train Acc: 0.5129, Test Acc: 0.5611, Time: 63.8, lr: 0.009000
ViT: Epoch: 9 | Train Acc: 0.5261, Test Acc: 0.5691, Time: 65.3, lr: 0.010000
ViT: Epoch: 10 | Train Acc: 0.5385, Test Acc: 0.6042, Time: 64.7, lr: 0.009050
ViT: Epoch: 11 | Train Acc: 0.5614, Test Acc: 0.6253, Time: 63.5, lr: 0.008100
ViT: Epoch: 12 | Train Acc: 0.5723, Test Acc: 0.6294, Time: 65.4, lr: 0.007150
ViT: Epoch: 13 | Train Acc: 0.5840, Test Acc: 0.6392, Time: 64.5, lr: 0.006200
ViT: Epoch: 14 | Train Acc: 0.5929, Test Acc: 0.6500, Time: 64.5, lr: 0.005250
ViT: Epoch: 15 | Train Acc: 0.6055, Test Acc: 0.6678, Time: 65.9, lr: 0.004300
ViT: Epoch: 16 | Train Acc: 0.6233, Test Acc: 0.6817, Time: 64.0, lr: 0.003350
ViT: Epoch: 17 | Train Acc: 0.6342, Test Acc: 0.6858, Time: 64.5, lr: 0.002400
ViT: Epoch: 18 | Train Acc: 0.6477, Test Acc: 0.6881, Time: 66.7, lr: 0.001450
ViT: Epoch: 19 | Train Acc: 0.6580, Test Acc: 0.7069, Time: 65.6, lr: 0.000500
ViT: Epoch: 20 | Train Acc: 0.6696, Test Acc: 0.7092, Time: 65.1, lr: 0.000400
ViT: Epoch: 21 | Train Acc: 0.6689, Test Acc: 0.7092, Time: 65.2, lr: 0.000300
ViT: Epoch: 22 | Train Acc: 0.6719, Test Acc: 0.7111, Time: 64.6, lr: 0.000200
ViT: Epoch: 23 | Train Acc: 0.6719, Test Acc: 0.7104, Time: 68.8, lr: 0.000100
ViT: Epoch: 24 | Train Acc: 0.6747, Test Acc: 0.7112, Time: 63.9, lr: 0.000000```
    
    3. Share the gradcam images for 10 misclassified images. 
