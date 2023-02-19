Objective
---------

- Write a custom ResNet architecture for CIFAR10 that has the following architecture:
  - PrepLayer - Conv 3x3 s1, p1) >> BN >> RELU [64k]
  - Layer1 -
      - X = Conv 3x3 (s1, p1) >> MaxPool2D >> BN >> RELU [128k]
      - R1 = ResBlock( (Conv-BN-ReLU-Conv-BN-ReLU))(X) [128k] 
      - Add(X, R1)
   - Layer 2 -
      - Conv 3x3 [256k]
      - MaxPooling2D
      - BN
      - ReLU
    - Layer 3 -
      - X = Conv 3x3 (s1, p1) >> MaxPool2D >> BN >> RELU [512k]
      - R2 = ResBlock( (Conv-BN-ReLU-Conv-BN-ReLU))(X) [512k]
      - Add(X, R2)
    - MaxPooling with Kernel Size 4
    - FC Layer 
    - SoftMax
- Uses One Cycle Policy such that:
    - Total Epochs = 24
    - Max at Epoch = 5
    - LRMIN = FIND
    - LRMAX = FIND
    - NO Annihilation
- Uses this transform -RandomCrop 32, 32 (after padding of 4) >> FlipLR >> Followed by CutOut(8, 8)
- Batch size = 512
- Target Accuracy: 90% (93.8% quadruple scores).
  
 Training and Validation Plots
 ----------------------------
![image](https://user-images.githubusercontent.com/10797988/218083088-501ecbf3-d464-4547-9027-da33d11c46a0.png)
![image](https://user-images.githubusercontent.com/10797988/218083153-9c4d6482-0ac3-4876-99d0-76c685b6a788.png)

Misclassified Images (20)
--------------------
![image](https://user-images.githubusercontent.com/10797988/218083378-b708ebfe-cfde-4270-83e0-cd74de5f0444.png)

GradCan Output for Misclassified Images (10)
------------------------------------
Least Number of Pixels for Grad Cam = 8

![image](https://user-images.githubusercontent.com/10797988/218083551-a9799d93-956a-49df-8402-53b5be7b257c.png)

