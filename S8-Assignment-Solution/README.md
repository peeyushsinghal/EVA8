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
 Best validation accuracy = 87.20%
![image](https://user-images.githubusercontent.com/10797988/219923128-6db2b08a-5289-47be-8423-3334c93c72a5.png)
![image](https://user-images.githubusercontent.com/10797988/219923132-864438f0-ecc2-4e26-96c0-979d376f7268.png)


Misclassified Images (20)
--------------------
![image](https://user-images.githubusercontent.com/10797988/219923141-baf26706-f296-45b7-b8a6-9d9747947235.png)

GradCam Output for Misclassified Images (10)
------------------------------------

![image](https://user-images.githubusercontent.com/10797988/219923146-a2a92201-9364-4c1d-be86-ec120dd75654.png)



