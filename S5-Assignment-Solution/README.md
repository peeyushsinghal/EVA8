Objective
-----------

- Write a single model.py file that includes GN/LN/BN and takes an argument to decide which normalization to include
- Write a single notebook file to run all the 3 models above for 20 epochs each
- Create these graphs:
  - Graph 1: Test/Validation Loss for all 3 models together
  - Graph 2: Test/Validation Accuracy for 3 models together
  - graphs must have proper annotation
- Find 10 misclassified images for each of the 3 models, and show them as a 5x2 image matrix in 3 separately annotated images. 

Implementation details - Batch Normalization (BN), Layer Normalization (LN) and Group Normalization (GN)
-------------------
* Previous assignment code was taken as base
* The Neural Network Module was changed to take in a parameter in constructor to understand BN, LN, GN
* Based on the param, the corresponding layers were used.
* BN : It is done using the channel as the attribute, futher L1 regularlization was added only for BN (lambda = 0.001)
* LN : It is done using the channel and dimensions of image
* GN : It is done using number of groups and channel as the attribute

Observations
-------------
* The accuracy figures are below than before.

Validation / Test accuracy and loss
--------------------------------------



Misclassification of Images
--------------------------------------


