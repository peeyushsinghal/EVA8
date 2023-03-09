## Assignment 9
-----------
## Build the following network:

1. That takes a CIFAR10 image (32x32x3)
2. Add 3 Convolutions to arrive at AxAx48 dimensions (e.g. 32x32x3 | 3x3x3x16 >> 3x3x16x32 >> 3x3x32x48)
3. Apply GAP and get 1x1x48, call this X
4. Create a block called ULTIMUS that:
  - Creates 3 FC layers called K, Q and V such that:
    - X\*K = 48*48x8 > 8
    - X\*Q = 48*48x8 > 8 
    - X\*V = 48*48x8 > 8 
  - then create AM = SoftMax(QTK)/(8^0.5) = 8*8 = 8
  - then Z = V\*AM = 8*8 > 8
  - then another FC layer called Out that:
    - Z\*Out = 8*8x48 > 48
5. Repeat this Ultimus block 4 times
6. Then add final FC layer that converts 48 to 10 and sends it to the loss function.
7. Model would look like this C>C>C>U>U>U>U>FFC>Loss
8. Train the model for 24 epochs using the OCP that I wrote in class. Use ADAM as an optimizer. 
9. Submit the link and answer the questions on the assignment page:
  - Share the link to the main repo (must have Assignment 7/8/9 model7/8/9.py files (or similarly named))
  - Share the code of model9.py
  - Copy and paste the Training Log
  - Copy and paste the training and validation loss chart
  
Data Set
------------

We have used the CIFAR-10 dataset

Model File
---------
Model file is kept at https://github.com/peeyushsinghal/pytorch-models-utils/blob/main/models/transformer_ultimus.py

Training Logs
-------------
```
EPOCH: 1
Loss=2.302584171295166 Batch_id=390 Accuracy=10.01: 100%|██████████| 391/391 [00:18<00:00, 21.62it/s]

Test set: Average loss: 0.0002, Accuracy: 1000/10000 (10.00%)

EPOCH: 2
Loss=2.302584171295166 Batch_id=390 Accuracy=10.00: 100%|██████████| 391/391 [00:18<00:00, 20.75it/s]

Test set: Average loss: 0.0002, Accuracy: 1000/10000 (10.00%)

EPOCH: 3
Loss=2.302584171295166 Batch_id=390 Accuracy=10.00: 100%|██████████| 391/391 [00:18<00:00, 21.27it/s]

Test set: Average loss: 0.0002, Accuracy: 1000/10000 (10.00%)

EPOCH: 4
Loss=2.302584171295166 Batch_id=390 Accuracy=10.00: 100%|██████████| 391/391 [00:18<00:00, 21.19it/s]

Test set: Average loss: 0.0002, Accuracy: 1000/10000 (10.00%)

EPOCH: 5
Loss=2.302584171295166 Batch_id=390 Accuracy=10.00: 100%|██████████| 391/391 [00:19<00:00, 20.36it/s]

Test set: Average loss: 0.0002, Accuracy: 1000/10000 (10.00%)

EPOCH: 6
Loss=2.302584171295166 Batch_id=390 Accuracy=10.00: 100%|██████████| 391/391 [00:19<00:00, 20.35it/s]

Test set: Average loss: 0.0002, Accuracy: 1000/10000 (10.00%)

EPOCH: 7
Loss=2.302584171295166 Batch_id=390 Accuracy=10.00: 100%|██████████| 391/391 [00:19<00:00, 20.01it/s]

Test set: Average loss: 0.0002, Accuracy: 1000/10000 (10.00%)

EPOCH: 8
Loss=2.302584171295166 Batch_id=390 Accuracy=10.00: 100%|██████████| 391/391 [00:18<00:00, 21.33it/s]

Test set: Average loss: 0.0002, Accuracy: 1000/10000 (10.00%)

EPOCH: 9
Loss=2.302584171295166 Batch_id=390 Accuracy=10.00: 100%|██████████| 391/391 [00:18<00:00, 21.67it/s]

Test set: Average loss: 0.0002, Accuracy: 1000/10000 (10.00%)

EPOCH: 10
Loss=2.302584171295166 Batch_id=390 Accuracy=10.00: 100%|██████████| 391/391 [00:19<00:00, 20.14it/s]

Test set: Average loss: 0.0002, Accuracy: 1000/10000 (10.00%)

EPOCH: 11
Loss=2.302584171295166 Batch_id=390 Accuracy=10.00: 100%|██████████| 391/391 [00:18<00:00, 21.38it/s]

Test set: Average loss: 0.0002, Accuracy: 1000/10000 (10.00%)

EPOCH: 12
Loss=2.302584171295166 Batch_id=390 Accuracy=10.00: 100%|██████████| 391/391 [00:19<00:00, 20.54it/s]

Test set: Average loss: 0.0002, Accuracy: 1000/10000 (10.00%)

EPOCH: 13
Loss=2.302584171295166 Batch_id=390 Accuracy=10.00: 100%|██████████| 391/391 [00:18<00:00, 21.02it/s]

Test set: Average loss: 0.0002, Accuracy: 1000/10000 (10.00%)

EPOCH: 14
Loss=2.302584171295166 Batch_id=390 Accuracy=10.00: 100%|██████████| 391/391 [00:18<00:00, 21.38it/s]

Test set: Average loss: 0.0002, Accuracy: 1000/10000 (10.00%)

EPOCH: 15
Loss=2.302584171295166 Batch_id=390 Accuracy=10.00: 100%|██████████| 391/391 [00:19<00:00, 20.01it/s]

Test set: Average loss: 0.0002, Accuracy: 1000/10000 (10.00%)

EPOCH: 16
Loss=2.302584171295166 Batch_id=390 Accuracy=10.00: 100%|██████████| 391/391 [00:18<00:00, 21.26it/s]

Test set: Average loss: 0.0002, Accuracy: 1000/10000 (10.00%)

EPOCH: 17
Loss=2.302584171295166 Batch_id=390 Accuracy=10.00: 100%|██████████| 391/391 [00:19<00:00, 20.47it/s]

Test set: Average loss: 0.0002, Accuracy: 1000/10000 (10.00%)

EPOCH: 18
Loss=2.302584171295166 Batch_id=390 Accuracy=10.00: 100%|██████████| 391/391 [00:23<00:00, 16.95it/s]

Test set: Average loss: 0.0002, Accuracy: 1000/10000 (10.00%)

EPOCH: 19
Loss=2.302584171295166 Batch_id=390 Accuracy=10.00: 100%|██████████| 391/391 [00:18<00:00, 21.06it/s]

Test set: Average loss: 0.0002, Accuracy: 1000/10000 (10.00%)

EPOCH: 20
Loss=2.302584171295166 Batch_id=390 Accuracy=10.00: 100%|██████████| 391/391 [00:19<00:00, 20.03it/s]

Test set: Average loss: 0.0002, Accuracy: 1000/10000 (10.00%)

EPOCH: 21
Loss=2.302584171295166 Batch_id=390 Accuracy=10.00: 100%|██████████| 391/391 [00:18<00:00, 21.05it/s]

Test set: Average loss: 0.0002, Accuracy: 1000/10000 (10.00%)

EPOCH: 22
Loss=2.302584171295166 Batch_id=390 Accuracy=10.00: 100%|██████████| 391/391 [00:18<00:00, 21.54it/s]

Test set: Average loss: 0.0002, Accuracy: 1000/10000 (10.00%)

EPOCH: 23
Loss=2.302584171295166 Batch_id=390 Accuracy=10.00: 100%|██████████| 391/391 [00:19<00:00, 19.76it/s]

Test set: Average loss: 0.0002, Accuracy: 1000/10000 (10.00%)

EPOCH: 24
Loss=2.302584171295166 Batch_id=390 Accuracy=10.00: 100%|██████████| 391/391 [00:18<00:00, 21.10it/s]

Test set: Average loss: 0.0002, Accuracy: 1000/10000 (10.00%)
```

Class level Accuracy
-------------------
```
Accuracy of airplane : 100 %
Accuracy of automobile :  0 %
Accuracy of  bird :  0 %
Accuracy of   cat :  0 %
Accuracy of  deer :  0 %
Accuracy of   dog :  0 %
Accuracy of  frog :  0 %
Accuracy of horse :  0 %
Accuracy of  ship :  0 %
Accuracy of truck :  0 %


```


