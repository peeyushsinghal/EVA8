Submission towards assignment 2.5

### Assignment ask
Write a neural network that can:
1. take 2 inputs:
1.1 an image from the MNIST dataset (say 5), and
1.2 a random number between 0 and 9, (say 7)
2and gives two outputs:
2.1 the "number" that was represented by the MNIST image (predict 5), and
2.2 the "sum" of this number with the random number and the input image to the network (predict 5 + 7 = 12)
                 
3 you can mix fully connected layers and convolution layers
4 you can use one-hot encoding to represent the random number input and the "summed" output.
1. Random number (7) can be represented as 0 0 0 0 0 0 0 1 0 0
2 Sum (13) can be represented as:
    0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0
    0b1101 (remember that 4 digits in binary can at max represent 15, so we may need to go for 5 digits. i.e. 10010

Your code MUST be:
- well documented (via readme file on GitHub and comments in the code)
- must mention the data representation
- must mention your data generation strategy (basically the class/method you are using for random number generation)
- must mention how you have combined the two inputs (basically which layer you are combining)
- must mention how you are evaluating your results 
- must mention "what" results you finally got and how did you evaluate your results
- must mention what loss function you picked and why!
- training MUST happen on the GPU
- Accuracy is not really important for the SUM


### Training logs

Inside Training epoch - 1, processing step 1 , Training loss: 2.6222634315490723
Inside Training epoch - 1, processing step 501 , Training loss: 1.1648956537246704
Inside Training epoch - 1, processing step 1001 , Training loss: 0.7958481907844543
Inside Training epoch - 1, processing step 1501 , Training loss: 0.7049594521522522
After completion of epoch 1   Training loss: 0.7129550576210022
Testing...
Test loss: 36.376513838768005 total_correct_label: 58928 accuracy_labels: 98.213 %  total_correct_sum: 34738 accuracy_correct_sum: 57.897 % 
Inside Training epoch - 2, processing step 1 , Training loss: 0.6237501502037048
Inside Training epoch - 2, processing step 501 , Training loss: 0.3853496313095093
Inside Training epoch - 2, processing step 1001 , Training loss: 0.15507294237613678
Inside Training epoch - 2, processing step 1501 , Training loss: 0.20194187760353088
After completion of epoch 2   Training loss: 0.10701191425323486
Testing...
Test loss: 7.575702026486397 total_correct_label: 59216 accuracy_labels: 98.693 %  total_correct_sum: 57215 accuracy_correct_sum: 95.358 % 
Inside Training epoch - 3, processing step 1 , Training loss: 0.05682281032204628
Inside Training epoch - 3, processing step 501 , Training loss: 0.05732327699661255
Inside Training epoch - 3, processing step 1001 , Training loss: 0.06491690874099731
Inside Training epoch - 3, processing step 1501 , Training loss: 0.012780263088643551
After completion of epoch 3   Training loss: 0.016423340886831284
Testing...
Test loss: 4.79213584959507 total_correct_label: 59246 accuracy_labels: 98.743 %  total_correct_sum: 58191 accuracy_correct_sum: 96.985 % 
Inside Training epoch - 4, processing step 1 , Training loss: 0.03952173516154289
Inside Training epoch - 4, processing step 501 , Training loss: 0.03701169043779373
Inside Training epoch - 4, processing step 1001 , Training loss: 0.022090056911110878
Inside Training epoch - 4, processing step 1501 , Training loss: 0.02581946738064289
After completion of epoch 4   Training loss: 0.2428630143404007
Testing...
Test loss: 3.3275577016174793 total_correct_label: 59633 accuracy_labels: 99.388 %  total_correct_sum: 58539 accuracy_correct_sum: 97.565 % 
Inside Training epoch - 5, processing step 1 , Training loss: 0.09834140539169312
Inside Training epoch - 5, processing step 501 , Training loss: 0.0070363543927669525
Inside Training epoch - 5, processing step 1001 , Training loss: 0.02909870631992817
Inside Training epoch - 5, processing step 1501 , Training loss: 0.0795339047908783
After completion of epoch 5   Training loss: 0.27176910638809204
Testing...
Test loss: 2.732940660789609 total_correct_label: 59654 accuracy_labels: 99.423 %  total_correct_sum: 58847 accuracy_correct_sum: 98.078 % 
### Training logs (old)

/usr/local/lib/python3.7/dist-packages/torch/nn/functional.py:718: UserWarning: Named tensors and all their associated APIs are an experimental feature and subject to change. Please do not use them for anything important until they are released as stable. (Triggered internally at  /pytorch/c10/core/TensorImpl.h:1156.)
  return torch.max_pool2d(input, kernel_size, stride, padding, dilation, ceil_mode)

After completion of epoch 1   Training loss: 0.4649723470211029
Testing...
Test loss: 34.224786818027496 total_correct_label: 58709 accuracy_labels: 97.848 %  total_correct_sum: 37697 accuracy_correct_sum: 62.828 % 

After completion of epoch 2   Training loss: 0.2918633222579956
Testing...
Test loss: 9.194906830787659 total_correct_label: 59380 accuracy_labels: 98.967 %  total_correct_sum: 56513 accuracy_correct_sum: 94.188 % 

After completion of epoch 3   Training loss: 0.059926837682724
Testing...
Test loss: 4.166031699627638 total_correct_label: 59368 accuracy_labels: 98.947 %  total_correct_sum: 58391 accuracy_correct_sum: 97.318 % 

After completion of epoch 4   Training loss: 0.06129102408885956
Testing...
Test loss: 2.3967976197600365 total_correct_label: 59704 accuracy_labels: 99.507 %  total_correct_sum: 59007 accuracy_correct_sum: 98.345 % 

After completion of epoch 5   Training loss: 0.012244242243468761
Testing...
Test loss: 1.9272407814860344 total_correct_label: 59767 accuracy_labels: 99.612 %  total_correct_sum: 59142 accuracy_correct_sum: 98.57 % 
