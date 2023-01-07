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

Inside Training epoch - 1, processing step 1 , Training loss: 0.04191651940345764 

Inside Training epoch - 1, processing step 1001 , Training loss: 0.04989407956600189 

After completion of epoch 1   Training loss: 0.005842952989041805
Testing...
Test loss: 2.970931351184845 total_correct_label: 59592 accuracy_labels: 99.32 %  total_correct_sum: 58594 accuracy_correct_sum: 97.657 % 
Inside Training epoch - 2, processing step 1 , Training loss: 0.026007680222392082 

Inside Training epoch - 2, processing step 1001 , Training loss: 0.03281652554869652 

After completion of epoch 2   Training loss: 0.03183192387223244
Testing...
Test loss: 1.9940551780164242 total_correct_label: 59738 accuracy_labels: 99.563 %  total_correct_sum: 59166 accuracy_correct_sum: 98.61 % 
Inside Training epoch - 3, processing step 1 , Training loss: 0.00639250036329031 

Inside Training epoch - 3, processing step 1001 , Training loss: 0.11242609471082687 

After completion of epoch 3   Training loss: 0.0017954612849280238
Testing...
Test loss: 1.6167227271944284 total_correct_label: 59785 accuracy_labels: 99.642 %  total_correct_sum: 59250 accuracy_correct_sum: 98.75 % 
Inside Training epoch - 4, processing step 1 , Training loss: 0.001807750784792006 

Inside Training epoch - 4, processing step 1001 , Training loss: 0.024995243176817894 

After completion of epoch 4   Training loss: 0.010514327324926853
Testing...
Test loss: 1.4462827667593956 total_correct_label: 59843 accuracy_labels: 99.738 %  total_correct_sum: 59314 accuracy_correct_sum: 98.857 % 
Inside Training epoch - 5, processing step 1 , Training loss: 0.0007904624217189848 

Inside Training epoch - 5, processing step 1001 , Training loss: 0.0009802425047382712 

After completion of epoch 5   Training loss: 0.006870337761938572
Testing...
Test loss: 1.4358101598918438 total_correct_label: 59866 accuracy_labels: 99.777 %  total_correct_sum: 59321 accuracy_correct_sum: 98.868 % 
