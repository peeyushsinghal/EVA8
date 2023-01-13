S3 Assignment - 2 parts

PART 1
------
Screenshot of excel file
-----------------------
![image](https://user-images.githubusercontent.com/10797988/212402042-d6e219ae-abf7-4eb3-b370-4a3a5ade6f79.png)

Steps
--------
Loop 1, 2, 3 for n times
1. Forward Pass : Understanding total loss given the current weights
		
- h1=w1 * i1 + w2 * i2			
- h2=w3 * i1 + w4 * i2			
- a_h1 = σ(h1) =1/(1+exp(-h1))			
- a_h2 = σ(h2) =1/(1+exp(-h2))			
- o1 = w5 * a_h1 + w6 * a_h2 			
- o2 = w7 * a_h1 + w8 * a_h2 			
- a_o1 = σ(o1) =1/(1+exp(-o1))			
- a_o2 = σ(o2) =1/(1+exp(-o2))			
- E1 = 0.5* (t1-a_o1)^2			
- E2 = 0.5* (t2-a_o2)^2			
- E_t = E1 + E2			

2. Finding partial derivative of the total loss with respect to weights
- 𝜕E_t/𝜕w1 = [(a_o1-t1) * a_o1*(1-a_o1)*w5+(a_o2-t2) * a_o2*(1-a_o2)*w7]*[a_h1*(1-a_h1)]*[i1]
- 𝜕E_t/𝜕w2 = [(a_o1-t1) * a_o1*(1-a_o1)*w5+(a_o2-t2) * a_o2*(1-a_o2)*w7]*[a_h1*(1-a_h1)]*[i2]
- 𝜕E_t/𝜕w3 = [(a_o1-t1) * a_o1*(1-a_o1)*w6+(a_o2-t2) * a_o2*(1-a_o2)*w8]*[a_h2*(1-a_h2)]*[i1]
- 𝜕E_t/𝜕w4 = [(a_o1-t1) * a_o1*(1-a_o1)*w6+(a_o2-t2) * a_o2*(1-a_o2)*w8]*[a_h2*(1-a_h2)]*[i2]
- 𝜕E_t/𝜕w5 =(a_o1-t1) *a_o1*(1-a_o1)*a_h1
- 𝜕E_t/𝜕w6 =(a_o1-t1) *a_o1*(1-a_o1)*a_h2
- 𝜕E_t/𝜕w7 =(a_o2-t2) *a_o2*(1-a_o2)*a_h1
- 𝜕E_t/𝜕w8 =(a_o2-t2) *a_o2*(1-a_o2)*a_h2

3. Updating the weights 
-       wi = wi - η * (𝜕E_t/𝜕wi) , where i = 1,2,3,4,5,6,7,8


Screenshots of Learning Rate changes
-----------------
Learning Rate = 0.1
![image](https://user-images.githubusercontent.com/10797988/212402229-eedd7164-b65a-4a37-bc8b-c1f1953b68c9.png)

Learning Rate = 0.2
![image](https://user-images.githubusercontent.com/10797988/212402259-695bf291-1cf0-487b-b196-33fd2516c947.png)


Learning Rate = 0.5
![image](https://user-images.githubusercontent.com/10797988/212402274-94a8b91e-8978-4ba0-8610-70ed858e944c.png)


Learning Rate = 0.8
![image](https://user-images.githubusercontent.com/10797988/212402292-8f592a45-3c1a-48eb-bbd2-8a83fc36fa2d.png)

Learning Rate = 1.0
![image](https://user-images.githubusercontent.com/10797988/212402308-78b4c55e-fe12-4d7c-836d-ea14a75b0712.png)

Learning Rate = 2.0
![image](https://user-images.githubusercontent.com/10797988/212402318-a0b5226a-a242-467f-99ce-da65e4ad412e.png)



PART 2
------
ASK
- Refer to this code: COLABLINK (https://colab.research.google.com/drive/1uJZvJdi5VprOQHROtJIHy0mnY2afjNlx#scrollTo=h_Cx9q2QFgM7)
- WRITE IT AGAIN SUCH THAT IT ACHIEVES
- 99.4% validation accuracy
- Less than 20k Parameters
- You can use anything from above you want. 
- Less than 20 Epochs
- Have used BN, Dropout, a Fully connected layer, have used GAP. 

SOLUTION
- 99.43% achieved in Epoch 15 with Total params: 19,898 (see https://github.com/peeyushsinghal/EVA8/blob/main/S3-Assignment-Solution/EVA4_Session_2.ipynb)
- Use of Conv layers (with and without padding), 
- use of BN : to make sure that the features available to the next layer is good
- use of max pooling : used twice, at - least 2 Conv layers away from output
- use of GAP : instead of FC layers
- use of 1x1 : to reduce the number of channels

Model Summary
------------
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
            Conv2d-1           [-1, 32, 26, 26]             832
       BatchNorm2d-2           [-1, 32, 26, 26]              64
            Conv2d-3           [-1, 16, 26, 26]           4,624
       BatchNorm2d-4           [-1, 16, 26, 26]              32
         MaxPool2d-5           [-1, 16, 13, 13]               0
           Dropout-6           [-1, 16, 13, 13]               0
            Conv2d-7           [-1, 32, 13, 13]           4,640
       BatchNorm2d-8           [-1, 32, 13, 13]              64
            Conv2d-9           [-1, 32, 13, 13]           9,248
      BatchNorm2d-10           [-1, 32, 13, 13]              64
        MaxPool2d-11             [-1, 32, 6, 6]               0
          Dropout-12             [-1, 32, 6, 6]               0
           Conv2d-13             [-1, 10, 6, 6]             330
        AvgPool2d-14             [-1, 10, 1, 1]               0
	
Total params: 19,898
Trainable params: 19,898
Non-trainable params: 0

Training Logs
------------
Epoch : 1
  0%|          | 0/938 [00:00<?, ?it/s]<ipython-input-50-c392a1a1881b>:109: UserWarning: Implicit dimension choice for log_softmax has been deprecated. Change the call to include dim=X as an argument.
  return F.log_softmax(x)
loss=0.09435084462165833 batch_id=937: 100%|██████████| 938/938 [00:19<00:00, 48.17it/s]

Test set: Average loss: 0.1038, Accuracy: 9694/10000 (96.94%)

Epoch : 2
loss=0.04427072033286095 batch_id=937: 100%|██████████| 938/938 [00:20<00:00, 45.83it/s]

Test set: Average loss: 0.0481, Accuracy: 9855/10000 (98.55%)

Epoch : 3
loss=0.012302487157285213 batch_id=937: 100%|██████████| 938/938 [00:19<00:00, 47.99it/s]

Test set: Average loss: 0.0363, Accuracy: 9876/10000 (98.76%)

Epoch : 4
loss=0.08877681940793991 batch_id=937: 100%|██████████| 938/938 [00:19<00:00, 48.28it/s]

Test set: Average loss: 0.0318, Accuracy: 9893/10000 (98.93%)

Epoch : 5
loss=0.03348008170723915 batch_id=937: 100%|██████████| 938/938 [00:19<00:00, 47.95it/s]

Test set: Average loss: 0.0307, Accuracy: 9903/10000 (99.03%)

Epoch : 6
loss=0.01472412794828415 batch_id=937: 100%|██████████| 938/938 [00:19<00:00, 48.33it/s]

Test set: Average loss: 0.0266, Accuracy: 9909/10000 (99.09%)

Epoch : 7
loss=0.10081133991479874 batch_id=937: 100%|██████████| 938/938 [00:19<00:00, 48.45it/s]

Test set: Average loss: 0.0242, Accuracy: 9928/10000 (99.28%)

Epoch : 8
loss=0.013235016725957394 batch_id=937: 100%|██████████| 938/938 [00:19<00:00, 48.25it/s]

Test set: Average loss: 0.0231, Accuracy: 9914/10000 (99.14%)

Epoch : 9
loss=0.021460464224219322 batch_id=937: 100%|██████████| 938/938 [00:19<00:00, 48.51it/s]

Test set: Average loss: 0.0240, Accuracy: 9919/10000 (99.19%)

Epoch : 10
loss=0.08201948553323746 batch_id=937: 100%|██████████| 938/938 [00:19<00:00, 47.20it/s]

Test set: Average loss: 0.0233, Accuracy: 9922/10000 (99.22%)

Epoch : 11
loss=0.022448187693953514 batch_id=937: 100%|██████████| 938/938 [00:19<00:00, 48.95it/s]

Test set: Average loss: 0.0241, Accuracy: 9914/10000 (99.14%)

Epoch : 12
loss=0.19182398915290833 batch_id=937: 100%|██████████| 938/938 [00:19<00:00, 49.07it/s]

Test set: Average loss: 0.0215, Accuracy: 9929/10000 (99.29%)

Epoch : 13
loss=0.09115644544363022 batch_id=937: 100%|██████████| 938/938 [00:19<00:00, 49.29it/s]

Test set: Average loss: 0.0268, Accuracy: 9911/10000 (99.11%)

Epoch : 14
loss=0.0069293309934437275 batch_id=937: 100%|██████████| 938/938 [00:19<00:00, 48.59it/s]

Test set: Average loss: 0.0185, Accuracy: 9932/10000 (99.32%)

Epoch : 15
loss=0.0031304245349019766 batch_id=937: 100%|██████████| 938/938 [00:19<00:00, 48.50it/s]

Test set: Average loss: 0.0182, **Accuracy: 9943/10000 (99.43%)**

Epoch : 16
loss=0.005510952789336443 batch_id=937: 100%|██████████| 938/938 [00:19<00:00, 48.55it/s]

Test set: Average loss: 0.0172, **Accuracy: 9941/10000 (99.41%)**

Epoch : 17
loss=0.0024028862826526165 batch_id=937: 100%|██████████| 938/938 [00:19<00:00, 49.26it/s]

Test set: Average loss: 0.0190, Accuracy: 9936/10000 (99.36%)

Epoch : 18
loss=0.03194262832403183 batch_id=937: 100%|██████████| 938/938 [00:19<00:00, 47.77it/s]

Test set: Average loss: 0.0186, Accuracy: 9937/10000 (99.37%)

Epoch : 19
loss=0.0006877807900309563 batch_id=937: 100%|██████████| 938/938 [00:19<00:00, 48.23it/s]

Test set: Average loss: 0.0180, Accuracy: 9939/10000 (99.39%)

Epoch : 20
loss=0.0015419232659041882 batch_id=937: 100%|██████████| 938/938 [00:19<00:00, 49.22it/s]

Test set: Average loss: 0.0186, Accuracy: 9934/10000 (99.34%)
