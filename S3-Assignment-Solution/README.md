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
- Refer to this code: COLABLINK (https://colab.research.google.com/drive/1uJZvJdi5VprOQHROtJIHy0mnY2afjNlx#scrollTo=h_Cx9q2QFgM7)
- WRITE IT AGAIN SUCH THAT IT ACHIEVES
- 99.4% validation accuracy
- Less than 20k Parameters
- You can use anything from above you want. 
- Less than 20 Epochs
- Have used BN, Dropout, a Fully connected layer, have used GAP. 
