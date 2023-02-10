Train ResNet18 on Cifar10 for 20 Epochs. The assignment must:
pull your Github code to google colab (don't copy-paste code)
prove that you are following the above structure
that the code in your google collab notebook is NOTHING.. barely anything. There should not be any function or class that you can define in your Google Colab Notebook. Everything must be imported from all of your other files
your colab file must:
train resnet18 for 20 epochs on the CIFAR10 dataset
show loss curves for test and train datasets
show a gallery of 10 misclassified images
show gradcam output on 10 misclassified images. Remember if you are applying GradCAM on a channel that is greater than 5px
Once done, upload the code to GitHub, and share the code. This readme must link to the main repo so we can read your file structure. 
Train for 20 epochs
Get 10 misclassified images
Get 10 GradCam outputs on any misclassified images (remember that you MUST use the library we discussed in the class)
Apply these transforms while training:
RandomCrop(32, padding=4)
CutOut(16x16)
