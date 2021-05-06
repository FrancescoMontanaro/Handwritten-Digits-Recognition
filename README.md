# Handwritten Digits Recognition

![AfraidCompassionateAnemoneshrimp-size_restricted](https://user-images.githubusercontent.com/56433128/117266150-3cc61600-ae55-11eb-83d8-31b265bc6ce5.gif)

This is a simple Project to recognize handwitten digits through a **Deep Learning Model**. 
The Dataset used is the *MNIST* Dataset, composed by 60.000 labeled training images and 10.000 labeled test images.

![Images-generated-by-different-GAN-architectures-trained-on-MNIST-1-and-MNIST-2-The ppm](https://user-images.githubusercontent.com/56433128/117200445-cccf7580-adeb-11eb-97ab-01ebd94c1caf.png)

The Neural Network is made of Convolutional layes, for the features extraction, and Fully Connected layers, for the final classification. In order to limit overfitting some techniques have been adopted, such as **Dropout** and **Early Stopping** (with a patience of 5 epochs over the validation loss). 

The following image shows a summary of the model:

<img width="470" alt="Schermata 2021-05-05 alle 21 55 31" src="https://user-images.githubusercontent.com/56433128/117201202-a9f19100-adec-11eb-9f48-b8412a2ba1de.png">

The final model has been trained for **30 epochs** and has reached an **Accuracy of 0.9397** on the test set.
