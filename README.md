# Object Classification, Detection, and Segmentation

This repository contains a collection of projects focusing on various aspects of computer vision, including object classification, detection, and segmentation. Each folder represents a unique approach or application, demonstrating the versatility and depth of computer vision techniques.

## Projects Overview
The goal of this project is to implement computer vision methods on supertuxkart game. we used object classification, detection, segmentation, autonomous driving, and autonomous multi agent match in ice hockey game where the opponenets are AI agents.

### 1. Object Classification Linear Classifier
- An implementation of a linear classifier for object classification.

The following figure shows the classes that have been used.

![Visualization](/images/viz.png)

### 2. Object Classification CNN
- A Convolutional Neural Network (CNN) approach for object classification.


### 3. Object Classification Fully Convolutional
- A fully convolutional network model for advanced object classification tasks.


### 4. Point-based Object Detection and Segmentation
- Point-based algorithms for object detection and segmentation.

This picture shows the results of a deep learning that uses the segmentation results and find the object center from segmentation model that has been developed.
  
![Heat](/images/heat.png)
  
This figure shows the results of the trained object detection model.
  
![Box](/images/box.png)
  
### 5. Vision-based Driving CNN
- A CNN model tailored for vision-based autonomous driving applications.

The following figure shows the correct angle of the kart using blue circle, which is basically the best direction that kart has to move toward.
  
![Controller](/images/controller.png)
  
The following figure shows the correct angle using deep learning of the kart using red circle, which is basically the best direction that kart has to move toward based on our prediction model.
  
![Data](/images/data.png)

### 6. Image-based Agent Ice Hockey
- An image-based AI agent for playing ice hockey, utilizing vision algorithms.

The goal of this game is to find the puck and hit it in a way that the puck goes into the opponent goal. We purely used images as input so we trained a multi class object detection model that can find all the objects in the images.

We had two cars and the opponent team also had two cars.

This figure shows our overall method. 

![method](/images/2.png)

The following figures and videos shows the resutls of our object detection and segmentation models.

![Picture2](/images/Picture2.png)
  
![Mask Output 1](/images/mask_output1.png)
  
![Mask Output 2](/images/mask_output2.png)

![5](https://github.com/noghabaei/Object-Classification-Detection-and-Segmentation/assets/15921300/29ba3260-66e0-40d2-bddd-b58ad54645d4)

Here you can see a top down view of the match. 

![6](https://github.com/noghabaei/Object-Classification-Detection-and-Segmentation/assets/15921300/acbb939e-9e29-448d-a802-fe9c9424950b)

The following videos show how agent performed in various conditions with different trained models and approaches. 

![7](https://github.com/noghabaei/Object-Classification-Detection-and-Segmentation/assets/15921300/a31b341d-bee2-4840-9145-9c4a633bd25b)

Please note that this is a very open ended project and solving this in every possible condition is almost impossible and the performance of the agent depends on various factors, therefore sometimes agents might perform sub optimal.
  
![4](https://github.com/noghabaei/Object-Classification-Detection-and-Segmentation/assets/15921300/7e0a3a80-f8cc-46f0-9db6-9c0ba11c1ca8)



## Getting Started

Clone this repository to get started with these projects. Each folder contains individual README files with specific setup instructions and requirements.

```bash
git clone https://github.com/noghabaei/Object-Classification-Detection-and-Segmentation.git
