# Geometric_shape_classification
- In this project a CNN model is designed using Pytorch.
- The Neural Network will decide which one of the 9 classes: Circle, Square, Octagon, Heptagon, Nonagon, Star, Hexagon, Pentagon, Triangle does the input image belong to.
- The dataset contains 90,000 images, where each image has size 200x200.
- There are 10,000 images per class.

# Steps for running the code
- Save all the files and the dataset in the same directory.

- Execute the Model.py file, it will save the model.
- Execute the Prediction.py file, it will use the saved model to make predictions on different dataset, that dataset should be in the same directory too.

Following graph shows the training and testing loss Vs epoch
<div><img width="243" alt="lossVSepoch" src="https://user-images.githubusercontent.com/43211343/159060687-7e9c619c-5e72-4fe2-a82b-e99f31ae4b07.PNG"></div>

Following graph shows the training and testing accuracy Vs epoch
<div><img width="256" alt="accuracyVSepoch" src="https://user-images.githubusercontent.com/43211343/159060782-65ad6476-b0f7-4bfc-bfbb-e9cd76a77f7b.PNG"><div>
