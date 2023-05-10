In Task 1A, the goal was to train and test two different neural networks on the MINST dataset. The first part of 1A was to do this on a Fully Connected Neural Network, and the second part was to do the same thing on a Convolutional Neural Network. 

In Task 1B, i had to prove i could monitor the loss function, accuracy and Confusion matrix, which i could do by providing the graphs showing how the accuracy would change thorughout the epochs.

In Task 1C, i had to employ a pre-trained network and run the MINST dataset through it, then i had to show it would give me an accuracy of over 98.7%.

In Task 2A part 1, i had to show i could seperate 12 images and 12 labels from the rest of the dataset, and plot the bounding boxes on the 12 images using the labels, then to plot these 12 images in a 3x4 subplot. 

In Task 2A part2, i had to pre-process the data set and split it up into training, testing and validating datasets in preparation for the next task.

In Task 2B part 1, i had to use a YOLO pre-trained network to draw bounding boxes around the desired objects in my images, which was drones and be able to classify them as drones. I chose YOLOv5 because it was well documented. 
To achieve ths, i had to change the COCO128.yaml file, which stated how many different objects it had to classify as well as what they were. The training data was provided for training with the bounding boxes provided, so it would know how to draw the bounding boxes on the test and validating data sets. 
