This code demonstrates how to build and evaluate a Convolutional Neural Network (CNN) to classify handwritten digits using the MNIST dataset. The first step is importing the necessary libraries. 
These include Keras libraries for working with the dataset, creating the CNN, and handling the preprocessing and evaluation steps. Specifically, we use the `mnist` dataset from Keras, the `to_categorical` 
function for one-hot encoding labels, and layers like `Conv2D`, `MaxPooling2D`, and `Dense` to define the architecture of the CNN.

![image](https://github.com/user-attachments/assets/75a734cb-2316-4782-81e8-ed9edcb34876)

Next, the MNIST dataset is loaded and split into training and test sets. The training set is used to teach the model, and the test set is used to evaluate its performance on unseen data. 
Before feeding the data into the model, we preprocess it by normalizing the pixel values, scaling them to a range between 0 and 1. This helps the model learn more efficiently and converge faster during training. 
We also convert the labels into one-hot encoding, where each label (digit 0-9) is represented as a vector of binary values, which is essential for categorical classification tasks.

![image](https://github.com/user-attachments/assets/36463502-5fd0-498e-94f5-afd8dfce0f1c)

With the data preprocessed, the CNN architecture is defined. The model starts with a convolutional layer (`Conv2D`) with 32 filters, each 3x3 in size, followed by a max-pooling layer (`MaxPooling2D`) that reduces 
the spatial dimensions of the feature maps. The second convolutional layer has 64 filters to capture more complex patterns, and another max-pooling layer follows. Afterward, the model flattens the 2D feature 
maps into a 1D vector, which is then passed through a dense (fully connected) layer with 128 neurons. Finally, the output layer consists of 10 neurons, one for each possible digit, with a softmax activation 
function to generate probabilities for each class.

![image](https://github.com/user-attachments/assets/21a2b55e-a84a-4367-b22e-820aa6066d86)

After defining the architecture, the model is compiled with the Adam optimizer, which adjusts the learning rate dynamically during training. The loss function used is categorical crossentropy, which is ideal for 
multi-class classification problems, and the accuracy metric tracks how well the model classifies the digits correctly.

The model is then trained using the training data for 10 epochs with a batch size of 128. Training in batches helps improve computational efficiency, and using multiple epochs allows the model to refine its 
predictions over time. After training, the model is evaluated on the test set, and we obtain the test loss and accuracy. These metrics give us an idea of how well the model generalizes to new, unseen data.

![image](https://github.com/user-attachments/assets/01b7af9f-d760-4d88-8c44-8d1076173a50)

Once the model is trained, predictions are made on the test set. The model outputs probabilities for each class, and we convert these into the actual predicted class labels by selecting the highest probability. 
To assess the model’s performance in more detail, a confusion matrix is generated. This matrix shows how many times the model correctly or incorrectly classified each digit, providing a more granular view of its 
performance. We visualize the confusion matrix as a heatmap, which makes it easier to spot where the model might be confusing certain digits.

In summary, this code walks through the entire process of building, training, and evaluating a CNN for digit classification. It demonstrates how a CNN can be used to automatically learn features from images, 
make predictions, and be evaluated based on how accurately it classifies digits. The use of a confusion matrix provides further insight into where the model performs well and where there is room for improvement.

![image](https://github.com/user-attachments/assets/d630d5fa-b498-4081-8351-c4d8fc659c78)
