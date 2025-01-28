In Task 1A, the goal was to train and test two different neural networks on the MINST dataset. The first part of 1A was to do this on a Fully Connected Neural Network, and the second part was to do the same thing on a Convolutional Neural Network. 
![image](https://github.com/user-attachments/assets/b508c2ff-7acf-4919-a97d-89e0bce93c28)

I arranged my code so that all the libraries I am using are at the beginning of the code. I did so to make it easier to see if i am missing a library, or if i am importing a library i do not actually use. 
![image](https://github.com/user-attachments/assets/8640dba8-00e6-4abe-9131-588998087b3f)

Further, I access the MINST dataset, and break it into training data and test data, as well as setting its dimensions before i begin to build the model architecture.

![image](https://github.com/user-attachments/assets/7835b14f-1672-4aba-8051-03e3e5d80152)

I then set my models architecture, and used 'relu' as the activation beccause it has a faster convergence rate as well as being computationally efficient. Then i set my loss function and compiled the model.

![image](https://github.com/user-attachments/assets/2c94788e-3a28-4cff-8135-6075e832cf7c)

This workflow demonstrates the typical steps involved in training and evaluating a machine learning model, as well as making sense of its outputs.
First, the model is trained on the dataset by going through it multiple times (epochs) and processing it in smaller chunks (batches). This step is where the model learns by adjusting its internal parameters to minimize errors and improve how well it performs.
Once the training is done, the model generates predictions for a sample from the training data. These predictions come out as raw numbers, called logits, which represent how confident the model is in each possible output. To understand how well it’s doing, the loss function is used to calculate the difference between the predictions and the actual answers (ground truth). This gives a clear idea of how accurate the model is for that specific sample.
Since raw predictions (logits) aren’t very intuitive, they’re passed through a softmax function. Softmax turns those raw numbers into probabilities, which makes it much easier to interpret the results. For example, instead of seeing raw scores, you can now see the model’s confidence as percentages, like “90% confident this is class A.”
After training, the model is evaluated on a separate dataset that it hasn’t seen before (the test set). This step measures how well the model performs on new data, helping to check if it’s truly learned something useful or just memorized the training data.
Finally, to make predictions even simpler, a probability model is created. This combines the trained model with a softmax layer, so every prediction it makes comes out as probabilities by default. It’s a convenient way to work with the model without having to manually apply softmax every time.
Overall, this process takes the model from learning and adjusting itself to making predictions that are easy to understand and evaluate. It’s a great way to ensure the model is effective and its outputs are meaningful.

![image](https://github.com/user-attachments/assets/cbee2285-b426-4934-bfaa-e509f262e588)

This part of the workflow focuses on working with the trained model to make predictions, calculate losses, and review the model's performance.
First, the probability model is used to generate predictions for the first five samples of the test dataset. The probability model outputs these predictions as probabilities, making it easy to interpret the model’s confidence for each class.
Next, the original trained model (without the softmax layer) is used to predict the outcome for just the first sample in the test dataset. These raw predictions are then compared to the actual target value using the loss function. This gives the test loss, which is a measure of how far the model’s predictions are from the true answer for that specific sample.
The predictions, along with the corresponding training and test losses, are printed out. This gives a snapshot of how well the model is performing on both the training and test datasets. Finally, the previously evaluated results for the model (such as the overall loss and accuracy on the test set) are also printed to summarize its performance on unseen data.
In short, this code provides a mix of interpreting predictions, calculating specific losses, and reviewing overall test performance to get a complete picture of how the model behaves on both training and test data.
