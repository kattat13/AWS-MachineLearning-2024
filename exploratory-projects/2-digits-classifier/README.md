# Project Name: *Developing a Handwritten Digits Classifier with PyTorch Project*

|                    |                                                                                               |
|--------------------|-----------------------------------------------------------------------------------------------|
| Libraries Used     | - PyTorch: For building and training neural networks.<br>- torchvision: For dataset loading and transformations.<br>- NumPy: For numerical operations.<br>- Matplotlib: For data visualization.     |
| Dataset            | MNIST: A dataset of handwritten digits containing 60,000 training images and 10,000 test images. |
| Tools Used         | - Jupyter Notebook: For coding and experimentation.<BR>- DataLoader: For loading data in batches.<BR>- Transforms: For preprocessing images (e.g., normalization, conversion to tensor).|
| Modeling Techniques| - Neural Networks: Building a feedforward neural network or convolutional neural network (CNN).<BR>- Loss Functions: Using CrossEntropyLoss for classification tasks.<BR>- Optimizers: Utilizing optimizers like Adam or SGD to minimize loss. |
| Evaluation Metrics      | - Accuracy: To assess the performance of the model.<BR>- Confusion Matrix: To visualize the performance across different classes.            |

## Project Overview
<img src="./img/badge.png" width="480" align="right">

This project focuses on developing a neural network model to classify handwritten digits from the MNIST dataset using PyTorch. The model is trained to recognize digits from 0 to 9 based on pixel values from images.

## Key Features
- **Data Loading and Preprocessing**: Efficiently loads the MNIST dataset, converting images into tensors and applying normalization to enhance model performance.
- **Model Architecture**: Implements a neural network with at least two hidden layers and a forward method that outputs prediction probabilities for each of the ten digit classes using the softmax function.
- **Training the Model**: Defines an appropriate loss function (CrossEntropyLoss) and utilizes optimizers from `torch.optim` to minimize loss during the training process.
- **DataLoader Usage**: Creates `DataLoader` objects for both training and testing datasets, facilitating efficient data batching and shuffling.
- **Model Evaluation**: Evaluates the model's performance using the test dataset, comparing predictions against true labels to assess accuracy.
- **Hyperparameter Tuning**: Experiments with various hyperparameters to optimize model performance, achieving over 90% classification accuracy.
- **Model Persistence**: Implements the `torch.save()` function to save the trained model's weights for future use or deployment.


## Learning Outcomes
- A solid understanding of how to preprocess image data for neural networks.
- Experience in designing and implementing neural network architectures using PyTorch.
- Knowledge of various loss functions and optimizers suitable for classification tasks.
- Skills in evaluating model performance and making data-driven decisions to improve accuracy through hyperparameter tuning.
- Practical experience in saving and loading trained models for deployment.

