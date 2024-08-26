# Project Name: *Build a Machine Learning Workflow for Scones Unlimited on AWS Sagemaker*

|                    |                                                                                               |
|--------------------|-----------------------------------------------------------------------------------------------|
| Libraries Used     | - Boto3: AWS SDK for Python, used to interact with AWS services like S3 and SageMaker.<BR>- Pandas: Data manipulation and analysis library, useful for handling datasets.<BR>- NumPy: Library for numerical operations, often used for array manipulations.<BR>- Matplotlib/Seaborn: Libraries for data visualization, helpful for plotting and analyzing data.<BR>- Scikit-learn: Machine learning library for preprocessing and model evaluation (if needed).<BR>- PyTorch: Deep learning frameworks that may be used for building and training models.|
| Dataset            | [CIFAR-100](https://www.cs.toronto.edu/~kriz/cifar.html) |
| Tools Used         | - AWS SageMaker: The main platform for building, training, and deploying machine learning models.<BR>- AWS S3: Storage service used to store datasets and model artifacts.<BR>- AWS Lambda: Serverless compute service used to run code in response to events, such as image classification.<BR>- AWS Step Functions: Service for orchestrating serverless workflows, chaining Lambda functions together.<BR>- Jupyter Notebook: An interactive environment for writing and running code, used for developing the project.|


## Project Overview

<img src="img/badge.png" width="480" align="right">

The goal of this project is to build a complete machine learning workflow with focus on image classification model that can automatically detect which kind of vehicle delivery drivers have, in order to route them to the correct loading bay and orders. Assigning delivery professionals who have a bicycle to nearby orders and giving motorcyclists orders that are farther can help Scones Unlimited optimize their operations. 
Project uses mainly **Amazon SageMaker**. It involves setting up a SageMaker environment, performing data extraction, transformation, and loading (ETL), training an image classification model, deploying the model, and creating a serverless architecture using **AWS Lambda** and **Step Functions**.


## Key Features
- **SageMaker Studio Setup**: Configured a SageMaker Studio environment with the necessary permissions and settings.
- **Data ETL**: Loaded and transformed image data from a specified URL, saving it to an S3 bucket with appropriate labeling.
- **Model Training**: Trained an image classification model using SageMaker's built-in algorithms, with hyperparameters optimized for performance.
- **Model Deployment**: Deployed the trained model to create a real-time inference endpoint, allowing for predictions on new images.
- **Serverless Architecture**: Developed three AWS Lambda functions to handle image data processing, classification, and filtering of low-confidence predictions.
- **Step Functions Integration**: Orchestrated the Lambda functions using AWS Step Functions to create a seamless workflow for image classification.
- **Model Monitoring**: Implemented monitoring to track model performance and visualize results, ensuring the model remains effective over time.

## Learning Outcomes
- Hands-on experience with AWS SageMaker and its capabilities for building machine learning models.
- Understand the ETL process and how to prepare data for machine learning.
- Learn how to train and deploy machine learning models in a cloud environment.
- Develop skills in creating serverless applications using AWS Lambda and Step Functions.
- Acquire knowledge in monitoring machine learning models to ensure ongoing performance and reliability.
