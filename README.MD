\# Stylometry Analyzer



A machine learning pipeline to identify authors based on linguistic style and stylistic features.



\## Description

This project implements a hybrid feature extraction method for author identification. It combines TF-IDF vectorization with specific stylistic markers including sentence length, punctuation density, and stop-word frequency. The system evaluates the performance of three different classifiers: Random Forest, Support Vector Machines (SVM), and Multinomial Naive Bayes.



\## Features

\- Custom stylistic feature extraction (5-dimensional vector)

\- TF-IDF word importance analysis

\- Model comparison and performance metrics (Accuracy, F1-Score)

\- Support for multiple authors



\## Requirements

\- numpy

\- pandas

\- scikit-learn



\## Usage

1\. Clone the repository to your local machine.

2\. Install the necessary libraries using pip.

3\. Run `python main.py` to train the models and see the predictions on the test cases.



\## Implementation Details

The analyzer scales features for SVM and Random Forest while maintaining unscaled counts for the Multinomial Naive Bayes model to ensure mathematical consistency across different classifier requirements.

