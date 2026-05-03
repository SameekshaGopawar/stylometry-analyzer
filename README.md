# Stylometry Analyzer

A machine learning pipeline to identify authors based on linguistic style and stylistic features.

## Description
This project implements a hybrid feature extraction method for author identification. It combines TF-IDF vectorization with specific stylistic markers including sentence length, punctuation density, and stop-word frequency. The system evaluates the performance of three different classifiers: Random Forest, Support Vector Machines (SVM), and Multinomial Naive Bayes.

## Features
- Custom stylistic feature extraction
- TF-IDF word importance analysis
- Model comparison and performance metrics
- Support for multiple authors

## Requirements
- numpy
- pandas
- scikit-learn

## Performance & Results

### Model Comparison
The models were trained to distinguish between the styles of **Charles Dickens** and **Ernest Hemingway**.

| Model | Accuracy | Precision | Recall | F1-Score |
| :--- | :--- | :--- | :--- | :--- |
| **Random Forest** | 100% | 1.00 | 1.00 | 1.00 |
| **SVM** | 100% | 1.00 | 1.00 | 1.00 |
| **Naive Bayes** | 50% | 0.25 | 0.50 | 0.33 |

**Best Model:** Random Forest (F1-Score: 1.0000)

### Sample Predictions
- **Test Case 1:** "The room was quiet. Very quiet. She entered slowly..." 
  - **Result:** Hemingway (61% Confidence)
- **Test Case 2:** "The magnificent parlor, bedecked with exquisite furnishings..."
  - **Result:** Dickens (62% Confidence)
