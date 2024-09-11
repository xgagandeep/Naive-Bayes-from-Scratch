

# Naive Bayes from Scratch

**Repository:** [xgagandeep/Naive-Bayes-from-Scratch](https://github.com/xgagandeep/Naive-Bayes-from-Scratch)  
**Date:** 2020  
**Language:** Python  
**Libraries:** NumPy, pandas, scikit-learn

## Description

This project implements the Naive Bayes classification algorithm from scratch. The notebook demonstrates how to preprocess data, split it into training and testing sets, and build a Naive Bayes classifier to make predictions and evaluate its performance.

## Features

- **Data Preprocessing:** Reads and encodes categorical data using `LabelEncoder`.
- **Model Training:** Implements a Naive Bayes classifier with methods to calculate prior probabilities, conditional probabilities, and predictions.
- **Model Evaluation:** Evaluates the model's accuracy on a test dataset.

## Files

- **`Naive bayes from scratch.ipynb`:** Jupyter Notebook containing the implementation of the Naive Bayes classifier.

## Installation

To run this project, you need Python and the required libraries installed. Follow these steps:

1. **Clone the repository:**

   ```bash
   git clone https://github.com/xgagandeep/Naive-Bayes-from-Scratch.git
   ```

2. **Navigate to the project directory:**

   ```bash
   cd Naive-Bayes-from-Scratch
   ```

3. **Install the required libraries:**

   ```bash
   pip install numpy pandas scikit-learn
   ```

4. **Download the dataset:**

   This project uses the Mushroom Classification dataset. Ensure you have the dataset (`mushrooms.csv`) available in the directory specified in the notebook or update the path accordingly.

5. **Run the Jupyter Notebook:**

   ```bash
   jupyter notebook Naive\ bayes\ from\ scratch.ipynb
   ```

## Usage

1. **Data Preprocessing:**
   - Load the dataset and encode categorical features using `LabelEncoder`.

2. **Model Training:**
   - Implement the Naive Bayes classifier functions:
     - `prior_prob(x_train, y_train, label)`: Computes the prior probability of a class label.
     - `cond_prob(x_train, y_train, feature_col, feature_value, label)`: Computes the conditional probability of a feature given a class label.
     - `predict(x_train, y_train, xtest)`: Predicts the class label for a given test sample.

3. **Model Evaluation:**
   - Use the `score(x_train, y_train, x_test, y_test)` function to evaluate the classifier's accuracy on the test dataset.

4. **Testing:**
   - Test the classifier with sample data and print the predicted and actual labels.

## Example

The notebook demonstrates the full workflow of training and evaluating a Naive Bayes classifier. It loads a dataset of mushrooms, preprocesses the data, trains the classifier, and evaluates its performance by predicting on test samples.

## Contribution

Feel free to contribute to this project by submitting issues or pull requests. For any questions or feedback, please open an issue on the GitHub repository.
