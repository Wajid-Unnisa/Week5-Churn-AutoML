# Week 5: Customer Churn Prediction with AutoML

**Author:** Wajid Unnisa  
**Course:** MSDS 600 – Introduction to Data Science

## Project Overview

This assignment uses customer churn data prepared in Week 2 to compare machine learning models with PyCaret and H2O AutoML. The project includes model evaluation, a saved PyCaret model, and a Python module that predicts churn probabilities for new customers.

## Files

| File | Purpose |
|---|---|
| Wajid_Unnisa_Week5_AutoML.ipynb | Main notebook with analysis, results, reflection, and AI use disclosure |
| churn_predictor.py | Prediction functions, data preparation, and ChurnPredictor class |
| churn_model.pkl | Saved PyCaret model and preprocessing pipeline |
| churn_data_cleaned.csv | Cleaned data used for model training and evaluation |
| new_churn_data.csv | Prepared new-customer data for testing |
| new_unmodified_churn_data.csv | Unmodified new-customer data for testing preprocessing |

## Model Results

Models were selected using cross-validation AUC and evaluated on held-out test data.

| Tool | Selected Model | Test AUC |
|---|---|---|
| PyCaret | Gradient Boosting Classifier | 0.8425 |
| H2O AutoML | Stacked Ensemble | 0.8431 |

The AUC difference was small and does not establish a clear practical advantage for either model. The PyCaret model achieved approximately 49.55% recall at the default classification threshold, so it missed about half of the actual churners in the test data. Further validation of the decision threshold would be useful before applying the model to a retention campaign.

## Running the Prediction Script

The PyCaret work used Python 3.10 and PyCaret 3.3.2, with pandas and NumPy. Use a compatible environment to load the saved model. The H2O notebook section also requires H2O and Java.

Keep the Python script, saved model, and CSV files in the same folder. Open a terminal in that folder with the appropriate Python environment activated, then run:

    python churn_predictor.py

When prompted for the CSV filename, enter either:

    new_churn_data.csv

or:

    new_unmodified_churn_data.csv

The script prints a churn probability and predicted class for each customer. A predicted class of 1 indicates churn, and 0 indicates no churn, using a threshold of 0.5.

The notebook also demonstrates churn probability percentiles relative to the training predictions.

## Input Data Note

The two supplied new-customer files contain different contract and payment-method values for customer 6348-TACGU under the mappings used in the script. This customer therefore receives different predicted churn probabilities: approximately 0.3176 for the prepared file and 0.0733 for the unmodified file.

Both supplied files were retained unchanged, and their results are reported separately in the notebook.

## AI Use Disclosure

ChatGPT was used to help understand the assignment, develop and troubleshoot code, review results, improve explanations, and provide GitHub guidance. I ran the code in my own Jupyter environment and reviewed the suggestions before including them.

The detailed AI use note and conversation links are included in the notebook.
