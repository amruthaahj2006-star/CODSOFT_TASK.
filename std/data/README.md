# Dataset Directory

This directory stores the Credit Card Fraud Detection dataset.

## Expected Dataset Location

Place the dataset file here:
```
data/creditcard.csv
```

## Dataset Overview

The dataset contains transactions made by credit cards in September 2013 by European cardholders.
- **Source**: [Kaggle Credit Card Fraud Detection Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- **Timeframe**: 284,807 transactions over two days.
- **Fraud Rate**: Highly unbalanced (~492 frauds out of 284,807 transactions, approximately 0.172%).

### Features:
- `Time`: Seconds elapsed between each transaction and the first transaction in the dataset.
- `V1` to `V28`: Numerical input variables resulting from a PCA transformation (to protect customer privacy and sensitive identity).
- `Amount`: Transaction amount in USD/EUR.
- `Class`: Response variable / Target feature:
  - `0`: Legitimate Transaction
  - `1`: Fraudulent Transaction

## How to Download:

1. Visit [Kaggle Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud).
2. Download `creditcard.csv.zip`.
3. Unzip the file and place `creditcard.csv` directly into this `data/` folder:
   ```
   data/creditcard.csv
   ```
4. If you wish to test the pipeline immediately before downloading the full 150MB file, run:
   ```bash
   python data/download_or_sample.py
   ```
   This generates a realistic sample dataset with matching statistical properties so the pipeline can execute end-to-end without errors.
