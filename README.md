# Telco-Synth

A synthetic telecommunications data generator with machine learning pipeline for package recommendation.

## Features

- **Synthetic Data Generation**: Creates realistic telco customer profiles and package catalogs
- **Customer Segmentation**: 10 behavioral microsegments (Young Social, Professionals, Seniors, etc.)
- **Package Catalog**: 19 telco packages across Basic, Premium, Economy, and Starter tiers
- **ML Pipeline**: Logistic regression model for package acceptance prediction
- **Scoring System**: Customer × package recommendation grid with probability scores

## Quick Start

```bash
python main.py
```

## Output Files

- `telco_training_data.csv` - Training dataset
- `telco_test_data.csv` - Test dataset  
- `telco_test_scored.csv` - Model predictions
- `telco_test_top1_reco.csv` - Top recommendations per customer

## Requirements

- pandas
- numpy
- scikit-learn
- matplotlib

## Use Cases

- Package recommendation systems
- Customer segmentation analysis
- ML model development and testing
- Telco business intelligence

