# Credit Card Fraud Detection with Autoencoders

This project implements an autoencoder-based anomaly detection system for identifying fraudulent credit card transactions. It uses a deep learning model trained on normal transactions to detect anomalies that might indicate fraud.

## Project Structure
```
.
├── models/
│   └── anomaly_detector.weights.h5
├── src/
│   ├── train_model.py
│   └── test_anomaly_detector.py
├── Dockerfile
├── requirements.txt
└── README.md
```

## Features

- Autoencoder-based anomaly detection
- Support for both single and batch transaction processing
- Real-time inference capabilities
- Docker support for easy deployment
- MLflow integration for experiment tracking

## Prerequisites

- Python 3.8+
- TensorFlow 2.x
- Docker (optional)

## Installation

### Local Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/credit-card-fraud-detection.git
cd credit-card-fraud-detection
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

### Docker Installation

1. Build the Docker image:
```bash
docker build -t fraud-detection .
```

2. Run the container:
```bash
docker run -p 8000:8000 fraud-detection
```

## Usage

### Testing Single Transaction

```python
from test_anomaly_detector import detect_anomalies

transaction = {
    'distance_from_home': 57.877857,
    'distance_from_last_transaction': 0.311180,
    'ratio_to_median_purchase_price': 1.945940,
    'repeat_retailer': 1,
    'used_chip': 1,
    'used_pin_number': 0,
    'online_order': 0
}

results = detect_anomalies(transaction, "models/anomaly_detector.weights.h5")
print(f"Is fraudulent: {results['is_fraud'][0]}")
print(f"Reconstruction error: {results['reconstruction_error'][0]}")
```

### Testing Multiple Transactions

```python
transactions = [
    {
        'distance_from_home': 57.877857,
        'distance_from_last_transaction': 0.311180,
        'ratio_to_median_purchase_price': 1.945940,
        'repeat_retailer': 1,
        'used_chip': 1,
        'used_pin_number': 0,
        'online_order': 0
    },
    {
        'distance_from_home': 250.0,
        'distance_from_last_transaction': 200.0,
        'ratio_to_median_purchase_price': 5.0,
        'repeat_retailer': 0,
        'used_chip': 0,
        'used_pin_number': 0,
        'online_order': 1
    }
]

results = detect_anomalies(transactions, "models/anomaly_detector.weights.h5")
```

## Input Features

The model expects the following features for each transaction:

- `distance_from_home`: Distance from home location (float)
- `distance_from_last_transaction`: Distance from last transaction location (float)
- `ratio_to_median_purchase_price`: Ratio to median purchase price (float)
- `repeat_retailer`: Whether the retailer was visited before (0 or 1)
- `used_chip`: Whether chip was used (0 or 1)
- `used_pin_number`: Whether PIN was used (0 or 1)
- `online_order`: Whether it was an online order (0 or 1)

## Model Training

To retrain the model with your own data:

1. Prepare your dataset in the same format as described above
2. Update the training parameters in `train_model.py` if needed
3. Run the training script:
```bash
python train_model.py
```

## MLflow Tracking

The training process includes MLflow tracking. To view the experiments:

1. Start the MLflow UI:
```bash
mlflow ui
```

2. Open your browser and navigate to `http://localhost:5000`

## License

This project is licensed under the MIT License - see the LICENSE file for details.
