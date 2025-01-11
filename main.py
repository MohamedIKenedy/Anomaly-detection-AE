import pandas as pd
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.losses import mae
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

def create_model(input_shape=7, code_dim=2):
    """Recreate the autoencoder architecture"""
    input_layer = Input(shape=(input_shape,))
    x = Dense(64, activation='relu')(input_layer)
    x = Dense(16, activation='relu')(x)
    code = Dense(code_dim, activation='relu')(x)
    x = Dense(16, activation='relu')(code)
    x = Dense(64, activation='relu')(x)
    output_layer = Dense(input_shape, activation='relu')(x)
    
    return Model(input_layer, output_layer, name='anomaly')

def prepare_input_data(data):
    """Prepare input data with the same preprocessing as training"""
    numeric_features = ['distance_from_home', 'distance_from_last_transaction', 'ratio_to_median_purchase_price', 'repeat_retailer', 'used_chip', 'used_pin_number', 'online_order']
    
    # Create a new scaler - Note: In production, you should use the scaler from training
    scaler = StandardScaler()
    
    # Ensure all features are present and in the correct order
    if isinstance(data, dict):
        data = pd.DataFrame([data])
    elif isinstance(data, list):
        data = pd.DataFrame(data)
        
    # Verify all required features are present
    missing_cols = set(numeric_features) - set(data.columns)
    if missing_cols:
        raise ValueError(f"Missing required features: {missing_cols}")
    
    # Scale the features
    scaled_data = scaler.fit_transform(data[numeric_features])
    return scaled_data

def detect_anomalies(data, model_path, threshold=0.4543):
    """
    Detect anomalies in the input data using the trained autoencoder
    
    Parameters:
    data: DataFrame or dict with the required features
    model_path: Path to the saved model weights
    threshold: Reconstruction error threshold for anomaly detection
    
    Returns:
    Dictionary containing predictions and reconstruction errors
    """
    try:
        # Prepare input data
        processed_data = prepare_input_data(data)
        
        # Create and load model
        model = create_model(input_shape=processed_data.shape[1])
        model.load_weights(model_path)
        
        # Get reconstructions
        reconstructions = model.predict(processed_data, verbose=0)
        
        # Calculate reconstruction errors
        reconstruction_errors = mae(reconstructions, processed_data).numpy()
        
        # Make predictions
        predictions = reconstruction_errors > threshold
        
        # Prepare results
        results = {
            'is_fraud': predictions.tolist(),
            'reconstruction_error': reconstruction_errors.tolist(),
            'threshold': threshold
        }
        
        return results
        
    except Exception as e:
        return {'error': str(e)}

if __name__ == "__main__":
    # Example usage
    # Single transaction
    sample_transaction = {
        'distance_from_home': 57.877857,
        'distance_from_last_transaction': 0.311180,
        'ratio_to_median_purchase_price': 1.945940,
        'repeat_retailer': 1,
        'used_chip': 1,
        'used_pin_number': 0,
        'online_order': 0
    }
    
    # Multiple transactions
    sample_transactions = [
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
    
    # Detect anomalies
    print("\nTesting single transaction:")
    results = detect_anomalies(sample_transaction, "models/anomaly_detector.weights.h5")
    print(f"Is fraudulent: {results['is_fraud'][0]}")
    print(f"Reconstruction error: {results['reconstruction_error'][0]:.4f}")
    print(f"Threshold: {results['threshold']}")
    
    print("\nTesting multiple transactions:")
    results = detect_anomalies(sample_transactions, "models/anomaly_detector.weights.h5")
    for i, (is_fraud, error) in enumerate(zip(results['is_fraud'], results['reconstruction_error'])):
        print(f"\nTransaction {i+1}:")
        print(f"Is fraudulent: {is_fraud}")
        print(f"Reconstruction error: {error:.4f}")
