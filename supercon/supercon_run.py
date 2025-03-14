import torch
import torch.nn as nn
import pennylane as qml
import numpy as np
from sklearn.preprocessing import StandardScaler

def quantum_circuit(inputs, weights):
    n_qubits = 4  # number of qubits
    for i in range(n_qubits):
        qml.RX(inputs[i], wires=i)
    
    n_layers = weights.shape[0]
    for layer in range(n_layers):
        for i in range(n_qubits):
            qml.RZ(weights[layer, i, 0], wires=i)
            qml.RY(weights[layer, i, 1], wires=i)
            qml.RZ(weights[layer, i, 2], wires=i)
        for i in range(n_qubits - 1):
            qml.CNOT(wires=[i, i + 1])
    
    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

n_qubits = 4
dev = qml.device("default.qubit", wires=n_qubits)
qnode = qml.QNode(quantum_circuit, dev, interface="torch")


class HybridModel(nn.Module):
    def __init__(self, input_size, n_qubits):
        super(HybridModel, self).__init__()
        self.input_size = input_size
        self.n_qubits = n_qubits
        self.feature_projector = nn.Linear(input_size, n_qubits)
        self.n_layers = 2
        self.weights = nn.Parameter(0.01 * torch.randn(self.n_layers, n_qubits, 3))
        self.post_net = nn.Sequential(
            nn.Linear(n_qubits, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    
    def forward(self, x):
        batch_size = x.shape[0]
        x = self.feature_projector(x.float())
        
        q_out = []
        for i in range(batch_size):
            quantum_output = torch.tensor(qnode(x[i], self.weights), dtype=torch.float32)
            q_out.append(quantum_output.unsqueeze(0))
        
        q_out = torch.cat(q_out, dim=0)
        x = self.post_net(q_out)
        return x

def save_model(model, scaler_X, scaler_y, input_size, n_qubits, filename="trained_model_weights.pth"):
    """Save the model and scaling parameters"""
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'scaler_X_mean': scaler_X.mean_,
        'scaler_X_scale': scaler_X.scale_,
        'scaler_y_mean': scaler_y.mean_,
        'scaler_y_scale': scaler_y.scale_,
        'input_size': input_size,
        'n_qubits': n_qubits
    }
    torch.save(checkpoint, filename)

import os

def load_model_and_predict(input_data, weights_path=None):
    if weights_path is None:
        # Get the directory of the current file
        base_dir = os.path.dirname(os.path.abspath(__file__))
        # Construct an absolute path to the weights file
        weights_path = os.path.join(base_dir, "trained_model_weights.pth")
    # Now use weights_path to load the model...
    # load_model(weights_path)

    try:
        # Load saved model and scalers
        checkpoint = torch.load(weights_path, weights_only=False)
        
        # Get the expected input size from the checkpoint
        input_size = checkpoint['input_size']
        
        # Verify input dimensions
        if input_data.shape[1] != input_size:
            raise ValueError(f"Input data must have {input_size} features. Got {input_data.shape[1]} features instead.")
        
        # Initialize scalers with correct feature dimensions
        scaler_X = StandardScaler()
        scaler_y = StandardScaler()
        
        # Set scaler parameters
        scaler_X.mean_ = checkpoint['scaler_X_mean']
        scaler_X.scale_ = checkpoint['scaler_X_scale']
        scaler_y.mean_ = checkpoint['scaler_y_mean']
        scaler_y.scale_ = checkpoint['scaler_y_scale']
        
        # Initialize model
        model = HybridModel(input_size, checkpoint['n_qubits'])
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        # Prepare input data
        scaled_input = scaler_X.transform(input_data)
        input_tensor = torch.FloatTensor(scaled_input)
        
        # Make prediction
        with torch.no_grad():
            scaled_prediction = model(input_tensor)
            prediction = scaler_y.inverse_transform(scaled_prediction.numpy())
        
        return prediction
    except Exception as e:
        raise Exception(f"Prediction error: {str(e)}")

# Example usage and model training
if __name__ == "__main__":
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Define model parameters
    input_size = 81  # Using 81 features as required by the model
    n_qubits = 4
    
    # Create synthetic dataset
    X = np.random.rand(100, input_size)  # 100 samples, 81 features
    y = np.sin(X.mean(axis=1) * 2 * np.pi).reshape(-1, 1) + 0.1 * np.random.randn(100, 1)  # Simple function with noise
    
    # Scale the data
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    
    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y)
    
    # Convert to torch tensors
    X_tensor = torch.FloatTensor(X_scaled)
    y_tensor = torch.FloatTensor(y_scaled)
    
    # Initialize and train the model
    model = HybridModel(input_size, n_qubits)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()
    
    # Train for a few epochs
    for epoch in range(10):
        optimizer.zero_grad()
        y_pred = model(X_tensor)
        loss = criterion(y_pred, y_tensor)
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 2 == 0:
            print(f'Epoch [{epoch+1}/10], Loss: {loss.item():.4f}')
    
    # Save the trained model
    save_model(model, scaler_X, scaler_y, input_size, n_qubits)
    
    # Test prediction
    test_input = np.random.rand(1, 81)  # Example with 81 features
    prediction = load_model_and_predict(test_input)
    print(f"\nTest prediction for random input: {prediction[0][0]:.4f}")
