import pennylane as qml
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader

# Set seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# Define the Quantum Circuit Function
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

# Set up the PennyLane device and QNode
n_qubits = 4
dev = qml.device("default.qubit", wires=n_qubits)
qnode = qml.QNode(quantum_circuit, dev, interface="torch")

# Define the Hybrid Model using PyTorch
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
        
        # Process each sample in the batch
        q_out = []
        for i in range(batch_size):
            # Get quantum circuit output for each sample
            quantum_output = torch.tensor(qnode(x[i], self.weights), dtype=torch.float32)
            q_out.append(quantum_output.unsqueeze(0))
        
        # Combine all quantum outputs
        q_out = torch.cat(q_out, dim=0)
        x = self.post_net(q_out)
        return x

# Data Loading and Preprocessing Functions
def load_data(file_path,nrows=None):
    data = pd.read_csv(file_path, nrows=200)
    features = data.iloc[:, :-1].values
    targets = data.iloc[:, -1].values
    return features, targets

def prepare_data(features, targets):
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    X_scaled = scaler_X.fit_transform(features)
    y_scaled = scaler_y.fit_transform(targets.reshape(-1, 1))
    return X_scaled, y_scaled, scaler_X, scaler_y

# Main Training Routine
def run_experiment(dataset_file='train.csv', n_epochs=50):
    # Load and prepare data
    print("Loading and preparing data...")
    features, targets = load_data(dataset_file)
    X_scaled, y_scaled, scaler_X, scaler_y = prepare_data(features, targets)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_scaled, test_size=0.2, random_state=42
    )
    
    # Create data loaders
    train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    
    # Initialize model
    input_size = X_train.shape[1]
    model = HybridModel(input_size, n_qubits)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()
    
    print("Starting training...")
    for epoch in range(n_epochs):
        model.train()
        total_loss = 0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y.view(-1, 1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        if (epoch + 1) % 10 == 0:
            avg_loss = total_loss / len(train_loader)
            print(f"Epoch {epoch+1}/{n_epochs}, Loss: {avg_loss:.4f}")
    
    # Save model weights and scalers
    WEIGHTS_PATH = "trained_model_weights.pth"
    SCALER_PATH = "scalers.pth"
    
    # Save model state and scalers
    torch.save({
        'model_state_dict': model.state_dict(),
        'input_size': input_size,
        'n_qubits': n_qubits,
        'scaler_X_mean': scaler_X.mean_,
        'scaler_X_scale': scaler_X.scale_,
        'scaler_y_mean': scaler_y.mean_,
        'scaler_y_scale': scaler_y.scale_
    }, WEIGHTS_PATH)
    
    print("\n✅ Model weights and scalers saved successfully to", WEIGHTS_PATH)
    
    return model, scaler_X, scaler_y

if __name__ == "__main__":
    run_experiment(dataset_file='train.csv', n_epochs=50)



