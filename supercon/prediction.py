import joblib
import numpy as np

# Import the exact model class (modify based on your actual class definition)
from superconducting_qnn import HybridModel
from superconducting_qnn import QuantumCircuit  # Ensure this is the correct import

# Load the trained model
model = joblib.load("trained_model.pkl")
print("✅ Model loaded successfully!\n")

# Define new material properties (replace with actual values)
new_materials = np.array([
    [1.2, 2.5, 0.8, 1.1, 3.8],  # Material 1 - using the same example values from supercon_run.py
    [2.3, 4.5, 6.7, 8.9, 5.6]   # Material 2
])

# Make predictions
predictions = model.predict(new_materials)

# Display results in a readable format
print("🔹 **Predictions for New Materials** 🔹")
for i, pred in enumerate(predictions):
    print(f"Material {i+1}: Predicted Property = {pred:.4f}")


