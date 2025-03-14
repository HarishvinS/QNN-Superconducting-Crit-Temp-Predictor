# QNN-Superconducting-Crit-Temp-Predictor


## Overview
**QNN-Supercon-Predictor** is a Quantum Neural Network (QNN) model designed to predict superconducting materials' properties. The project explores the intersection of Quantum Computing and Material Science, leveraging quantum machine learning techniques to optimize material discovery for superconducting qubits.

## Features
- Implements a **Quantum Neural Network (QNN)** to predict superconducting material properties.
- Supports **Variational Quantum Circuits** for learning material property trends.
- Integrates **PennyLane** for hybrid quantum-classical machine learning.

## Usage
Download the repository and run the app.py file. No other setup is needed. The model weights are already trained so no need to run re-train the model, unless you have a better dataset you can train it on.
Once the web app is running locally, you can enter the 81 input parameters of the material of your choice if you have them on hand. If you don't have the required 81 features, you can simply hit the "Generate All Random Values" button and you'll good to go! Scroll down to the bottom of page and hit the "Predict Critical Temperature" Button. 



## Future Work
- Host the web app online so that users don't have to download the repository to access the app.
- Train the model on a lower number of features and ensure that the model is accurate, even with the decreased amount of features.

## License
This project is licensed under the MIT License.

