# Bitcoin Price Prediction using Bidirectional LSTMs

This project implements a time-series forecasting model to predict Bitcoin (BTC) prices based on historical daily data. It utilizes a deep learning approach with a Bidirectional Long Short-Term Memory (LSTM) network to capture sequential dependencies in financial market data.

## Features
- **Time-Series Windowing**: Implements a sliding window approach (4-day lookback) to transform the series into a supervised learning problem.
- **Deep Learning Architecture**: Features a Bidirectional LSTM layer for robust feature extraction from both past and future temporal contexts.
- **Performance Evaluation**: Utilizes Root Mean Squared Error (RMSE) and R² score metrics to validate predictive accuracy.

## Repository Structure
- `BitCoin_Predictor.ipynb`: Interactive Jupyter Notebook containing data preprocessing, model training, and visual analysis.
- `BTC-USD.csv`: Historical dataset used for training (September 2014 – Present).
- `requirements.txt`: Necessary dependencies to run the environment.

## Technical Implementation
- **Data Scaling**: Uses `MinMaxScaler` to normalize price data between [0, 1] for stable neural network training.
- **Model Parameters**: 
  - 100 LSTM units with a Bidirectional wrapper.
  - Dropout via EarlyStopping to prevent overfitting.
  - Adam optimizer for efficient gradient descent.

## Results
The model achieved an **R² score of ~0.98**, indicating high precision in following the underlying price trends. Predicted vs. Actual values are visualized in the notebook's final sections.

## Usage
1. Clone the repository.
2. Install dependencies: `pip install tensorflow pandas numpy matplotlib seaborn scikit-learn`
3. Ensure `BTC-USD.csv` is in the same directory as the notebook.
4. Run the notebook cells to train the model and view predictions.
