# Forest Cover Type Prediction

This project provides an machine learning model to predict forest cover types based on cartographic features, using a TensorFlow neural network. The model is trained on the Forest Cover Type dataset, and a Streamlit web application allows users to input features for prediction and view model performance metrics, including training history, accuracy, loss and feature importance.

## Features

- **Prediction**: Input numerical (e.g., Elevation, Slope) and categorical (Wilderness Area, Soil Type) features to predict one of seven forest cover types (e.g., Spruce/Fir, Lodgepole Pine).
- **Model Evaluation**: Displays training and validation metrics (accuracy, loss) from the training process.
- **Feature Importance**: Visualizes permutation importance of features using validation data.
- **Dark Theme UI**: Premium, user-friendly Streamlit interface with Plotly visualizations.

## Repository Structure

- `app.py`: Streamlit web app for predictions and visualizing model performance.
- `requirements.txt`: Lists all required Python packages.
- `scaler.pkl`: Pre-fitted StandardScaler for preprocessing features.
- `history.json`: Training history (loss, accuracy, val_loss, val_accuracy).
- `forest_cover_model/`: Final model weights (optional, for reference).
- `.streamlit/config.toml`: Streamlit configuration for dark theme.

## Prerequisites

- Python 3.8+
- Git

## Setup Instructions

1. **Clone the Repository**:

   ```bash
   git clone https://github.com/chamindusenehas/forest_cover_predictor.git
   cd forest-cover-predictor
   ```

2. **Install Dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Streamlit App**:

   ```bash
   streamlit run app.py
   ```

## Usage

   - Open the app in your browser (default: `http://localhost:8501`).
   - **Prediction Page**:
     - Enter numerical features (e.g., Elevation, Slope) and select categorical features (Wilderness Area, Soil Type).
     - Click "Predict Cover Type" to see the predicted forest cover type.
   - **Model Details Page**:
     - View training metrics (accuracy, loss) and history plots.
     - Explore feature importance plots to understand which features contribute most to predictions.


## Dependencies

See `requirements.txt` for the full list. Key packages:

- TensorFlow 2.10.0: For model training and inference.
- Streamlit 1.31.0: For the web interface.
- NumPy 1.26.4, Pandas 2.2.0: For data processing.
- scikit-learn 1.5.0: For preprocessing and evaluation metrics.
- Plotly 5.22.0: For visualizations.
- joblib 1.4.2: For saving/loading the scaler.

## Notes

- **Feature Importance**: Computed using permutation importance on validation data. Negative importance suggests a feature may reduce model performance if included.

## Contributing

Feel free to open issues or submit pull requests for improvements, such as additional metrics, UI enhancements, or model optimizations.