
# 🏡 House Price Prediction Pro

A Machine Learning-powered web application built using Python and Streamlit to analyze housing data and predict house prices based on property features.

## 🚀 Features

- Interactive house price prediction form
- Professional Streamlit dashboard
- Housing dataset exploration and filtering
- House price distribution visualization
- City-wise average house price analysis
- Linear Regression, Decision Tree, and Random Forest comparison
- Model evaluation using MAE, RMSE, and R² Score
- Feature importance visualization
- Download filtered dataset as CSV
- Automated data preprocessing and categorical encoding

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Streamlit
- Plotly
- Joblib

## 📂 Project Structure

```text
HousePriceProject/
├── app.py
├── train_model.py
├── data.csv
├── requirements.txt
├── README.md
├── house_price_model.joblib
└── model_config.json
```

**Note:** `house_price_model.joblib` and `model_config.json` are generated automatically when the training script runs.

## ⚙️ Installation and Setup

### 1. Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd HousePriceProject
```

Alternatively, download the project and open its folder in VS Code.

### 2. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 3. Train the machine learning models

```bash
python train_model.py
```

The script trains and evaluates three regression models, selects the model with the lowest test-set MAE, and saves the model and its configuration.

### 4. Run the Streamlit application

```bash
python -m streamlit run app.py
```

Open the local URL displayed in your terminal if the browser does not open automatically.

## 📊 Model Evaluation

The application compares three regression algorithms:

| Algorithm | Purpose |
|---|---|
| Linear Regression | Establishes a linear prediction baseline |
| Decision Tree Regressor | Learns nonlinear decision rules |
| Random Forest Regressor | Combines multiple decision trees |

The models are evaluated using:

- **MAE:** Mean Absolute Error
- **RMSE:** Root Mean Squared Error
- **R² Score:** Measures how well the model explains variation in the target

The best model is selected according to the lowest MAE on the test set. Actual performance depends on the dataset and training configuration.

## 📈 Application Sections

### Dashboard
Displays dataset statistics, price distribution, city-wise price analysis, and sample records.

### Predict Price
Accepts property details and generates a price estimate using the saved machine learning pipeline.

### Dataset Explorer
Allows users to inspect columns, search records, filter rows with missing values, and download the filtered data.

### Model Performance
Displays model comparison metrics and feature importance when supported by the selected model.

## 📁 Dataset

The application expects a file named `data.csv` in the project directory. It must contain a target column named `price`.

The model excludes the raw `date` and `street` columns during training and processes missing values and categorical features through a machine learning pipeline.

## 🔮 Future Improvements

- Improve prediction accuracy through feature engineering and model tuning
- Add additional regression algorithms
- Deploy the application online
- Add more detailed location-based visualizations
- Add automated model retraining and monitoring

## ⚠️ Disclaimer

Predictions are estimates based on patterns in the supplied dataset. They are not professional property valuations and should not be treated as guaranteed market prices.

## 👩‍💻 Author

**Aditi Shrivastav**

Machine Learning and Data Science Project

---

⭐ If you find this project useful, consider giving the repository a star.
