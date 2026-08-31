# 🌫️ Air Quality Index (AQI) Prediction

> **Domain:** Environment & Energy | **Problem Type:** Regression | **Dataset:** Air Quality Data in India (2015–2020)

A machine learning project that predicts the **Air Quality Index (AQI)** of Indian cities using pollutant concentration data. Built with scikit-learn, this notebook walks through the full ML pipeline — from problem definition to model evaluation — using real-world data from India's Central Pollution Control Board (CPCB).

---

## 📌 Problem Statement

Air Quality Index (AQI) quantifies how polluted the air is on a given day. A high AQI indicates dangerous air quality. This project predicts the AQI value using pollutant measurements such as PM2.5, PM10, NO₂, SO₂, CO, and more.

**Why does this matter?**
- Citizens can avoid outdoor activities on high-AQI days
- Governments can take proactive measures to reduce pollution
- Hospitals can prepare for increased patient load on high-AQI days
- Machine Learning captures complex, non-linear relationships between pollutants and AQI that traditional methods cannot

---

## 📂 Project Structure

```
ML_Mini_project_Environment_&_Energy/
├── src/
│   └── AQI_quality_Prediction.ipynb   ← Main notebook (this project)
└── Dataset/
    └── AQI/
        ├── city_day.csv               ← Primary dataset used
        ├── city_hour.csv
        ├── station_day.csv
        ├── station_hour.csv
        └── stations.csv
```

---

## 📊 Dataset

| Property         | Details                                            |
|------------------|----------------------------------------------------|
| **Dataset Name** | Air Quality Data in India (2015–2020)              |
| **Source**       | Kaggle / CPCB (Central Pollution Control Board)    |
| **File Used**    | `city_day.csv`                                     |
| **Rows**         | 29,531                                             |
| **Columns**      | 16                                                 |
| **Target**       | `AQI` (numeric value)                              |
| **Time Period**  | January 2015 – July 2020                           |
| **Coverage**     | 26 Indian cities                                   |

### Feature Columns

| Column        | Description                                 |
|---------------|---------------------------------------------|
| `City`        | Name of the city                            |
| `Date`        | Date of measurement                         |
| `PM2.5`       | Fine particulate matter (highly harmful)    |
| `PM10`        | Coarse particulate matter                   |
| `NO`          | Nitric oxide                                |
| `NO2`         | Nitrogen dioxide                            |
| `NOx`         | Total nitrogen oxides                       |
| `NH3`         | Ammonia                                     |
| `CO`          | Carbon monoxide                             |
| `SO2`         | Sulphur dioxide                             |
| `O3`          | Ozone                                       |
| `Benzene`     | Benzene (organic chemical)                  |
| `Toluene`     | Toluene (organic chemical)                  |
| `Xylene`      | Xylene (organic chemical)                   |
| `AQI`         | **Target** — Air Quality Index value        |
| `AQI_Bucket`  | Category: Good / Moderate / Poor / Severe  |

### ⚠️ Dataset Limitations
- ~16% of AQI values are missing
- No weather data (temperature, humidity, wind speed)
- Data covers up to July 2020 only

---

## 🔬 Methodology

The notebook follows a structured 9-step ML pipeline:

1. **Problem Definition** — Define the problem, stakeholders, and why ML is appropriate
2. **Data Loading** — Load `city_day.csv` into a Pandas DataFrame
3. **Data Understanding** — Inspect shape, data types, missing values, and statistics
4. **Exploratory Data Analysis (EDA)** — Visualize distributions, correlations, and trends
5. **Data Preprocessing** — Handle missing values, encode categorical variables, scale features
6. **Feature Engineering** — Select relevant features for modeling
7. **Model Training** — Train four regression models
8. **Model Evaluation** — Compare models using MAE, MSE, RMSE, and R² Score
9. **Model Comparison** — Visualize performance and select the best model

---

## 🤖 Models Trained

| Model                  | Description                                                                |
|------------------------|----------------------------------------------------------------------------|
| **Linear Regression**  | Baseline model; fits a linear relationship between features and AQI        |
| **Decision Tree**      | Tree-based model with `max_depth=8` to prevent overfitting                 |
| **Random Forest**      | Ensemble of 100 decision trees (`n_estimators=100`); reduces variance      |
| **Gradient Boosting**  | Sequential tree boosting (`n_estimators=100`); corrects previous errors    |

---

## 📈 Results

| Model              |   MAE |      MSE |  RMSE | R² Score |
|--------------------|------:|---------:|------:|---------:|
| Linear Regression  | 30.99 | 3,513.09 | 59.27 |   0.8081 |
| Decision Tree      | 25.68 | 2,487.22 | 49.87 |   0.8642 |
| **Random Forest**  | **20.50** | **1,630.90** | **40.38** | **0.9109** |
| Gradient Boosting  | 23.39 | 1,869.55 | 43.24 |   0.8979 |

> 🏆 **Best Model: Random Forest Regressor** — achieved the lowest RMSE (40.38) and the highest R² Score (0.9109), explaining ~91% of the variance in AQI values.

---

## 🛠️ Tech Stack

- **Language:** Python 3
- **Notebook:** Jupyter Notebook
- **Libraries:**
  - `pandas` — Data manipulation
  - `numpy` — Numerical computation
  - `matplotlib` & `seaborn` — Data visualization
  - `scikit-learn` — Machine learning models and metrics

---

## 🚀 Getting Started

### Prerequisites

```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

### Run the Notebook

1. Clone the repository:
   ```bash
   git clone https://github.com/RiteshJadhav283/AQI-prediction-Model.git
   cd AQI-prediction-Model
   ```

2. Launch Jupyter:
   ```bash
   jupyter notebook src/AQI_quality_Prediction.ipynb
   ```

3. Make sure the dataset is present at `Dataset/AQI/city_day.csv` before running the cells.

---

## 📉 Metric Definitions

| Metric       | Full Name                  | Interpretation                          |
|--------------|----------------------------|-----------------------------------------|
| **MAE**      | Mean Absolute Error        | Average error in AQI units. Lower = better |
| **MSE**      | Mean Squared Error         | Penalizes large errors more. Lower = better |
| **RMSE**     | Root Mean Squared Error    | Same unit as AQI. Lower = better        |
| **R² Score** | Coefficient of Determination | % variance explained. Higher = better (max 1.0) |

---

## 🌆 Cities Covered

The dataset covers **26 Indian cities** including Ahmedabad, Delhi, Mumbai, Bengaluru, Chennai, Hyderabad, Kolkata, and more.

---

## 📄 License

This project is for educational purposes. The dataset is sourced from [Kaggle](https://www.kaggle.com/) and the Central Pollution Control Board (CPCB), Government of India.

---

*Built as part of an ML Mini Project on Environment & Energy.*
