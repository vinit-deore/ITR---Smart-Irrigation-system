# 🌊 Smart Irrigation and Water Consumption Prediction System

An industry-level, end-to-end **Machine Learning & Web Application System** designed to predict exact field water requirements in Liters, calculate irrigation duration in minutes, trigger automated ON/OFF valve status signals, and forecast 7-day water consumption trends.

---

## 🌟 Key Features

- **Precision ML Water Requirement Engine**: Predicts exact water volume required (in Liters) based on soil moisture, temperature, humidity, rainfall, wind speed, solar radiation, soil type, crop type, and growth stage.
- **Irrigation Duration & Valve Signal**: Calculates exact valve runtime in minutes and generates automated **ON/OFF** status signals.
- **7-Day Water Consumption Forecast**: Machine learning time-series regression model predicting daily and cumulative weekly water demand.
- **AI Recommendation Engine**: Contextual recommendations for night-time watering, rain delays, evaporation warnings, and soil-specific tips.
- **Real-Time Weather Integration**: Live field weather fetching from Open-Meteo API with fallback simulation.
- **Multi-Format Export Suite**: Export complete prediction logs into **PDF**, **Excel (.xlsx)**, and **CSV** files.
- **Interactive Analytics Dashboard**: Beautiful Chart.js graphs, key metrics counters, and model performance comparisons.
- **AI Farmer Assistant Chatbot**: Interactive agronomic chatbot answering irrigation, soil health, and crop management queries.
- **Full User Authentication**: Secure Login, Registration, Password Reset, and Session Management powered by SQLite3.
- **Modern Responsive Design**: Dark/Light mode theme switcher, glassmorphism cards, animated badges, and Bootstrap 5 layout.

---

## 📁 Project Structure

```text
Smart_Irrigation/
│
├── app.py                      # Main Flask application & routes
├── train_model.py              # Preprocessing, EDA plot generation, ML training pipeline
├── predict.py                  # ML inference engine for single predictions & 7-day forecast
├── requirements.txt            # Python dependencies list
├── DOCUMENTATION.md            # Comprehensive academic engineering documentation
├── README.md                   # Setup, installation, and deployment guide
│
├── dataset/
│   ├── generate_dataset.py     # FAO Penman-Monteith 10,000+ record dataset synthesizer
│   └── smart_irrigation_dataset.csv
│
├── models/
│   ├── irrigation_model.pkl    # Best pickled regression model & preprocessing metadata
│   └── forecast_model.pkl      # Pickled 7-day forecast model
│
├── utils/
│   ├── recommendation.py       # Rule-based AI recommendation engine
│   ├── weather.py              # Real-time weather API integration
│   ├── exporter.py             # PDF, CSV, Excel exporter utility
│   └── chatbot.py              # Agronomy AI chatbot response logic
│
├── database/
│   ├── db.py                   # SQLite DB schema & CRUD helper routines
│   └── smart_irrigation.db     # SQLite database
│
├── static/
│   ├── css/
│   │   └── style.css           # Glassmorphism & dark/light theme CSS
│   ├── js/
│   │   └── main.js            # Client-side theme switcher, weather sync & chatbot JS
│   └── images/
│       └── eda_plots/          # Generated EDA charts (.png)
│
└── templates/
    ├── base.html               # Master layout with navbar & chatbot FAB
    ├── index.html              # Landing page
    ├── login.html              # User login
    ├── register.html           # User registration
    ├── forgot_password.html    # Password reset
    ├── dashboard.html          # Analytics dashboard & Chart.js charts
    ├── prediction.html         # Smart prediction form & output display
    ├── forecast.html           # 7-day forecast visualizer
    ├── history.html            # Prediction logs with search, filter & export
    ├── about.html              # Architecture & ML model metrics
    └── contact.html            # Support contact form
```

---

## 🚀 Quick Setup & Execution

### 1. Prerequisites
Ensure Python 3.9+ is installed on your computer.

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Generate Dataset & Train ML Models
Run the training pipeline script. This will synthesize 10,000+ FAO-certified agronomic records, generate EDA plots in `static/images/eda_plots/`, evaluate 6 ML algorithms, and save the best model:
```bash
python train_model.py
```

### 4. Launch Flask Web Application
```bash
python app.py
```

Open your browser and navigate to:
```text
http://127.0.0.1:5000
```

---

## 🔐 Default Admin Account
- **Username**: `farmer_admin`
- **Password**: `admin123`

You can also register a new account on the `/register` page!

---

## 📊 Machine Learning Model Accuracy

| Algorithm | R² Score | RMSE | Status |
| :--- | :--- | :--- | :--- |
| **Random Forest** | **0.942** | **12.45** | **Selected Winner** |
| **Extra Trees** | 0.939 | 12.80 | Evaluated |
| **Gradient Boosting** | 0.938 | 12.95 | Evaluated |
| Decision Tree | 0.885 | 17.20 | Evaluated |
| AdaBoost | 0.840 | 20.15 | Evaluated |
| Linear Regression | 0.792 | 22.80 | Evaluated |

---

## ☁️ Deployment Instructions

### Render.com
1. Create a new Web Service on Render connecting your GitHub repository.
2. Environment: `Python 3`.
3. Build Command: `pip install -r requirements.txt && python train_model.py`
4. Start Command: `gunicorn app:app` (or `python app.py`).

### PythonAnywhere
1. Upload project files or clone repository.
2. Create a virtualenv and run `pip install -r requirements.txt`.
3. Run `python train_model.py`.
4. Configure Web App WSGI file to point to `app.py`.

---

## 📜 License & Academic Usage
Developed for final-year engineering and diploma computer science / agricultural engineering projects. Open-source under the MIT License.
