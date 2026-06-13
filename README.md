# Diabetes Prediction Web App

> A Django-based web application that predicts whether a patient is diabetic or not using Logistic Regression — trained on the Pima Indians Diabetes Dataset.

---

## 🩺 Overview

This project is a machine learning web application built with **Django**. A user inputs 8 health parameters and the system returns a **Positive** or **Negative** diabetes prediction in real time.

The model is trained fresh on every prediction request using **Logistic Regression** with **StandardScaler** normalization — no pre-saved `.pkl` file required.

---

## ⚙️ How It Works

1. User visits the prediction page and enters 8 health metrics
2. The app reads the `diabetes.csv` dataset and trains a Logistic Regression model
3. The input is standardized using `StandardScaler`
4. The model predicts: **Positive** (diabetic) or **Negative** (not diabetic)
5. Result is displayed on the same page

---

## 🧪 Input Features

| # | Feature | Description |
|---|---|---|
| 1 | Pregnancies | Number of times pregnant |
| 2 | Glucose | Plasma glucose concentration |
| 3 | Blood Pressure | Diastolic blood pressure (mm Hg) |
| 4 | Skin Thickness | Triceps skinfold thickness (mm) |
| 5 | Insulin | 2-Hour serum insulin (mu U/ml) |
| 6 | BMI | Body mass index |
| 7 | Diabetes Pedigree Function | Genetic diabetes risk score |
| 8 | Age | Age in years |

---

## 🤖 ML Details

| Property | Value |
|---|---|
| Algorithm | Logistic Regression |
| Dataset | Pima Indians Diabetes Dataset |
| Preprocessing | StandardScaler normalization |
| Train/Test Split | 80% / 20% (stratified) |
| Random State | 2 |
| Framework | scikit-learn |

Dataset source: [Kaggle — Pima Indians Diabetes](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)

---

## 🛠️ Tech Stack

- **Backend:** Django (Python)
- **ML:** scikit-learn, NumPy, pandas
- **Frontend:** HTML, CSS (Django templates)
- **Database:** SQLite (Django default)

---

## 🚀 Getting Started

### 1. Clone the repo
```bash
git clone https://github.com/rupokavi/DiabetesPrediction.git
cd DiabetesPrediction
```

### 2. Install dependencies
```bash
pip install django scikit-learn pandas numpy
```

### 3. Add the dataset
Download `diabetes.csv` from [Kaggle](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database) and place it in the project root:
```
DiabetesPrediction/
└── diabetes.csv   ← place here
```

Then update the path in `views.py`:
```python
# Change this line in result() function:
diabetes_dataset = pd.read_csv('diabetes.csv')  # relative path
```

### 4. Run the server
```bash
python manage.py runserver
```

Visit `http://127.0.0.1:8000/` in your browser.

---

## 📁 Project Structure

```
DiabetesPrediction/
├── DiabetesPrediction/        ← Django app
│   ├── views.py               ← ML logic + prediction
│   ├── urls.py
│   └── settings.py
├── templates/
│   ├── home.html
│   └── predict.html           ← Input form + result display
├── static/
│   └── DiabetesPrediction/
│       └── images/
├── diabetes.csv               ← Dataset (not tracked in git)
├── db.sqlite3
├── manage.py
└── README.md
```

---

## ⚠️ Known Limitations

- The model is **retrained on every request** — not ideal for production; should be saved as a `.pkl` file using `joblib`
- The dataset path in `views.py` was originally **hardcoded** as an absolute path — update to a relative path before running on a new machine
- `diabetes.csv` is **not included** in the repo (add to `.gitignore` or upload separately)

---

## 🔮 Future Improvements

- [ ] Save trained model with `joblib` to avoid retraining on every request
- [ ] Add model accuracy display on the result page
- [ ] Try Random Forest / SVM and compare accuracy
- [ ] Deploy on Heroku or Railway

---

## 👤 Author

**Rupok Islam Avi**
B.Sc. in Industrial & Production Engineering, RUET
[Portfolio](https://rupokavi.github.io/aviii) · [GitHub](https://github.com/rupokavi)

---

*Built as a personal ML + web development practice project*
