# IBM Edunet Foundation Internship
# Employee_Salary_Prediction_Using_Machine_Learning
![maxresdefault](https://github.com/user-attachments/assets/889722b0-177e-406d-bb7b-5cfdc6e0f24e)

The Income Classification App is a user-friendly web application built using Streamlit that predicts whether an individual's income exceeds $50K per year based on demographic and employment attributes. It utilizes a trained machine learning classification model and supports both single-entry predictions via form inputs and batch predictions through CSV uploads.

This project demonstrates practical skills in data preprocessing, model training, and real-time deployment using Python libraries such as pandas, scikit-learn, and joblib. It's a compact example of end-to-end machine learning application deployment with an interactive UI.


# Income Classification App

A **Streamlit** web application that predicts whether a person earns **>50K or ≤50K** annually based on demographic and employment attributes. The app uses a pre-trained machine learning classification model and supports both real-time input predictions and batch predictions via CSV file upload.

---

## Features

-  Predict whether income is >50K or ≤50K
-  Input features manually or upload a CSV for batch prediction
-  Real-time predictions with interactive visual UI
-  Built with `Streamlit`, `scikit-learn`, `pandas`, and `joblib`
-  Easily deployable and customizable for other classification use-cases

---

##  Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/income-classification-app.git
cd income-classification-app
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Add the trained model
Ensure you have a trained model file named `income_classifier.pkl` in the root directory.

### 4. Run the application
```bash
streamlit run app.py
```

---

##  File Structure

- `app.py` - Main Streamlit application script
- `income_classifier.pkl` - Trained ML model (not included)
- `requirements.txt` - List of Python dependencies
- `README.md` - Project documentation

---

##  Example CSV Format

To use batch predictions, upload a CSV with the following column structure:

```csv
age,workclass,education,marital-status,occupation,relationship,race,sex,hours-per-week,native-country
39,State-gov,Bachelors,Never-married,Adm-clerical,Not-in-family,White,Male,40,United-States
...
```

---

##  Screenshots



## Author
**Aman Kumar Singh**  
🔗 [LinkedIn](https://www.linkedin.com/in/aman-kumar-singh-71a090206)  
📧 aksingh1652@gmail.com
