# IBM Edunet Foundation Internship
# Employee_Salary_Prediction_Using_Machine_Learning
![maxresdefault](https://github.com/user-attachments/assets/889722b0-177e-406d-bb7b-5cfdc6e0f24e)

The Income Classification App is a user-friendly web application built using Streamlit that predicts whether an individual's income exceeds $50K per year based on demographic and employment attributes. It utilizes a trained machine learning classification model and supports both single-entry predictions via form inputs and batch predictions through CSV uploads. This project demonstrates practical skills in data preprocessing, model training, and real-time deployment using Python libraries such as pandas, scikit-learn, and joblib. It's a compact example of end-to-end machine learning application deployment with an interactive UI.

# Project Objective
The objective of this project is to build a machine learning model that predicts whether an individual earns more than $50,000 per year based on various demographic and employment-related features. The project focuses on developing a full ML pipeline — from data preprocessing to model deployment — using a user-friendly Streamlit web interface for both single and batch predictions.

# Datasets Used in This Project
The project uses the Adult Income Dataset from the UCI Machine Learning Repository. It contains census information about individuals such as:
Age
Workclass
Education level
Marital status
Occupation
Race and gender
Hours worked per week
Native country
* Target variable:
income (binary classification: >50K or <=50K)

# Tools & Technologies
* Python – Programming language
* Pandas – Data manipulation
* NumPy – Numerical computations
* Scikit-learn – Model training and evaluation
* Joblib – Model serialization
* Streamlit – Web application framework
* Matplotlib & Seaborn – Data visualization

 ##  Screenshots


# Methodology
1.Data Cleaning & Preprocessing
* Handle missing values and inconsistent labels
* Encode categorical variables (LabelEncoder, OneHotEncoding if required)
* Feature scaling for numerical data if necessary
2.Model Training
* Use classification algorithms (e.g., Gradient Boosting, Random Forest, Logistic Regression)
* Split data into training and testing sets
* Evaluate with metrics like accuracy, precision, recall, and F1-score
3.Model Serialization
* Save the trained model using joblib for future use
4.Deployment
* Build an interactive UI using Streamlit
* Enable single-record and batch CSV prediction uploads

# Key Insights
* Features such as education level, hours worked per week, occupation, and marital status show strong correlation with income level.
* Married individuals and those in professional or managerial roles are more likely to earn above $50K.
* Education beyond high school significantly increases the chances of earning >$50K.
* Machine learning models like Gradient Boosting provided high accuracy and interpretability for this binary classification task.
* The Streamlit interface enhances accessibility by allowing predictions without coding, making the solution more practical.

# Income Classification App

A **Streamlit** web application that predicts whether a person earns **>50K or ≤50K** annually based on demographic and employment attributes. The app uses a pre-trained machine learning classification model and supports both real-time input predictions and batch predictions via CSV file upload.

## Features

-  Predict whether income is >50K or ≤50K
-  Input features manually or upload a CSV for batch prediction
-  Real-time predictions with interactive visual UI
-  Built with `Streamlit`, `scikit-learn`, `pandas`, and `joblib`
-  Easily deployable and customizable for other classification use-cases

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

## Author
* Aman Kumar Singh
* www.linkedin.com/in/aman-kumar-singh-71a090206
* aksingh1652@gmail.com
