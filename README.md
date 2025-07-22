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

 # Results Screenshots
<img width="449" height="245" alt="image" src="https://github.com/user-attachments/assets/c160e957-8eae-412b-8e54-0826a8061058" />
<img width="449" height="245" alt="image" src="https://github.com/user-attachments/assets/21486ac8-7857-4e51-b2ac-533ac1d31afb" />
<img width="449" height="245" alt="image" src="https://github.com/user-attachments/assets/7a7d40b2-ca42-496f-8113-3379445fc25b" />
<img width="449" height="245" alt="image" src="https://github.com/user-attachments/assets/deffcd55-df66-44d6-b07b-fe3a60961b3c" />
<img width="440" height="245" alt="image" src="https://github.com/user-attachments/assets/935ba187-86bb-4dae-b522-ad29a2b7b5b4" />
<img width="695" height="575" alt="image" src="https://github.com/user-attachments/assets/49991ad7-bf88-4a4a-a79f-9eb2bb66de4c" />
<img width="691" height="562" alt="image" src="https://github.com/user-attachments/assets/2e931717-09e7-4c49-bd29-63494a8838b9" />
<img width="623" height="547" alt="image" src="https://github.com/user-attachments/assets/c5bdb858-19b5-4b9e-ac3f-d536b06b12e8" />
<img width="1146" height="583" alt="image" src="https://github.com/user-attachments/assets/b71cdf6a-423b-43b2-a93f-2524e2a84d7d" />


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
