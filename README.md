## Student's Performance Project
The Student's Performance Project is a research project that aims to investigate the relationship between student performance and various other factors that affects his perforamnce

## 📁 Project Directory Structure

The following is the structure of the project:

```plaintext
📁 STUDENT_PERFORMANCE/
├── 📁 artifacts/
│   ├── data.csv
│   ├── model.pkl
│   ├── preprocessor.pkl
│   ├── test.csv
│   └── train.csv
├── 📁 logs/
├── 📁 src/
│   ├── 📁 __pycache__/
│   ├── 📁 components/
│   │   ├── __init__.py
│   │   ├── data_ingestion.py
│   │   ├── data_transformation.py
│   │   └── model_trainer.py
│   ├── 📁 notebook/
│   │   ├── 📁 data/
│   │   │   └── student.csv
│   │   ├── eda.ipynb
│   │   └── model_performance.ipynb
│   ├── 📁 pipeline/
│   │   ├── __init__.py
│   │   ├── exception.py
│   │   ├── logger.py
│   │   └── utils.py
│   └── __init__.py
├── 📁 st_venv_3.8/
├── 📁 Student_Performance.egg-info/
├── 📁 templates/
│   ├── data_input.html
│   └── home_page.html
├── .gitignore
├── app.py
├── README.md
├── requirements.txt
└── setup.py
```

## 📄 File-by-File Explanation

### 🔧 Root Files

- **app.py**  
  Entry point of the application. Handles routing and renders the HTML pages for user input and predictions.

- **requirements.txt**  
  Contains all the dependencies required to run the project in a Python environment.

- **setup.py**  
  Setup script for packaging the application.

- **.gitignore**  
  Specifies files and folders to be ignored by Git (e.g., virtual environments, logs, model files).

- **README.md**  
  Provides an overview of the project, installation instructions, usage guide, and structure documentation.

---

### 📁 artifacts/

Stores intermediate and final outputs like:

- **data.csv, train.csv, test.csv** – Original and split data files.
- **model.pkl** – Trained machine learning model.
- **preprocessor.pkl** – Serialized preprocessing pipeline.

---

### 📁 logs/

Stores logging files that help in debugging and tracking model training and data pipeline activities.

---

### 📁 templates/

- **data_input.html** – Web page for user data input.
- **home_page.html** – Landing page for the application.

---

### 📁 src/

Main source code directory, organized into modular components.

#### 📁 components/

- **data_ingestion.py** – Loads and splits the raw data into training and testing datasets.
- **data_transformation.py** – Applies preprocessing to data (scaling, encoding, etc.).
- **model_trainer.py** – Trains and evaluates the machine learning model.
- **\_\_init\_\_.py** – Marks the folder as a Python module.

#### 📁 notebook/

- **eda.ipynb** – Exploratory Data Analysis to understand trends, distributions, and relationships in the dataset.
- **model_performance.ipynb** – Analyzes model evaluation metrics like accuracy, precision, recall, etc.

##### 📁 notebook/data/

- **student.csv** – Dataset used for analysis and model building.

#### 📁 pipeline/

- **exception.py** – Custom exception handling for consistent error messages.
- **logger.py** – Centralized logging setup.
- **utils.py** – Utility functions used across the pipeline (e.g., saving models, loading files).
- **\_\_init\_\_.py** – Module initializer.

---

### 📁 st_venv_3.8/

Virtual environment directory (optional to include in version control if not ignored via `.gitignore`).

---

### 📁 Student_Performance.egg-info/

Contains metadata and information used when the project is packaged or distributed as a Python package.

## Student Performance Indicator Project Details

#### Life cycle of Machine learning Project

- Understanding the Problem Statement
- Data Collection
- Data Checks to perform
- Exploratory data analysis
- Data Pre-Processing
- Model Training
- Choose best model

### 1) Problem statement
- This project understands how the student's performance (test scores) is affected by other variables such as Gender, Ethnicity, Parental level of education, Lunch and Test preparation course.


### 2) Data Collection
- Dataset Source - https://www.kaggle.com/datasets/spscientist/students-performance-in-exams?datasetId=74977
- The data consists of 8 column and 1000 rows.

### 2.2 Dataset information
- gender : sex of students  -> (Male/female)
- race/ethnicity : ethnicity of students -> (Group A, B,C, D,E)
- parental level of education : parents' final education ->(bachelor's degree,some college,master's degree,associate's degree,high school)
- lunch : having lunch before test (standard or free/reduced) 
- test preparation course : complete or not complete before test
- math score
- reading score
- writing score


