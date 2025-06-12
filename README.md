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

