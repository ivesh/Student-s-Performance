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

- **exception.py** – Custom exception handling for consistent error messages.
- **logger.py** – Centralized logging setup.
- **utils.py** – Utility functions used across the pipeline (e.g., saving models, loading files). This file contains functions:
  **def save_model**: Saves the model created in the picke file. 
  **def evaluate_model**: Takes parameter X_train,y_train, X_test, y_test, model and param. The evaluate model checks the best model with score values and with grid search cv and chooses the best model.
  **def load_model** - Loads thr pickle file as read byte mode (rb). 

- **\_\_init\_\_.py** – Module initializer.

#### 📁 components/

- **data_ingestion.py** – Loads and splits the raw data into training and testing datasets.
  - **Dataingestionconfig(class)**- Has a decorater dataclass to simplify the creation of classes that are primarily used to store data by automatically generating common methods and providing customization options. This dataingestionconfig class is made to define where training,test and raw data resides.
  - **DataIngestion(class)**- Does not have a dataclass decorater as along with defining variables we also need to create functions for ingesting data from data sources along with having to perform train/test split.
    - **def __init__(Function)**: Initializes the data ingestion config which contains the path of train, test and raw data.
    - **def initiate_data_ingestion(Function)**: With logging and try catch bleck we are starting the reading of the data under src->notebook->data->student.csv. During this phase data can be read through different sources as required like MongoDB, SQL, or Big data. We then start loging the start of train/test spilt with the split being 80/20. By using the os python framework we also check if train_data even if availble we need to create a new path to store the train,test and raw data by performing train test split.
- **data_transformation.py** – Applies preprocessing to data (scaling, encoding, column trasformer etc.). In this section we are using column transformer to trasform data in the form of pipeline where one hot encoding, standard scaling is performed in a step by step procedure.
  - **DataTransformationConfig(Class)**- Any inputs received form the data ingestion has to be passed to data transformation.
  - **DataTransformation(Class)**- Contains functions to handle categorical and numerical features.
    - **def __init__(Function)**- Initializes the data transformation config which contains the path where the preprocessor.pkl file has to reside.
    - **def get_data_transformation_obj**- A function to return the preprocessor object that contains column transformer along with steps within the pielines which transforms numerical column using imputer to handle missing values, and standard scalar. 
    For Categorical colums we are handling the missing values using simple imputer's most frequent stratergy, one hot encoding, and standard scaling.
    - **def initiate_data_transformation(Function, Parameters: Self, train_path,test_path)**: This function is responsible for generating preprocessor.pkl file which is the preprocessed data to be given for model training. We are splitting the train_df to input_training_data and target_training_data and test_df input_test_data and target_test_data to perform fit_transform and transform to convert the same to arrays. We also use np.c_ function in NumPy is used to concatenate arrays along the second axis (columns). 
    Target column choosen is Math Score, however either total score or average score could also be taken as target score. We finally save the object as preprocessor.pkl under artifatcs folder.
- **model_trainer.py** – Trains and evaluates the machine learning models.
  - **def ModelTrainerConfig**: Contains the path where model.pkl is going to reside.
  - **def ModelTrainer**- The parameters will consist of self, train array, test array. We divide our taining/test dataset X_train, y-train,X_test, y_test. And, choose models to train for our regression problem. 
    "Linear Regression",
    "Decision Tree Regressor",
    "Random Forest Regressor",
    "Gradient Boosting Regressor",
    "AdaBoost Regressor",
    "KNN Regressor", 
    and "XGBoost Regressor".
    
    We also add hyperparameter tuning for each of these models.
    
      params = {
      
      "Decision Tree Regressor": {
          'criterion': ['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],  # Loss function to measure quality of a split
          # 'splitter': ['best', 'random'],  # 'best' chooses best split, 'random' selects randomly (useful in bagging)
          # 'max_features': ['sqrt', 'log2'],  # Number of features to consider at each split
          # 'max_depth': [None, 5, 10, 20],  # Limit depth to control overfitting
          # 'min_samples_split': [2, 5, 10],  # Min samples required to split an internal node
          # 'min_samples_leaf': [1, 2, 4],  # Min samples required at each leaf node
      },

      "Random Forest Regressor": {
          # 'criterion': ['squared_error', 'absolute_error'],  # Similar to Decision Tree
          # 'max_features': ['sqrt', 'log2', None],  # Controls feature randomness, 'sqrt' is default
          'n_estimators': [8, 16, 32, 64, 128, 256],  # Number of trees in the forest
          # 'max_depth': [None, 10, 20],  # Tree depth to control overfitting
          # 'min_samples_leaf': [1, 2, 4],  # Minimum samples at a leaf node
          # 'n_jobs': [-1]  # Use all processors
      },

      "Gradient Boosting Regressor": {
          'learning_rate': [.1, .01, .05, .001],  # Shrinks contribution of each tree
          'subsample': [0.6, 0.7, 0.75, 0.8, 0.85, 0.9],  # % of samples used for each tree (stochastic)
          'n_estimators': [8, 16, 32, 64, 128, 256],  # Number of boosting stages
          # 'loss': ['squared_error', 'huber', 'absolute_error', 'quantile'],  # Use 'huber' for robustness to outliers
          # 'criterion': ['squared_error', 'friedman_mse'],
          # 'max_features': ['auto', 'sqrt', 'log2'],  # Control feature subset size at each split
          # 'max_depth': [3, 5, 7],  # Tree depth for base learners
      },

      "Linear Regression": {
          'fit_intercept': [True, False],  # Whether to calculate the intercept
          'copy_X': [True, False],  # If False, input X may be overwritten (to save memory)
          'positive': [True, False],  # Constrain coefficients to be positive (can help interpretability)
          # 'n_jobs': [-1],  # Use all cores (only effective when y is multi-target or X is sparse)
          # 'tol': [1e-4, 1e-6, 1e-8],  # Solver precision
      },

      "XGBRegressor": {
          'learning_rate': [.1, .01, .05, .001],  # How much to shrink contribution of each tree
          'n_estimators': [8, 16, 32, 64, 128, 256],  # Number of boosting rounds
          # 'max_depth': [3, 4, 5, 6],  # Maximum depth of tree
          # 'subsample': [0.6, 0.8, 1.0],  # Row sampling
          # 'colsample_bytree': [0.6, 0.8, 1.0],  # Feature sampling
          # 'reg_alpha': [0, 0.01, 0.1],  # L1 regularization
          # 'reg_lambda': [0.1, 1.0],  # L2 regularization
      },

      "AdaBoost Regressor": {
          'learning_rate': [.1, .01, 0.5, .001],  # Shrinks contribution of each weak learner
          'n_estimators': [8, 16, 32, 64, 128, 256],  # Number of weak learners
          # 'loss': ['linear', 'square', 'exponential'],  # Controls weight update, 'square' can penalize large errors more
      },

      "KNN Regressor": {
          'n_neighbors': [3, 5, 7, 9, 11, 13],  # Number of neighbors to use
          'weights': ['uniform', 'distance'],  # Weight function used in prediction
          'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],  # Algorithm for neighbor search
          # 'p': [1, 2],  # Distance metric: 1 = Manhattan, 2 = Euclidean
      },

      "XGBoost Regressor": {
          'learning_rate': [.1, .01, .05, .001],  # Controls step size shrinkage
          'n_estimators': [8, 16, 32, 64, 128, 256],  # Total trees to fit
          'max_depth': [3, 4, 5, 6, 7, 8],  # Max depth of each tree
          # 'subsample': [0.5, 0.75, 1.0],  # % of rows sampled per tree
          # 'colsample_bytree': [0.5, 0.75, 1.0],  # % of columns per tree
          # 'reg_lambda': [1.0, 0.1, 10],  # L2 regularization
          # 'reg_alpha': [0, 0.1, 1],  # L1 regularization
      }
  }
    - After Generating the report as to which model has performed the best using **evaluate model** function from utils we choose the best model store the value in best_model_score.
   
- **\_\_init\_\_.py** – Marks the folder as a Python module.

#### 📁 notebook/

- **eda.ipynb** – Exploratory Data Analysis to understand trends, distributions, and relationships in the dataset.
- **model_performance.ipynb** – Analyzes model evaluation metrics like accuracy, precision, recall, etc.

##### 📁 notebook/data/

- **student.csv** – Dataset used for analysis and model building.

#### 📁 pipeline/

- **predict_pipeline.py** – Predict pipeline needs to predict the incoming feature data. It has 2 classes.
  **PredicctPipeline(class)**-
    **def __init__(function)** : Self initialized.
    **def Predict**- Predicts the custom data fetched from HTML input with the path given for preprocesser.pkl and model.pkl saved under artifacts folder.
  **CustomData(class)**- Mapping the html data from the user to the backend for prediction with the model.pkl file saved. 
    **def __init__(Function)**- Self initialized with all the features which was required to train the model.
    **def get_html_as_df(function)**- We are reading the html input features as dictionary and converting them to data frame as that's the way we trained our model.
- **train_pipeline.py** – Centralized logging setup.
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
## Model Evaluation

## Model Explanations and Performance Analysis

### Common Regression Metrics

- **Root Mean Squared Error (RMSE):** Represents the standard deviation of prediction errors; lower is better.
- **Mean Absolute Error (MAE):** Average absolute prediction error; lower is better.
- **R² Score (Coefficient of Determination):** Proportion of the variance in the target variable explained by the model; closer to 1 indicates better fit.

### 1. Linear Regression

**How it Works:**  
Fits a straight line by minimizing the squared differences between actual and predicted values. Assumes linear relationships between predictors and target.

**Results:**

|                | Train          | Test           |
|----------------|---------------|----------------|
| RMSE           | 5.3273        | 5.4096         |
| MAE            | 4.2787        | 4.2259         |
| R² Score       | 0.8741        | 0.8797         |

**Evaluation:**  
Linear regression gives a strong, consistent performance on both train and test sets, with high R² and low error metrics. The similarity between train and test metrics suggests the model generalizes well and is not overfitting.

### 2. Lasso Regression

**How it Works:**  
Linear regression with L1 regularization, which can shrink some coefficients to zero, effectively performing feature selection and reducing model complexity.

**Results:**

|                | Train          | Test           |
|----------------|---------------|----------------|
| RMSE           | 6.5938        | 6.5197         |
| MAE            | 5.2063        | 5.1579         |
| R² Score       | 0.8071        | 0.8253         |

**Evaluation:**  
Regularization reduces potential overfitting but can slightly decrease performance if important features are shrunk too much. R² is lower than standard linear regression, indicating higher bias but possibly increasing robustness to data noise.

### 3. Ridge Regression

**How it Works:**  
Linear regression with L2 regularization, penalizing large coefficients but without eliminating them. Helps prevent overfitting if there are many correlated predictors.

**Results:**

|                | Train          | Test           |
|----------------|---------------|----------------|
| RMSE           | 5.3233        | 5.3904         |
| MAE            | 4.2650        | 4.2111         |
| R² Score       | 0.8743        | 0.8806         |

**Evaluation:**  
Very similar performance to basic linear regression, indicating multicollinearity isn’t a large issue. Excellent generalization, high R², low RMSE and MAE.

### 4. K-Neighbors Regressor

**How it Works:**  
Predicts target value based on the average of the k-nearest data points in feature space. Captures non-linear relationships but is sensitive to data density.

**Results:**

|                | Train          | Test           |
|----------------|---------------|----------------|
| RMSE           | 5.7122        | 7.2516         |
| MAE            | 4.5187        | 5.6160         |
| R² Score       | 0.8553        | 0.7839         |

**Evaluation:**  
Decent fit on the training data but noticeably poorer generalization (lower R² on test), suggesting some overfitting or sensitivity to data distribution. The model may benefit from tuning k or data preprocessing.

### 5. Decision Tree

**How it Works:**  
Splits data into branches based on feature thresholds to minimize error at each node. Captures complex, non-linear relationships but can overfit easily.

**Results:**

|                | Train          | Test           |
|----------------|---------------|----------------|
| RMSE           | 0.2795        | 7.9549         |
| MAE            | 0.0187        | 6.3000         |
| R² Score       | 0.9997        | 0.7400         |

**Evaluation:**  
Extremely low errors and perfect R² on the training set — classic sign of overfitting. On test data, performance drops substantially, confirming poor generalization. Pruning or ensembling can help address this.

### 6. Random Forest Regressor

**How it Works:**  
An ensemble of multiple decision trees, averaging their predictions to reduce overfitting and improve generalization.

**Results:**

|                | Train          | Test           |
|----------------|---------------|----------------|
| RMSE           | 2.3131        | 6.0242         |
| MAE            | 1.8444        | 4.6728         |
| R² Score       | 0.9763        | 0.8509         |

**Evaluation:**  
Shows much-reduced overfitting compared to a single decision tree, with strong train-set accuracy and much better generalization on test data. Indicates this ensemble method is robust.

### 7. XGBRegressor

**How it Works:**  
Gradient boosting with decision trees; builds trees sequentially, each correcting predecessor’s errors. Known for high accuracy and effective handling of both linear and complex patterns.

**Results:**

|                | Train          | Test           |
|----------------|---------------|----------------|
| RMSE           | 1.0073        | 6.4733         |
| MAE            | 0.6875        | 5.0577         |
| R² Score       | 0.9955        | 0.8278         |

**Evaluation:**  
Extremely high performance on the training data—almost perfect fit—yet test errors are noticeably higher, showing mild overfitting. Still, this model captures nuanced relationships and may perform best with careful tuning and regularization.

### 8. AdaBoost Regressor

**How it Works:**  
Boosting method that sequentially builds weak learners (like shallow trees), emphasizing data points most often mispredicted. Focuses on minimizing bias and improving accuracy.

**Results:**

|                | Train          | Test           |
|----------------|---------------|----------------|
| RMSE           | 5.9005        | 6.0026         |
| MAE            | 4.8101        | 4.6575         |
| R² Score       | 0.8456        | 0.8519         |

**Evaluation:**  
Balanced performance across train and test sets, with no sign of overfitting. Errors and R² are competitive with other robust models, indicating reliable generalization.

## Summary Table

| Model               | Train R² | Test R² | Train RMSE | Test RMSE | Overfitting?         | Comments                             |
|---------------------|---------|---------|------------|-----------|----------------------|--------------------------------------|
| Linear Regression   | 0.8741  | 0.8797  | 5.33       | 5.41      | No                   | Strong baseline                      |
| Lasso               | 0.8071  | 0.8253  | 6.59       | 6.52      | No                   | Regularized, slightly lower fit      |
| Ridge               | 0.8743  | 0.8806  | 5.32       | 5.39      | No                   | Handles multicollinearity well       |
| K-Neighbors         | 0.8553  | 0.7839  | 5.71       | 7.25      | Moderate             | Lower generalization, sensitive      |
| Decision Tree       | 0.9997  | 0.7400  | 0.28       | 7.95      | Extreme              | Overfits, consider pruning           |
| Random Forest       | 0.9763  | 0.8509  | 2.31       | 6.02      | Minimal              | Good generalization                  |
| XGBRegressor        | 0.9955  | 0.8278  | 1.01       | 6.47      | Some                 | Powerful, tune for overfitting       |
| AdaBoost            | 0.8456  | 0.8519  | 5.90       | 6.00      | No                   | Stable, robust generalization        |

## Final Evaluation

- **Best Generalizers:** Linear Regression, Ridge, and Random Forest show strong and consistent train/test performance, indicating robust generalization.
- **Potential Overfitting:** Decision Tree and XGBRegressor fit training data extremely well but generalize less effectively. These models may need regularization or more data.
- **Recommendations:**  
  - Use ensemble methods (Random Forest, AdaBoost) for generally robust results.
  - Consider Linear or Ridge Regression for interpretability and stability.
  - Tune hyperparameters for K-Neighbors, XGB, and Decision Tree to reduce overfitting.
- **Metric Interpretation:**  
  - Close train and test scores = good generalization.
  - High train, much lower test scores = overfitting.
  - R² approaching 1, low RMSE/MAE = strong predictive power.

These analyses inform which models are best suited for production, balancing accuracy, generalizability, and interpretability.

[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/36958646/58a26b06-e955-498b-b2a4-8d9f014dc601/Gen-AI-Lead.docx
[2] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/36958646/987b8a54-484a-4604-a67f-6775db7f4459/Venkatesh_I.pdf-1.pdf

## Deployment options- 
- Elastic beanstalk Instance - To deploy in elastic bean stalk .ebextensions folder needs to be created under which python.config file should contain option setting. We have to make a new file called application.py and copy the flask code as AWS does not recognize app.py in config.
  python.config- code
    option_setting:
      "aws:elasticbeanstalk:container:python":
      WSGIPath:application:application
- **Docker Image**- To push it to amazon ECR(fully managed docker container to push docker images which are private) then can be downloaded to EC2 instance further. And, .github/workflow(Can be created under github actions which will have the yaml file needed for CI/CD pipelines) and dockerfile needs to be created. In steps:
  Step 1: Dockerfile Build
  Step 2: Github actions workflow
  Step 3: I AM user to be created in Amazon console- (While creating you can specify 2 permission for this user Amazon EC2 container registory full access and Amazon EC2 Full access.)
  Next->Users->Security credentials->Access Key(CLI)->create access key->Download csv file.
  Next go to-> Elastic Container Registry(ECR)->New container->Copy the ECR url
  Next EC2 instance-> Launch Instance-> Name(student performance), ubuntu(image), t2 medium (There will be charges hence have to delete it), ()->Lanch Instance -> Instaance ID-> Connect->Connect to ECR instance-> CLI will launch-> clear the screen
  - When we launch our EC2 instance we need to install some packages for dockers.
  Next, 
  Step 1: sudo apt-get update -y, sudo apt-get upgrade -y
  Step 2: curl -fsSL https://get.docker.com -o get-docker.sh
  Step 3: sudo sh get-docker.sh
  step 4: sudo usermod -aG docker ubuntu
  Step 5: newgrp docker
  - Create runner -> Git hub actions ->  




