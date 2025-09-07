# NephroPredict: Machine Learning for Chronic Kidney Disease Detection

**NephroPredict** is a predictive modeling project designed to detect **Chronic Kidney Disease (CKD)** using clinical data.  
Multiple machine learning algorithms are applied, tuned, and compared to identify the most reliable model.

---
##  Features

-  **Data Preprocessing**  
  - Handling missing values  
  - Encoding categorical variables  
  - Balancing classes using **SMOTE oversampling**

-  **Exploratory Data Analysis (EDA)**  
  - Distribution plots  
  - Correlation heatmaps  
  - Pair plots  

-  **Models Implemented**  
  - Logistic Regression  
  - K-Nearest Neighbors (KNN)  
  - Support Vector Machine (SVM)  
  - Decision Tree  
  - Random Forest  

-  **Model Tuning**  
  - Hyperparameter optimization with GridSearchCV  

-  **Final Recommendation**  
  - **Random Forest** (most stable and accurate model)

---

##  Dataset

- **Source**: [UCI Machine Learning Repository – CKD Dataset](https://github.com/Abubakar-Shabbir/Machine-Learning-for-Chronic-Kidney-Disease-Detection.git)  
- **Attributes**: Blood Pressure (Bp), Specific Gravity (Sg), Albumin (Al), Sugar (Su), Red Blood Cells (Rbc), Blood Urea (Bu), Serum Creatinine (Sc), Sodium (Sod), Potassium (Pot), Hemoglobin (Hemo), White Blood Cell Count (Wbcc), Red Blood Cell Count (Rbcc), Hypertension (Htn), Class (Target Variable)  
- **Target Variable**: `Class` (1-CKD / 0-Not CKD)

---

##  Exploratory Data Analysis (EDA)

Some of the visualizations included in the project:

- Histogram & count plots for categorical features  
- Boxplots to detect outliers  
- Correlation heatmap to identify feature relationships  
- Pair plots (after preprocessing & oversampling)  

---

##  Modeling & Evaluation

Each model was trained **before and after hyperparameter tuning**, and evaluated on multiple metrics:

- Accuracy  
- Precision  
- Recall  
- F1-score  
- Cross-validation (mean & std)  

---

##  Results & Comparison

### Key Findings:
- **Logistic Regression** → Strong baseline model, slight decrease after tuning.  
- **KNN** → Achieved 100% training accuracy but slight overfitting on test data.  
- **SVM** → Significant improvement after tuning (kernel, regularization).  
- **Decision Tree** → Performed well initially, tuning reduced accuracy.  
- **Random Forest** → Best performing model with 100% training & **96.55% test accuracy**.  

 **Overall Winner: Random Forest** – robust and consistent across all evaluations.


---

##  Getting Started

Clone the repository:

```bash
git clone https://github.com/Abubakar-Shabbir/Machine-Learning-for-Chronic-Kidney-Disease-Detection.git
cd Machine-Learning-for-Chronic-Kidney-Disease-Detection
```
# Install dependencies:
```bash
jupyter notebook
```
# Acknowledgments

**Dataset:** Kaggle

**Libraries:** Scikit-learn, Pandas, NumPy, Matplotlib, Seaborn

**Inspiration:** Healthcare-focused ML research

