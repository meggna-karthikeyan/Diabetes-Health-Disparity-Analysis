# 🩺 Exploring Diabetes Data | Health Disparity Analysis Using Obesity & Inactivity Metrics

![image](https://github.com/user-attachments/assets/785881bb-8ab2-4573-ae9b-a7b44af5e8cb)

## 📝 Overview
This project investigates the **relationship between diabetes, obesity, and physical inactivity** using 2018 CDC public health data. Through the use of **linear and polynomial regression models**, this project aims to quantify how obesity and inactivity contribute to diabetes prevalence and inform data-driven public health interventions.

## 📦 Dataset Source

🔗 **Source:** Centers for Disease Control and Prevention (CDC)  
📅 **Year:** 2018  
🗂️ **Key Variables:**  
- `Diabetes (%)` — % of diagnosed diabetes cases per county  
- `Obesity (%)` — % of obese individuals per county  
- `Inactivity (%)` — % of physically inactive individuals per county

## 🛠 Technologies Used

![Python](https://img.shields.io/badge/Python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54) 
![Pandas](https://img.shields.io/badge/Pandas-150458.svg?style=for-the-badge&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-11557c?style=for-the-badge&logo=plotly&logoColor=white)
![Seaborn](https://img.shields.io/badge/Seaborn-74a9cf?style=for-the-badge&logo=python&logoColor=white)

## 🔍 Objectives

- Understand how **obesity and inactivity** relate to **diabetes rates**.
- Build and evaluate **linear regression** and **polynomial regression** models.
- Analyze model performance using **R²**, **Mean Absolute Error**, and **K-Fold cross-validation**.
- Identify limitations in dataset and modeling assumptions.
- Derive actionable public health insights.

## 🧠 Key Findings

### 📈 Correlation Analysis
- **Obesity ↔ Inactivity Correlation:** Moderate positive correlation (**0.47**)
- **Diabetes ↔ Obesity/Inactivity:** Also moderately positive; not strong enough for highly predictive modeling

### 📉 Regression Analysis
| Model Type             | R² Score | Interpretation                               |
|------------------------|----------|----------------------------------------------|
| Linear Regression      | ~0.50    | Explains ~50% of diabetes variance           |
| Polynomial Regression  | Varies   | Slight improvement over linear in some cases |

- **Scatter plots with regression lines** help visually assess model fit.
- **Polynomial regression** captured non-linearity slightly better than linear regression.

### ⚠️ Limitations Highlighted by Results
- R² ≈ 0.50 suggests **many other factors influence diabetes** (e.g., diet, genetics, income)
- Models are useful for **trend analysis**, but **not precise prediction**

## 📊 Methodology

1. **Data Preprocessing:**
   - Cleaned and merged relevant columns
   - Split into **training/testing datasets**

2. **Exploratory Data Analysis (EDA):**
   - Computed **descriptive stats**, **skewness**, **kurtosis**
   - Plotted **histograms**, **scatterplots**, **correlation matrix**

3. **Model Building:**
   - Applied **linear regression** using `sklearn.linear_model.LinearRegression`
   - Built **polynomial regression models** with `PolynomialFeatures`
   - Evaluated using **R²** and **cross-validation**

4. **Interpretation:**
   - Used **visualizations** and **statistical outputs** to support conclusions

## 📈 Outputs

- ✅ Regression scatterplots (linear & polynomial)
- ✅ Summary statistics table
- ✅ Correlation matrix heatmap
- ✅ R² scores vs polynomial degree
- ✅ Cross-validation results

## 🎯 Public Health Implications

✅ Moderate correlation suggests that **lifestyle interventions** (weight loss, physical activity) can be impactful  
✅ Policymakers should consider **multi-factor health models** when designing community outreach  
✅ The model can be improved by adding **diet, age, ethnicity, and genetic history**

## 📌 Future Scope

- Integrate **multi-year data** and **larger sample size**
- Include **socioeconomic** and **demographic variables**
- Use **logistic regression** or **classification models** for binary diabetes prediction
- Deploy as an **interactive dashboard** for real-time analysis

🚀 **Using predictive analytics to fight health disparities and shape better public health outcomes.**
