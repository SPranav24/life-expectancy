
# 🌍 Life Expectancy Analysis: A Data Science Approach

## 📊 Overview
This project analyzes global life expectancy data to understand how health metrics, socioeconomic factors, and education impact lifespan across countries. It includes data preprocessing, visualization, hypothesis testing (t-test & bootstrapping), and predictive modeling using ensemble regressors.

---

## 🎯 Objectives

1. **Quantify Health Impacts**  
   Explore how adult mortality, HIV/AIDS prevalence, and infant deaths influence life expectancy.

2. **Analyze Socioeconomic Factors**  
   Study the effect of GDP, health expenditure, and income composition on longevity.

3. **Assess Educational Influence**  
   Investigate the correlation between years of schooling and life expectancy.

4. **Compare Developed vs. Developing Countries**  
   Use statistical testing to measure differences in life expectancy between economic groups.

---

## 🧠 Key Techniques Used

- **Data Preprocessing:** Handling missing values, scaling, encoding.
- **Visualization:** Boxplots, scatter plots, histograms, correlation heatmaps.
- **Descriptive Statistics:** Variance, correlation, distribution analysis.
- **Hypothesis Testing:**  
  - Independent t-test  
  - Bootstrapping Confidence Intervals  
  - A/B Testing logic
- **Regression Models:**  
  - Linear Regression  
  - Random Forest  
  - Gradient Boosting  
  - Support Vector Regressor  
  - Voting Regressor (ensemble)

---

## 📁 Dataset

- **Source:** `Life Expectancy Data.csv`  
- **Features:**  
  - Health indicators (e.g., Adult Mortality, Alcohol use)  
  - Economic indicators (GDP, Total expenditure)  
  - Education (Schooling)  
  - Status (Developed / Developing)  
  - Target: Life expectancy (in years)

---

## 📈 Model Performance

| Metric | Value |
|--------|-------|
| Mean Squared Error (MSE) | ~Low |
| Mean Absolute Error (MAE) | ~Low |
| R² Score | ~High |

> Ensemble regression achieved robust results indicating strong predictive power.

---

## ✅ Key Insights

- Developed countries have **significantly higher life expectancy**.
- **Education** and **income** show a **positive correlation** with lifespan.
- **Infant mortality** and **adult mortality** negatively impact longevity.
- **Policy implication:** Invest in healthcare and education to improve national life expectancy.

---

## 🚧 Limitations

- Missing or imputed data may affect accuracy.
- Results may not capture intra-country regional disparities.
- Assumptions like independence in A/B testing may not hold perfectly.

---

## 🛠️ How to Run

1. Ensure the dataset (`Life Expectancy Data.csv`) is in the same directory.
2. Run the script `ed1faaf5-880a-4ce0-aa10-21a6173da0f3.py` in a Python environment.
3. Dependencies:
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn
   ```

---

## 📌 Author

**Sai Pranav**  
Final Year Student | Data Science Enthusiast  
*Exploring the intersection of health and data.*
