#!/usr/bin/env python
# coding: utf-8

# # Objectives:
# 
# **1.Quantify the Impact of Health Metrics on Life Expectancy:**
# 
# ->Investigate the relationship between Adult Mortality, HIV/AIDS prevalence, and infant deaths with life expectancy.
# 
# ->Hypothesis: Higher adult mortality and infant death rates negatively affect life expectancy.
# 
# **2.Assess Socioeconomic Factors Influencing Life Expectancy:**
# 
# ->Evaluate the effect of GDP, percentage expenditure, and Income composition of resources.
# 
# ->Hypothesis: Countries with higher GDP and income composition exhibit longer life expectancy.
# 
# **3.Study the Role of Education:**
# 
# ->Examine how Schooling influences life expectancy.
# 
# ->Hypothesis: Greater access to education correlates positively with life expectancy.
# 
# **4.Compare Developed and Developing Countries:**
# 
# ->Use the Status variable to compare life expectancy trends between developed and developing nations.
# 
# ->Hypothesis: Developed countries have significantly higher life expectancy due to better healthcare and infrastructure.
# 
# 
# # Real-World Significance:
# 
# ->This case study is essential to:
# 
# ->Identify actionable insights for public health policy.
# 
# ->Help governments prioritize spending on education, healthcare, and economic development.
# 
# ->Evaluate the effectiveness of vaccination and nutrition programs.
# 
# ->Understand disparities between developed and developing nations to address global health inequities.

# In[45]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


# In[46]:


data=pd.read_csv("Life Expectancy Data.csv")


# In[47]:


data.head()


# In[48]:


data.shape


# # Handling missing values

# In[49]:


print(data.isnull().sum())


# In[50]:


data["Life expectancy "].fillna(data["Life expectancy "].mean(),inplace=True)


# In[51]:


data["Adult Mortality"].fillna(data["Adult Mortality"].mean(),inplace=True)


# In[52]:


data[' BMI '].fillna(data[' BMI '].mean(), inplace=True)


# In[53]:


num_cols=['Total expenditure', 'GDP', 'Population', ' thinness  1-19 years', 
       ' thinness 5-9 years', 'Income composition of resources', 'Schooling','Alcohol']

data[num_cols] = data[num_cols].fillna(data[num_cols].mean())

cat_cols = ['Hepatitis B', 'Polio', 'Diphtheria ']

data[cat_cols] = data[cat_cols].apply(lambda x: x.fillna(x.mode()[0]))


# # Data Transformation

# In[54]:


from sklearn.preprocessing import StandardScaler
one_hot_encoded_data = pd.get_dummies(data, columns=['Status'], prefix='Status')
one_hot_encoded_data=one_hot_encoded_data.drop("Country",axis=1)
scaler = StandardScaler()
data2 = pd.DataFrame(scaler.fit_transform(one_hot_encoded_data), columns=one_hot_encoded_data.columns)

data2


# In[55]:


plt.figure(figsize=(15, 12))
for i, col in enumerate(num_cols, 1):
    plt.subplot(3, 3, i)
    sns.boxplot(data2[col])
    plt.title(col)
    plt.tight_layout()


# # Data Visualisation

# In[56]:


plt.scatter(data2["Life expectancy "],data2["Adult Mortality"],marker='o',edgecolor="red")


# In[57]:


plt.scatter(data2["Life expectancy "],data2["Alcohol"],marker='o',edgecolor="red")


# In[58]:


data2.hist(figsize=(15,12), bins=30)
plt.suptitle("Distribution of Features")
plt.show()


# # Descriptive Analysis

# In[59]:


columns_to_analyze = ['Life expectancy ', 'GDP', 'Schooling', 'Adult Mortality', 'Alcohol']

filtered_data = data2[columns_to_analyze].dropna()

descriptive_stats = filtered_data.describe().T
descriptive_stats['variance'] = filtered_data.var()  # Add variance manually
print("Descriptive Statistics:")
print(descriptive_stats)

correlation_matrix = filtered_data.corr()
print("\nCorrelation Matrix:")
print(correlation_matrix)

plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm", cbar=True)
plt.title('Correlation Matrix Heatmap', fontsize=14)
plt.show()


# ->Life Expectancy and Education (Schooling): Strong positive correlation (approx. 0.7).                                        
# ->Life Expectancy and GDP: Moderate positive correlation, though influenced by outliers.                                       
# ->Infant Deaths and Life Expectancy: Strong negative correlation, underscoring the importance of child survival rates.

# **Hypothesis:**                                                                                                                 
# Developed countries have a significantly higher life expectancy compared to developing countries.
# 
# Null Hypothesis (𝐻0):There is no difference in life expectancy between developed and developing countries.
# 
# Alternative Hypothesis:(𝐻𝑎): Developed countries have a higher life expectancy than developing countries.
# 

# In[63]:


import numpy as np
from scipy.stats import ttest_ind

filtered_data = data[['Status', 'Life expectancy ']].dropna()
developed = filtered_data[filtered_data['Status'] == 'Developed']['Life expectancy ']
developing = filtered_data[filtered_data['Status'] == 'Developing']['Life expectancy ']
sns.set(style="whitegrid")

plt.figure(figsize=(9, 6))
sns.histplot(bootstrap_diff, bins=30, kde=True, color='skyblue', stat='density')

plt.axvline(ci_lower, color='red', linestyle='--', label=f'2.5% CI: {ci_lower:.2f}')
plt.axvline(ci_upper, color='green', linestyle='--', label=f'97.5% CI: {ci_upper:.2f}')

plt.title('Bootstrap Distribution of Mean Differences', fontsize=16)
plt.xlabel('Mean Difference (Developed - Developing)', fontsize=14)
plt.ylabel('Density', fontsize=14)

plt.legend()

plt.tight_layout()
plt.show()

t_stat, p_value = ttest_ind(developed, developing, equal_var=False) 
print("T-Test Results:")
print(f"T-Statistic: {t_stat:.2f}")
print(f"P-Value: {p_value:.5f}")

if p_value < 0.05:
    print("Reject the null hypothesis: Developed countries have significantly higher life expectancy.")
else:
    print("Fail to reject the null hypothesis: No significant difference in life expectancy.")

np.random.seed(42) 
n_bootstrap = 1000
bootstrap_diff = []

for _ in range(n_bootstrap):
    boot_developed = np.random.choice(developed, size=len(developed), replace=True)
    boot_developing = np.random.choice(developing, size=len(developing), replace=True)
    bootstrap_diff.append(np.mean(boot_developed) - np.mean(boot_developing))

ci_lower = np.percentile(bootstrap_diff, 2.5)
ci_upper = np.percentile(bootstrap_diff, 97.5)
print(f"\nBootstrap Confidence Interval for Mean Difference: [{ci_lower:.2f}, {ci_upper:.2f}]")

if ci_lower > 0:
    print("The confidence interval does not include 0, supporting the conclusion that developed countries have higher life expectancy.")
else:
    print("The confidence interval includes 0, suggesting no significant difference between groups.")


# **Why Use Bootstrapping?**
# 
# ->Bootstrapping helps estimate the sampling distribution of a statistic without assuming a specific data distribution.             
# ->Useful for creating confidence intervals when the underlying population distribution is unknown or sample size is small.
# 
# **Results interpretation :**
# 
# ->Developed countries have a significantly higher life expectancy compared to developing countries. The evidence is strong enough to rule out random chance as the explanation for the observed difference.
# 
# ->The bootstrapping results confirm the findings of the t-test, providing robust evidence for the difference in life expectancy.
# 
# **A/B Testing :**
# 
# Objective:
# To conduct an A/B test comparing two groups and evaluating if the difference in life expectancy is statistically significant.
# 
# Significance Level:
# α=0.05
# 
# Implementation: The t-test conducted earlier serves as the A/B test. 
# The results (p=0.00000) indicate that the null hypothesis is rejected, confirming a significant difference.
# 
# Bootstrap Validation: The bootstrapped confidence interval 
# [11.58,12.58] adds further evidence to the A/B test results, ensuring that the observed difference is not due to sampling error or distributional assumptions.

# # Model Selection
# 
# **Regression :**

# In[61]:


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

data = data2.dropna(subset=['Life expectancy '])

X = data.drop(['Life expectancy ','Year'], axis=1)
y = data['Life expectancy ']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
categorical_features = X.select_dtypes(include=['object', 'category']).columns

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(drop='first'), categorical_features)
    ])

lr = LinearRegression()
rf = RandomForestRegressor(random_state=42)
svr = SVR()
gb = GradientBoostingRegressor(random_state=42)

lr_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', lr)
])

rf_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', rf)
])

svr_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', svr)
])

gb_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', gb)
])

ensemble = VotingRegressor(estimators=[
    ('lr', lr_pipeline),
    ('rf', rf_pipeline),
    ('svr', svr_pipeline),
    ('gb', gb_pipeline)
])

ensemble.fit(X_train, y_train)


y_pred = ensemble.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error (MSE): {mse:.4f}')
print(f'Mean Absolute Error (MAE): {mae:.4f}')
print(f'R-squared (R2 Score): {r2:.4f}')


# # Conclusion :
# 
# 
# # 1. Summary of Findings
# ->The analysis shows a statistically significant difference in life expectancy between developed and developing countries:      
# ->T-Test Results: Developed countries have significantly higher life expectancy than developing countries (p<0.05).              
# 
# **Bootstrapping Confidence Interval:**                                                                                        
# ->The mean difference in life expectancy is estimated to be between 11.58 and 12.58 years, confirming the significance of the difference.                                                                                                                     
# ->Hypothesis Supported: The findings strongly support the hypothesis that developed countries exhibit higher life expectancy due to better socioeconomic conditions, healthcare, and education.                                                                 
# 
# # 2. Implications                                                                                                                
# **For Policymakers:**                                                                                                           
# ->Investments in education and healthcare infrastructure in developing countries can substantially improve life expectancy.     
# ->Efforts to reduce child mortality and improve access to vaccines may significantly close the life expectancy gap.             
# **Global Health Perspective:**                                                                                                 
# ->The disparities in life expectancy highlight inequalities in resources and healthcare access between nations.                 
# ->International aid and development programs should prioritize these areas to foster global equity in health outcomes.            
# # 3. Limitations                                                                                                                 
# **Data Limitations:**                                                                                                           
# ->Missing values in variables like GDP and Hepatitis B coverage may introduce biases.                                          
# ->Lack of granular data for specific regions within countries (e.g., urban vs. rural) could obscure intra-country disparities.   
# **Bootstrapping Limitations:**                                                                                                 
# ->Bootstrapping assumes that the sample is representative of the population. If the sample is biased, the confidence interval may not accurately reflect the true mean difference.                                                                           
# **A/B Testing Limitations:**                                                                                                   
# ->A/B testing with t-tests assumes independent observations. If countries within the same region share common factors, this could violate the independence assumption.                                                                                     
# ->Does not account for confounding variables like population density or cultural differences.
# 

# In[ ]:




