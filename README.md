## 🧩 Insurance Cost Analysis (Python, Scikit-learn)

### 🎯 **Project Objective**

Predict individual **medical insurance charges** using demographic and lifestyle data such as **age**, **BMI**, **smoking status**, and **region**.
This project demonstrates how polynomial regression can capture non-linear relationships and significantly improve prediction accuracy.

---

### 📂 **Dataset**

**Source:**
[Medical Insurance Dataset](https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DA0101EN-Coursera/medical_insurance_dataset.csv)

**File Path Used in Project:**

```python
filepath = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DA0101EN-Coursera/medical_insurance_dataset.csv'
```

**Columns:**
`age`, `sex`, `bmi`, `children`, `smoker`, `region`, `charges`

**Target Variable:**
`charges` – represents the medical insurance cost.

---

### ⚙️ **Tools & Libraries Used**

* Python 🐍
* Pandas, NumPy – data cleaning and manipulation
* Matplotlib, Seaborn – visualization
* Scikit-learn – modeling and pipeline creation

---

### 🧹 **Data Preparation**

• Handled missing values and verified data types
• Encoded categorical features (`sex`, `smoker`, `region`) using `pd.get_dummies()`
• Split dataset into **training** and **testing** subsets
• Scaled numeric features using `StandardScaler()`

---

### 🧠 **Modeling Steps**

• Built a **Linear Regression** baseline model
• Created a **Pipeline** including:

* `StandardScaler()` – normalizes data
* `PolynomialFeatures(degree=2)` – adds squared and interaction terms
* `LinearRegression()` – fits the final model
  • Trained and tested on the dataset
  • Achieved an **R² Score ≈ 0.85**, showing strong model accuracy

---

### 📊 **Results & Insights**

* Polynomial regression captured complex relationships (e.g., between BMI and smoker status)
* The model explains ~85% of the variance in insurance charges
* **Smoker** is the most significant feature influencing medical costs
* Visualizations show predicted vs. actual charges closely aligned with the diagonal (perfect prediction line)

---

### 🖼 **Visualization Example**

```python
plt.figure(figsize=(6,6))
plt.scatter(Y_test, Y_pred_poly, alpha=0.6, color='teal')
plt.plot([Y_test.min(), Y_test.max()], [Y_test.min(), Y_test.max()],
         color='red', linestyle='--', linewidth=2, label='Perfect Prediction')
plt.xlabel("Actual Charges")
plt.ylabel("Predicted Charges")
plt.title("Actual vs Predicted Medical Charges (Polynomial Regression)")
plt.legend()
plt.show()
```

---

### 🧾 **Conclusion**

Polynomial Regression (degree=2) with feature scaling improved prediction accuracy from ~0.75 to **0.85 R²**, confirming the importance of capturing **non-linear interactions** in real-world cost prediction.


