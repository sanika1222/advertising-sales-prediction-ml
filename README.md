# Advertising Sales Prediction Pipeline 📈

An end-to-end Machine Learning web application designed to optimize advertising budgets. The application leverages a **Linear Regression** model trained on multichannelling expenditure allocations to reliably predict cross-product item unit sales.

## 🧮 Model Interpretations & Context

### 1. Which advertising factors affect sales?
* Based on the feature coefficients generated during model training, **TV** and **Digital Advertising** showcase the highest positive coefficients, making them the primary drivers for sales expansion.
* **Newspaper advertising** typically exhibits minimal to negligible correlation weights, highlighting low marketing conversion yield.

### 2. What does the \(R^2\) Score mean?
* The \(R^2\) score (Coefficient of Determination) represents the proportion of variance in the target variable (`Sales`) that can be predicted from our features. 
* For example, an \(R^2\) score of `0.88` implies that **88% of the fluctuation in product sales** is directly explained by the variations in our combined advertising channel expenditures.

### 3. How accurate is the model?
* Model accuracy is evaluated using **MAE** and **RMSE** benchmarks. An RMSE value of `1.20` indicates that on average, our system's target predictions deviate by roughly 1,200 item sales units from actual values.

### 4. Machine Learning Limitations
* **Linearity Presumption:** The underlying math assumes linear trends, neglecting potential diminishing returns models where heavy spending on a single channel halts conversions.
* **Omission of Seasonality:** External holiday surges, competitor behavior patterns, and shifting macro economics are not factored in.

---

## 🛠️ Step-by-Step Instructions to Run This App

### Prerequisites
Make sure python (3.8+) is configured on your environment framework.

### 1. Initialize Virtual Environment & Install Requirements
```bash
# Clone or open the project folder
cd Advertising-Sales-Prediction

# Create and boot virtual environment
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate

# Install application dependencies
pip install -r requirements.txt
```

### 2. Execute Training Pipeline
Compile and serialize the scaling transforms and linear calculations:
```bash
python train.py
```

### 3. Launch the Streamlit Interface Dashboard
```bash
streamlit run app.py
```
