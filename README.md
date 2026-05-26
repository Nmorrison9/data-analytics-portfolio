# 📊 Data Analytics Portfolio

A collection of intermediate data analytics projects covering credit risk modeling, SQL data exploration, and climate trend analysis.
Built using Python, pandas, scikit-learn, seaborn, matplotlib, and SQLite.

---

## **Credit Risk Analysis**

**Overview** — Builds and evaluates a Random Forest model to predict loan defaults. Addresses class imbalance using SMOTE, optimizes the decision threshold to reduce false negatives, and complements modeling with exploratory analysis of credit score distributions, default trends, and feature correlations.

**Dataset**
- Synthetic data generated via generate_data.py
- 5,000 rows, 13 columns
- Includes borrower demographics, loan details, and default status

**Methods** — Random Forest Classifier, SMOTE, pandas, seaborn, matplotlib, scikit-learn

**Key Findings**
- **Portfolio Risk Overview** — Identified a 7.76% default rate, establishing a baseline for risk benchmarking
- **Key Risk Drivers** — Credit score (-0.22) and debt-to-income ratio (0.24) are the strongest predictors of default
- **Behavioral Insights** — Higher interest rates correlate with increased default likelihood; default rates drop from 12.5% to 0.9% as credit scores improve
- **Model Performance** — Improved recall on default cases to 50% using SMOTE and a 0.3 decision threshold

**How to Run**
1. pip install pandas matplotlib seaborn scikit-learn imbalanced-learn
2. Run generate_data.py from the root directory
3. Run:
   - credit_risk/credit_risk_eda.py
   - credit_risk/credit_risk_model.py

---

## **SQL Exploration**

**Overview** — Connects to a SQLite retail database and runs analytical queries to evaluate revenue by category, customer behavior, product utilization, and order trends.

**Dataset**
- Synthetic data generated via generate_data.py
- customers: 1,000 rows, 7 columns
- products: 200 rows, 7 columns
- orders: 8,000 rows, 5 columns
- order_items: 15,000 rows, 5 columns

**Methods** — SQLite, SQL joins, aggregations, subqueries, strftime, pandas, matplotlib

**Key Findings**
- **Revenue Distribution** — Food, Books, and Home & Garden lead (~$1.7M–$1.8M); Electronics and Sports lag (~$1.35M–$1.41M)
- **Customer Insights** — ~$9K gap between top and 10th spender indicates concentration of high-value customers
- **Inventory Efficiency** — 100% product utilization with no dead stock
- **Trends & Risks** — Stable revenue with biannual peaks; identified a September 2023 dip and 10.36% cancellation rate

**How to Run**
1. pip install pandas matplotlib
2. Run generate_data.py
3. Run:
   sql_exploration/retail_exploration.sql.py

---

## **Climate Analysis**

**Overview** — Analyzes climate and energy data (1990–2023) to identify trends in CO2 emissions, renewable energy adoption, temperature change, and GDP relationships.

**Dataset**
- Synthetic data generated via generate_data.py
- 340 rows, 8 columns
- 10 countries across 34 years (1990–2023)

**Methods** — pandas, matplotlib, seaborn, line plots, scatter plots, regression analysis

**Key Findings**
- **Renewable Energy Growth** — Increased across all countries; UK and Canada lead adoption
- **Emissions Benchmarking** — Japan, Germany, and the UK have the lowest CO2 emissions
- **Climate Trends** — Temperature changes are gradual over time, reflecting long-term patterns
- **Economic Trade-offs** — Wealthier countries maintain high emissions due to industrial activity and energy demands

**How to Run**
1. pip install pandas matplotlib seaborn
2. Run generate_data.py
3. Run:
   data_viz/climate_analysis.py
