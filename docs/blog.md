# From Raw Sales Data to Customer Intelligence: Building an End-to-End MLOps Sales Analyser

**Author: Adithya**  
**Project: Sales Customer Intelligence**

## The problem

Sales datasets often contain much more than revenue totals. Hidden inside transaction history are signals about customer loyalty, purchase frequency, spending behaviour, product combinations, churn risk and geographic concentration.

The challenge is turning those raw records into a repeatable analytical workflow rather than a collection of disconnected notebooks.

This project builds a Streamlit-based customer intelligence application around that problem. A user uploads a sales CSV and the application moves through an eight-stage analytical lifecycle: ingestion, understanding, preprocessing, feature engineering, model proposal, training and evaluation, while keeping the implementation details separate from the business-facing interface.

## What I wanted the application to answer

The application was designed around five practical questions:

1. What is happening in the sales data?
2. Which customers behave similarly?
3. Which customers show evidence of churn risk?
4. Where and among which demographic groups is customer value concentrated?
5. What actions can be considered from the observed evidence?

## Architecture

The workflow follows eight stages:

**1. Project Management → 2. Data Ingestion → 3. Data Understanding → 4. Preprocessing → 5. Feature Engineering → 6. Model Proposal → 7. Model Training → 8. Model Evaluation**

The Streamlit interface intentionally hides the implementation logic from the main user views. A separate Dev page exposes concise diagnostics such as schema mapping, KMO/Bartlett results, clustering output, model metrics, association diagnostics and governance information.

## Stage 1–4: Getting trustworthy data

The first challenge is that sales files are rarely perfectly clean. The application therefore maps common source column names to canonical concepts such as `CustomerID`, `InvoiceDate`, `InvoiceNo`, `Product`, `Quantity`, `UnitPrice` and `TotalAmount`.

The preprocessing layer handles common numeric and date problems, checks duplicates and compares reported transaction amounts against `Quantity × UnitPrice`. Suspicious records are flagged rather than silently rewritten.

This distinction matters: cleaning should improve analytical consistency without hiding data-quality problems.

## Stage 5: Turning transactions into customer behaviour

The transaction table is transformed into customer-level features including:

- Recency
- Frequency
- Monetary value
- Average order value
- Total quantity
- Unique products
- Average items per order
- Customer tenure

These features support two complementary views of customer behaviour.

### Business-readable behavioural sets

RFM scoring creates actionable customer sets:

| Behavioural set | Interpretation | Example action area |
|---|---|---|
| Champions | Recent, frequent and high-value | Retention and rewards |
| Loyal High Value | Strong repeat behaviour and value | Share-of-wallet growth |
| Potential Loyalists | Recent customers with growth potential | Second/third purchase activation |
| New / Promising | Recent but limited history | Onboarding |
| High Value At Risk | High historical value with inactivity | Win-back |
| Loyal At Risk | Previously frequent but inactive | Reactivation |
| Hibernating | Low engagement | Low-cost reactivation |
| Regular / Needs Nurture | Middle behavioural range | Personalised nurture |

These names are deliberately separate from machine-learning cluster IDs. A K-Means label such as `Cluster 2` has no inherent business meaning; it must be profiled against its underlying behaviour before interpretation.

### Statistical customer clustering

K-Means is used as an independent exploratory method on behavioural purchase variables. Demographic attributes are then used to profile the resulting clusters rather than being arbitrarily converted into numeric codes and mixed into the distance calculation.

The cluster view uses a bubble chart to make the behavioural differences visible. Frequency can be placed on one axis, average spend on the other and customer count can determine bubble size.

## Demographics and geography

Customer behaviour becomes more useful when it can be profiled by age, gender, income, city and country.

The application therefore joins demographic attributes to customer profiles and produces country-by-behaviour views. A geographic heatmap shows country-level revenue concentration using country centroids.

The map is intentionally labelled as a country-level analytical visual. It should not be interpreted as precise customer coordinates.

## Stage 6–8: From modelling to evaluation

The project contains both predictive and exploratory modelling.

### Churn modelling

An important design correction was avoiding target leakage. Defining churn directly from current recency and then using that same recency to predict churn would make the evaluation misleading.

Instead, historical customer snapshots are used to create a forward-looking target: whether a customer makes no purchase during the following 60-day period. The model is evaluated on a later time period where possible.

Reported classification metrics include:

- Accuracy
- Precision
- Recall
- F1
- ROC-AUC
- PR-AUC
- Brier score
- Confusion matrix

The current customer population can then receive churn-risk probabilities after the model is refit on historical data.

### Regression

The application also evaluates customer-level future monetary value using regression candidates. It reports:

- R²
- Adjusted R²
- RMSE
- MAE

These metrics provide different views of explanatory power and prediction error rather than relying on one number.

### Association analysis

The transaction structure preserves invoice-level product baskets, allowing association-rule mining to identify products that frequently occur together. Support, confidence and lift help describe these relationships.

## Why MLOps matters here

The project is not just a model wrapped in a UI. The workflow separates raw data, cleaned data, engineered features, model candidates and evaluation outputs conceptually across the eight lifecycle stages.

The developer view also provides a concise audit surface for:

- schema mapping
- test results
- cluster configuration
- model comparison
- association diagnostics
- governance and lineage

This makes it easier to demonstrate how an analytical result was produced rather than showing only the final chart.

## The Streamlit experience

The business-facing interface contains:

**Overview** — executive KPIs and sales signals  
**Sales & EDA** — trends, distributions and data-quality checks  
**Customers & Churn** — customer risk and RFM views  
**Behavioral Segments** — actionable customer groups  
**Clusters** — exploratory K-Means analysis and bubble/bar visualisation  
**Demographics & Geo** — demographic profiling and country heatzones  
**Associations** — product affinities  
**Models & Metrics** — predictive model performance  
**Recommendations** — evidence-linked actions  
**Report** — Markdown/DOCX report generation

## Deployment

The application is structured as a normal GitHub repository with a `requirements.txt`, `.gitignore`, Streamlit configuration and documentation. It can be deployed using Streamlit Community Cloud with `app.py` as the entry point.

The Llama integration is optional. The report generator attempts to use an Ollama endpoint when configured, but the application has a deterministic fallback so the public demo does not depend on a private local model server.

## Responsible use and limitations

The project uses synthetic/demo sales data for demonstration. Churn probabilities are analytical estimates, not ground truth. A production implementation should validate predictions against observed future outcomes, review class imbalance and threshold costs, monitor drift and establish appropriate data governance.

Geographic heatzones use country centroids and therefore communicate country-level concentration rather than exact customer locations.

## What I learned

The main lesson was that a customer analytics application becomes more useful when the analytical layers have distinct purposes:

**RFM personas explain behaviour.**  
**K-Means explores natural similarity.**  
**Classification estimates future churn risk.**  
**Association rules reveal product relationships.**  
**Demographics and geography explain where those behaviours occur.**

Putting all of those layers into one reproducible pipeline creates a stronger bridge between machine learning, business analysis and deployment.

## Repository

The repository contains the Streamlit frontend, Python backend, report generator, developer diagnostics, deployment configuration and this project write-up.
