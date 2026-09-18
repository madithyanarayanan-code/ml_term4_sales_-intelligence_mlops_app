# Sales Customer Intelligence — MLOps Customer Analyser

An end-to-end Streamlit application that turns raw sales CSV files into customer intelligence: data quality checks, RFM behavioural segmentation, churn-risk modelling, clustering, demographics, country-level geographic analysis, association rules, model metrics and an executive report.

## Live deployment

Deploy this repository with Streamlit Community Cloud. The app is designed to run without private API keys. Llama/Ollama report polishing is optional; if Ollama is unavailable, the application generates a deterministic report instead.

## Pipeline

The project follows the defined eight-stage sequence:

1. Project Management
2. Data Ingestion
3. Data Understanding
4. Preprocessing
5. Feature Engineering
6. Model Proposal
7. Model Training
8. Model Evaluation

## What the app analyses

- CSV schema mapping and data-quality diagnostics
- Numeric/date cleaning and transaction consistency checks
- EDA and sales trends
- RFM customer profiles
- Behavioural customer sets: Champions, Loyal High Value, Potential Loyalists, New / Promising, High Value At Risk, Loyal At Risk, Hibernating and Regular / Needs Nurture
- Leakage-resistant forward 60-day churn modelling using historical snapshots
- Classification metrics: accuracy, precision, recall, F1, ROC-AUC, PR-AUC, Brier score and confusion matrix
- Customer clustering using behavioural variables, with demographics used for profiling rather than arbitrary encoding
- KMO and Bartlett tests
- Regression with R², adjusted R², RMSE and MAE
- Product association rules
- Demographic and country analysis
- Country-centroid revenue heatmap
- Management recommendations
- Markdown and DOCX report generation
- Separate developer diagnostics page

## Run locally

```bash
python -m venv .venv
# Windows PowerShell (if script activation is blocked, skip activation and use .venv/Scripts/python.exe)
.venv\\Scripts\\python.exe -m pip install -r requirements.txt
.venv\\Scripts\\python.exe -m streamlit run app.py
```

## Deploy on Streamlit Community Cloud

1. Create a public GitHub repository.
2. Upload the files in this repository to the repository root.
3. Open Streamlit Community Cloud and connect the GitHub repository.
4. Select `app.py` as the main file.
5. Deploy.

No secrets are required for the core application.

### Optional Llama integration

The report generator can use a locally reachable Ollama endpoint through:

```text
OLLAMA_URL
OLLAMA_MODEL
```

For public cloud deployment, do not assume a local Ollama server is available. The app automatically falls back to its deterministic report generator.

## Data privacy

Uploaded CSV data is processed by the running Streamlit instance. Do not upload confidential customer information to a public deployment. Use anonymised or synthetic data for demonstrations.

## Project structure

```text
.
├── app.py
├── customer_analyzer_backend.py
├── report_generator.py
├── dev_backend_report.py
├── pages/
│   └── Dev.py
├── requirements.txt
├── .streamlit/config.toml
├── docs/blog.md
├── LICENSE
└── .gitignore
```

## Limitations

This is an analytical demonstration project. Churn probabilities depend on the supplied transaction history and should be validated against real business outcomes before operational use. Country heatmaps represent country-centroid revenue concentration and are not precise customer-location maps.
#
