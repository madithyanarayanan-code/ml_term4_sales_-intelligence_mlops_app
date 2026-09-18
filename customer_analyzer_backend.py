"""
Sales Data Analyser - backend pipeline.
Pipeline intentionally mirrors the supplied MLOps lifecycle:
1 Project Management -> 2 Data Ingestion -> 3 Data Understanding ->
4 Preprocessing -> 5 Feature Engineering -> 6 Model Proposal ->
7 Model Training -> 8 Model Evaluation.

The UI should call run_pipeline(); all backend diagnostics are returned in `dev`.
"""
from __future__ import annotations

import io
import json
import math
import os
import re
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score, adjusted_rand_score, classification_report, confusion_matrix,
    average_precision_score, brier_score_loss, f1_score, mean_absolute_error, mean_squared_error, precision_score, r2_score,
    recall_score, silhouette_score, roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# ----------------------------- helpers -----------------------------

def _json_safe(x: Any) -> Any:
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x) if np.isfinite(x) else None
    if isinstance(x, (pd.Timestamp, datetime)):
        return x.isoformat()
    if isinstance(x, dict):
        return {str(k): _json_safe(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_json_safe(v) for v in x]
    if pd.isna(x):
        return None
    return x


def _numeric_clean(series: pd.Series) -> pd.Series:
    """Parse financial/quantity fields with common broken-data patterns."""
    def parse(v):
        if pd.isna(v):
            return np.nan
        s = str(v).strip()
        if not s:
            return np.nan
        # Common OCR/data-entry confusion.
        s = s.replace("O", "0").replace("o", "0")
        s = s.replace("₹", "").replace("$", "").replace("€", "")
        s = re.sub(r"\s+", "", s)
        # Handle 1.234,56 and 1234,56; preserve normal 1234.56.
        if "," in s and "." in s:
            if s.rfind(",") > s.rfind("."):
                s = s.replace(".", "").replace(",", ".")
            else:
                s = s.replace(",", "")
        elif "," in s:
            tail = s.split(",")[-1]
            if len(tail) in (1, 2, 3) and s.count(",") == 1:
                s = s.replace(",", ".")
            else:
                s = s.replace(",", "")
        s = re.sub(r"[^0-9.\-]", "", s)
        if s in ("", "-", "."):
            return np.nan
        try:
            return float(s)
        except Exception:
            return np.nan
    return series.map(parse)


def _norm_name(x: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(x).lower())


ALIASES = {
    "CustomerID": ["customerid", "customer", "customer_id", "clientid", "client", "custid"],
    "InvoiceDate": ["invoicedate", "invoice_date", "date", "orderdate", "transactiondate", "transaction_date"],
    "TotalAmount": ["totalamount", "total_amount", "amount", "sales", "revenue", "salesamount", "netamount", "value"],
    "InvoiceNo": ["invoiceno", "invoice_no", "invoice", "orderno", "orderid", "transactionid", "transaction_id"],
    "Quantity": ["quantity", "qty", "units", "unitssold", "volume"],
    "Product": ["product", "productname", "description", "item", "itemname", "stockcode", "sku", "productid"],
    "UnitPrice": ["unitprice", "unit_price", "price", "sellingprice"],
    "Country": ["country", "region", "market", "location"],
    "City": ["city", "town", "locality"],
    "Age": ["age", "customerage", "ageyears"],
    "Gender": ["gender", "sex"],
    "Income": ["income", "annualincome", "salary", "householdincome"],
    "IncomeBand": ["incomeband", "incomegroup", "income_segment"],
    "Churn": ["churn", "churned", "is_churn", "churnflag", "attrition"],
}


def map_schema(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    cols = list(df.columns)
    norm_to_col = {_norm_name(c): c for c in cols}
    mapping = {}
    confidence = {}
    for canonical, aliases in ALIASES.items():
        found = None
        for alias in aliases:
            if _norm_name(alias) in norm_to_col:
                found = norm_to_col[_norm_name(alias)]
                break
        if found is None:
            # fuzzy substring fallback
            for c in cols:
                nc = _norm_name(c)
                if any(_norm_name(a) in nc or nc in _norm_name(a) for a in aliases):
                    found = c
                    break
        if found:
            mapping[canonical] = found
            confidence[canonical] = "high" if _norm_name(found) in [_norm_name(a) for a in aliases] else "medium"

    out = df.copy()
    reverse = {v: k for k, v in mapping.items()}
    out = out.rename(columns=reverse)

    # Derived/fallback fields. We do not overwrite existing user data.
    if "InvoiceNo" not in out.columns:
        out["InvoiceNo"] = np.arange(len(out)).astype(str)
        confidence["InvoiceNo"] = "derived"
    if "CustomerID" not in out.columns:
        # Keep customer analytics usable even with anonymous transaction data.
        out["CustomerID"] = out["InvoiceNo"].astype(str)
        confidence["CustomerID"] = "derived_from_invoice"
    if "InvoiceDate" not in out.columns:
        # No date = cannot do true temporal churn; a stable placeholder allows the app to continue.
        out["InvoiceDate"] = pd.Timestamp.today().normalize()
        confidence["InvoiceDate"] = "derived_placeholder"
    if "TotalAmount" not in out.columns and {"Quantity", "UnitPrice"}.issubset(out.columns):
        out["TotalAmount"] = _numeric_clean(out["Quantity"]) * _numeric_clean(out["UnitPrice"])
        confidence["TotalAmount"] = "derived_quantity_x_unitprice"
    if "Quantity" not in out.columns:
        out["Quantity"] = 1.0
        confidence["Quantity"] = "derived_default"
    if "Product" not in out.columns:
        out["Product"] = out["InvoiceNo"].astype(str)
        confidence["Product"] = "derived_invoice"
    if "TotalAmount" not in out.columns:
        raise ValueError("Could not map a sales amount column. Add a column such as TotalAmount, Sales, Revenue or Amount.")

    return out, {"mapping": mapping, "confidence": confidence}


# ----------------------------- stages -----------------------------

def project_management(df: pd.DataFrame, project_name: str = "Sales Customer Intelligence") -> Dict[str, Any]:
    return {
        "stage": 1,
        "project": project_name,
        "objective": "Customer segmentation, churn risk, sales diagnostics and basket insights",
        "status": "ready",
        "rows_received": int(len(df)),
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "governance": ["schema mapped", "raw data retained in memory", "train/test separation", "artifact-ready outputs"],
    }


def data_ingestion(raw_bytes: bytes) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    df = pd.read_csv(io.BytesIO(raw_bytes))
    mapped, schema = map_schema(df)
    return mapped, {
        "stage": 2,
        "source_type": "CSV",
        "rows": int(len(mapped)),
        "columns": int(len(mapped.columns)),
        "schema": schema,
        "raw_columns": list(df.columns),
    }


def data_understanding(df: pd.DataFrame) -> Tuple[Dict[str, Any], pd.DataFrame]:
    profile = {
        "stage": 3,
        "rows": int(len(df)),
        "columns": int(len(df.columns)),
        "duplicate_rows": int(df.duplicated().sum()),
        "missing_total": int(df.isna().sum().sum()),
        "numeric_columns": df.select_dtypes(include=np.number).columns.tolist(),
        "categorical_columns": df.select_dtypes(exclude=np.number).columns.tolist(),
        "date_min": _json_safe(df["InvoiceDate"].min()),
        "date_max": _json_safe(df["InvoiceDate"].max()),
        "summary": df.describe(include="all").transpose().reset_index().rename(columns={"index": "column"}),
        "missing_by_column": df.isna().sum().sort_values(ascending=False).reset_index(name="missing").rename(columns={"index": "column"}),
    }
    corr_cols = [c for c in ["TotalAmount", "Quantity", "UnitPrice"] if c in df.columns]
    corr_frame = pd.DataFrame({c: _numeric_clean(df[c]) for c in corr_cols})
    profile["correlations"] = corr_frame.corr() if len(corr_cols) >= 2 else pd.DataFrame()
    return profile, df


def preprocessing(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    out = df.copy()
    before = len(out)
    # Parse date and numerics.
    out["InvoiceDate"] = pd.to_datetime(out["InvoiceDate"], errors="coerce", dayfirst=False)
    for c in ["TotalAmount", "Quantity", "UnitPrice"]:
        if c in out.columns:
            out[c] = _numeric_clean(out[c])
    # Normalize IDs and product text.
    out["CustomerID"] = out["CustomerID"].astype(str).str.strip()
    out["InvoiceNo"] = out["InvoiceNo"].astype(str).str.strip()
    out["Product"] = out["Product"].astype(str).str.strip().replace({"nan": np.nan, "": np.nan})

    invalid_amount = int(out["TotalAmount"].isna().sum())
    invalid_dates = int(out["InvoiceDate"].isna().sum())
    out["TotalAmount"] = out["TotalAmount"].fillna(out["TotalAmount"].median())
    out["Quantity"] = out["Quantity"].fillna(1.0)
    out["Product"] = out["Product"].fillna("Unknown Product")
    out["InvoiceDate"] = out["InvoiceDate"].fillna(out["InvoiceDate"].median())
    out = out.drop_duplicates().reset_index(drop=True)
    # Transaction-level consistency check. Keep the supplied TotalAmount as the
    # commercial source of truth, but expose suspicious rows instead of silently
    # overwriting them.
    expected_amount = out["Quantity"] * out["UnitPrice"]
    out["AmountDifference"] = out["TotalAmount"] - expected_amount
    denom = expected_amount.abs().replace(0, np.nan)
    out["AmountMismatchPct"] = (out["AmountDifference"].abs() / denom).replace([np.inf, -np.inf], np.nan)
    out["AmountMismatchFlag"] = (out["AmountMismatchPct"] > 0.01).fillna(False)

    # Remove impossible negative/zero sales only from analytics copy; returns can still exist in source.
    out["AnalyticsAmount"] = out["TotalAmount"].clip(lower=0)

    # Winsorize extreme transaction amounts for clustering/model stability.
    q01, q99 = out["AnalyticsAmount"].quantile([0.01, 0.99])
    out["ModelAmount"] = out["AnalyticsAmount"].clip(q01, q99)

    diagnostics = {
        "stage": 4,
        "rows_before": before,
        "rows_after": int(len(out)),
        "duplicates_removed": int(before - len(out)),
        "numeric_parse_missing_filled": int(invalid_amount),
        "date_parse_missing_filled": invalid_dates,
        "outlier_bounds": {"p01": float(q01), "p99": float(q99)},
        "amount_mismatch_rows": int(out["AmountMismatchFlag"].sum()),
        "transformations": ["numeric coercion", "date parsing", "missing-value imputation", "duplicate removal", "transaction amount consistency check", "amount clipping for models"],
    }
    return out, diagnostics


def feature_engineering(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    ref_date = df["InvoiceDate"].max() + pd.Timedelta(days=1)
    g = df.groupby("CustomerID")
    rfm = g.agg(
        Recency=("InvoiceDate", lambda x: int((ref_date - x.max()).days)),
        Frequency=("InvoiceNo", "nunique"),
        Monetary=("AnalyticsAmount", "sum"),
        AvgOrderValue=("AnalyticsAmount", "mean"),
        TotalQuantity=("Quantity", "sum"),
        UniqueProducts=("Product", "nunique"),
    ).reset_index()
    rfm["AvgItemsPerOrder"] = rfm["TotalQuantity"] / rfm["Frequency"].replace(0, 1)
    rfm["CustomerTenureDays"] = (ref_date - g["InvoiceDate"].min()).dt.days.values
    # Churn definition: inactive beyond a robust business threshold; configurable in UI later if desired.
    data_span = max((df["InvoiceDate"].max() - df["InvoiceDate"].min()).days, 1)
    churn_window = int(max(60, min(120, round(data_span * 0.25))))
    rfm["Churn"] = (rfm["Recency"] > churn_window).astype(int)
    rfm["ChurnDefinition"] = f"Recency > {churn_window} days"

    # Behavioural RFM scoring: 5 is strongest, 1 is weakest.
    def quintile_score(s: pd.Series, higher_is_better: bool = True) -> pd.Series:
        ranks = s.rank(method="first")
        score = pd.qcut(ranks, 5, labels=[1,2,3,4,5], duplicates="drop")
        score = pd.to_numeric(score, errors="coerce").fillna(3).astype(int)
        return score if higher_is_better else 6 - score

    rfm["R_Score"] = quintile_score(rfm["Recency"], False)
    rfm["F_Score"] = quintile_score(rfm["Frequency"], True)
    rfm["M_Score"] = quintile_score(rfm["Monetary"], True)
    rfm["RFM_Score"] = rfm[["R_Score", "F_Score", "M_Score"]].sum(axis=1)

    def behavioural_set(row):
        R, F, M = row["R_Score"], row["F_Score"], row["M_Score"]
        if R >= 4 and F >= 4 and M >= 4: return "Champions"
        if R >= 4 and F >= 3 and M >= 3: return "Loyal High Value"
        if R >= 4 and F <= 3 and M <= 3: return "Potential Loyalists"
        if R >= 4 and F <= 2: return "New / Promising"
        if R <= 2 and M >= 4: return "High Value At Risk"
        if R <= 2 and F >= 3: return "Loyal At Risk"
        if R <= 2 and F <= 2 and M <= 2: return "Hibernating"
        return "Regular / Needs Nurture"

    rfm["BehavioralSet"] = rfm.apply(behavioural_set, axis=1)

    # Customer demographics are transaction attributes; collapse to the most
    # common/median customer-level value without contaminating behavioural features.
    demographic_cols = [c for c in ["Country", "City", "Gender", "IncomeBand"] if c in df.columns]
    if demographic_cols:
        demo = df.groupby("CustomerID")[demographic_cols].agg(
            lambda s: s.dropna().astype(str).mode().iloc[0] if not s.dropna().empty else "Unknown"
        ).reset_index()
        rfm = rfm.merge(demo, on="CustomerID", how="left")
    for c in ["Age", "Income"]:
        if c in df.columns:
            vals = _numeric_clean(df[c])
            med = pd.DataFrame({"CustomerID": df["CustomerID"], c: vals}).groupby("CustomerID")[c].median().reset_index()
            rfm = rfm.merge(med, on="CustomerID", how="left")

    action_map = {
        "Champions": "Protect & reward",
        "Loyal High Value": "Grow share of wallet",
        "Potential Loyalists": "Convert to loyal",
        "New / Promising": "Build purchase habit",
        "High Value At Risk": "Immediate win-back",
        "Loyal At Risk": "Reactivation",
        "Hibernating": "Low-cost reactivation",
        "Regular / Needs Nurture": "Nurture and personalize",
    }
    rfm["RecommendedAction"] = rfm["BehavioralSet"].map(action_map)

    features = ["Recency", "Frequency", "Monetary", "TotalQuantity", "UniqueProducts", "AvgItemsPerOrder", "CustomerTenureDays"]
    diagnostics = {
        "stage": 5,
        "reference_date": ref_date,
        "customer_count": int(len(rfm)),
        "features": features,
        "churn_rule": f"Recency > {churn_window} days",
        "churn_rate": float(rfm["Churn"].mean()) if len(rfm) else 0.0,
        "behavioral_sets": rfm["BehavioralSet"].value_counts().to_dict(),
        "demographic_fields": [c for c in ["Country", "City", "Age", "Gender", "Income", "IncomeBand"] if c in rfm.columns],
        "segmentation_method": "RFM behavioural rules; K-Means retained as an independent unsupervised diagnostic.",
    }
    return rfm, diagnostics


def kmo_bartlett(rfm: pd.DataFrame) -> Dict[str, Any]:
    numeric = [c for c in ["Recency", "Frequency", "Monetary", "TotalQuantity", "UniqueProducts", "AvgItemsPerOrder", "CustomerTenureDays"] if c in rfm.columns]
    x = rfm[numeric].replace([np.inf, -np.inf], np.nan).dropna()
    x = x.loc[:, x.nunique() > 1]
    if x.shape[1] < 3 or x.shape[0] < 20:
        return {"available": False, "reason": "Need at least 20 customers and 3 varying numeric features for KMO/Bartlett diagnostics."}
    try:
        from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity, calculate_kmo
        chi2, p = calculate_bartlett_sphericity(x)
        _, kmo_model = calculate_kmo(x)
        return {"available": True, "kmo": float(kmo_model), "bartlett_chi_square": float(chi2), "bartlett_p_value": float(p), "features_used": x.columns.tolist()}
    except Exception as e:
        return {"available": False, "reason": f"factor_analyzer unavailable or failed: {e}"}


def clustering(rfm: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    out = rfm.copy()
    cols = ["Recency", "Frequency", "Monetary", "TotalQuantity", "UniqueProducts", "AvgItemsPerOrder"]
    X = out[cols].replace([np.inf, -np.inf], np.nan).fillna(0)
    Xs = StandardScaler().fit_transform(X)
    candidates = []
    max_k = min(8, len(out) - 1)
    if max_k < 2:
        out["Cluster"] = 0
        return out, {"available": False, "reason": "Not enough customers for clustering."}
    for k in range(2, max_k + 1):
        model = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = model.fit_predict(Xs)
        score = silhouette_score(Xs, labels)
        candidates.append((k, score))
    best_k, best_score = max(candidates, key=lambda z: z[1])
    model = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    out["Cluster"] = model.fit_predict(Xs)
    return out, {"available": True, "selected_k": int(best_k), "silhouette": float(best_score), "tested_k": [{"k": k, "silhouette": s} for k, s in candidates], "features": cols, "interpretation": "K-Means clusters behavioural purchase patterns; demographics are used afterward to profile clusters, not arbitrarily encoded into the distance metric."}


def association_analysis(df: pd.DataFrame) -> Dict[str, Any]:
    try:
        from mlxtend.frequent_patterns import apriori, association_rules
        basket = pd.crosstab(df["InvoiceNo"], df["Product"]) > 0
        if basket.shape[1] < 2:
            return {"available": False, "reason": "At least two products are needed for association analysis."}
        min_support = max(0.01, min(0.10, 3 / max(len(basket), 1)))
        freq = apriori(basket, min_support=min_support, use_colnames=True, max_len=3)
        if freq.empty:
            return {"available": True, "frequent_itemsets": pd.DataFrame(), "rules": pd.DataFrame(), "min_support": min_support}
        rules = association_rules(freq, metric="confidence", min_threshold=0.20)
        if not rules.empty:
            rules = rules.sort_values(["lift", "confidence"], ascending=False).head(50)
        return {"available": True, "frequent_itemsets": freq, "rules": rules, "min_support": min_support, "transactions": int(len(basket))}
    except Exception as e:
        return {"available": False, "reason": f"mlxtend unavailable or failed: {e}"}


def model_proposal(rfm: pd.DataFrame) -> Dict[str, Any]:
    return {
        "stage": 6,
        "classification": ["Logistic Regression", "Random Forest", "Gradient Boosting"],
        "regression": ["Ridge", "Random Forest Regressor"],
        "selection_rule": "Compare validation metrics; retain interpretable and best-performing candidate.",
        "target_classification": "Churn",
        "target_regression": "Monetary",
        "approval": "auto-approved for demonstration; production requires analyst approval",
    }


def _adj_r2(r2: float, n: int, p: int) -> float:
    if n <= p + 1:
        return float("nan")
    return float(1 - (1 - r2) * (n - 1) / (n - p - 1))


def _build_temporal_snapshots(df: pd.DataFrame, horizon_days: int = 60) -> pd.DataFrame:
    """Build leakage-resistant customer snapshots.

    Each row describes what was known about a customer at a historical cutoff.
    The classification target is whether the customer makes NO purchase in the
    following horizon. This prevents defining churn from the same Recency used
    as a predictor.
    """
    d = df.copy()
    d["InvoiceDate"] = pd.to_datetime(d["InvoiceDate"], errors="coerce")
    d = d.dropna(subset=["InvoiceDate", "CustomerID"]).sort_values("InvoiceDate")
    min_date, max_date = d["InvoiceDate"].min(), d["InvoiceDate"].max()
    first_cut = min_date + pd.Timedelta(days=90)
    last_cut = max_date - pd.Timedelta(days=horizon_days)
    if first_cut >= last_cut:
        return pd.DataFrame()
    # Month-end-ish cutoffs give multiple temporal observations while keeping
    # enough future time to observe the outcome.
    cuts = pd.date_range(first_cut.normalize(), last_cut.normalize(), freq="30D")
    rows = []
    for cut in cuts:
        hist = d[d["InvoiceDate"] <= cut]
        future = d[(d["InvoiceDate"] > cut) & (d["InvoiceDate"] <= cut + pd.Timedelta(days=horizon_days))]
        if hist.empty:
            continue
        ref = cut + pd.Timedelta(days=1)
        agg = hist.groupby("CustomerID").agg(
            Recency=("InvoiceDate", lambda x: int((ref - x.max()).days)),
            Frequency=("InvoiceNo", "nunique"),
            Monetary=("AnalyticsAmount", "sum"),
            AvgOrderValue=("AnalyticsAmount", "mean"),
            TotalQuantity=("Quantity", "sum"),
            UniqueProducts=("Product", "nunique"),
            CustomerTenureDays=("InvoiceDate", lambda x: int((ref - x.min()).days)),
        ).reset_index()
        agg["AvgItemsPerOrder"] = agg["TotalQuantity"] / agg["Frequency"].replace(0, 1)
        future_orders = future.groupby("CustomerID")["InvoiceNo"].nunique() if not future.empty else pd.Series(dtype=float)
        future_revenue = future.groupby("CustomerID")["AnalyticsAmount"].sum() if not future.empty else pd.Series(dtype=float)
        agg["FutureOrders60d"] = agg["CustomerID"].map(future_orders).fillna(0)
        agg["FutureRevenue60d"] = agg["CustomerID"].map(future_revenue).fillna(0)
        agg["FutureChurn60d"] = (agg["FutureOrders60d"] == 0).astype(int)
        agg["SnapshotDate"] = cut
        rows.append(agg)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def model_training(rfm: pd.DataFrame, cleaned: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
    out: Dict[str, Any] = {"stage": 7, "classification": [], "regression": [], "artifacts": {}}
    if cleaned is None:
        out["classification_note"] = "Temporal training data unavailable."
        return out

    snapshots = _build_temporal_snapshots(cleaned, horizon_days=60)
    cls_features = ["Recency", "Frequency", "Monetary", "AvgOrderValue", "TotalQuantity", "UniqueProducts", "AvgItemsPerOrder", "CustomerTenureDays"]
    if snapshots.empty or snapshots["FutureChurn60d"].nunique() < 2 or len(snapshots) < 80:
        out["classification_note"] = "Insufficient temporal snapshots for a leakage-resistant churn model."
        return out

    X = snapshots[cls_features].replace([np.inf, -np.inf], np.nan).fillna(0)
    y = snapshots["FutureChurn60d"].astype(int)
    dates = snapshots["SnapshotDate"]
    unique_dates = sorted(dates.unique())
    if len(unique_dates) >= 2:
        test_date = unique_dates[-1]
        train_mask = dates < test_date
        test_mask = dates == test_date
    else:
        train_mask, test_mask = train_test_split(np.arange(len(snapshots)), test_size=0.25, random_state=42, stratify=y)
        # convert index arrays to boolean masks
        tr_idx, te_idx = train_mask, test_mask
        train_mask = np.zeros(len(snapshots), dtype=bool); test_mask = np.zeros(len(snapshots), dtype=bool)
        train_mask[tr_idx] = True; test_mask[te_idx] = True

    Xtr, Xv = X.loc[train_mask], X.loc[test_mask]
    ytr, yv = y.loc[train_mask], y.loc[test_mask]
    if ytr.nunique() < 2 or yv.nunique() < 2:
        # Fall back to stratified holdout only if the temporal holdout lacks both classes.
        Xtr, Xv, ytr, yv = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
        evaluation_scheme = "stratified holdout fallback"
    else:
        evaluation_scheme = "temporal holdout: latest snapshot is unseen test period"

    models = {
        "Logistic Regression": Pipeline([("scale", StandardScaler()), ("model", LogisticRegression(max_iter=3000, class_weight="balanced"))]),
        "Random Forest": RandomForestClassifier(n_estimators=400, min_samples_leaf=3, random_state=42, class_weight="balanced_subsample"),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42, n_estimators=150, max_depth=2, learning_rate=0.05),
    }
    for name, model in models.items():
        model.fit(Xtr, ytr)
        pred = model.predict(Xv)
        proba = model.predict_proba(Xv)[:, 1]
        out["classification"].append({
            "model": name,
            "accuracy": float(accuracy_score(yv, pred)),
            "precision": float(precision_score(yv, pred, zero_division=0)),
            "recall": float(recall_score(yv, pred, zero_division=0)),
            "f1": float(f1_score(yv, pred, zero_division=0)),
            "roc_auc": float(roc_auc_score(yv, proba)) if yv.nunique() == 2 else None,
            "pr_auc": float(average_precision_score(yv, proba)) if yv.nunique() == 2 else None,
            "brier_score": float(brier_score_loss(yv, proba)) if yv.nunique() == 2 else None,
            "confusion_matrix": confusion_matrix(yv, pred).tolist(),
        })
    best_name = max(out["classification"], key=lambda x: (x["roc_auc"] if x["roc_auc"] is not None else 0, x["f1"]))["model"]
    out["artifacts"]["best_classifier"] = best_name

    # Calibrate the chosen model with cross-validation where possible.
    from sklearn.calibration import CalibratedClassifierCV
    base = models[best_name]
    try:
        cv = min(3, int(y.value_counts().min()))
        calibrated = CalibratedClassifierCV(base, method="sigmoid", cv=max(2, cv))
        calibrated.fit(X, y)
        final_classifier = calibrated
        calibration = "sigmoid cross-validation calibration"
    except Exception:
        final_classifier = base.fit(X, y)
        calibration = "uncalibrated fallback"
    out["artifacts"]["classifier_model"] = final_classifier
    out["artifacts"]["classifier_features"] = cls_features
    out["artifacts"]["churn_horizon_days"] = 60
    out["artifacts"]["evaluation_scheme"] = evaluation_scheme
    out["artifacts"]["calibration"] = calibration
    out["artifacts"]["training_snapshots"] = int(len(snapshots))
    out["artifacts"]["training_periods"] = int(len(unique_dates))
    out["classification_note"] = "Churn is a forward-looking 60-day no-purchase target built from historical customer snapshots; current-customer probabilities estimate risk of no purchase in the next 60 days."

    # Predict future 60-day revenue using the same temporal snapshots.
    reg_features = cls_features
    yr = np.log1p(snapshots["FutureRevenue60d"].astype(float))
    Xr = snapshots[reg_features].replace([np.inf, -np.inf], np.nan).fillna(0)
    if len(snapshots) >= 80 and yr.nunique() >= 5:
        if len(unique_dates) >= 2:
            train_mask_r = dates < unique_dates[-1]
            test_mask_r = dates == unique_dates[-1]
            Xtr_r, Xv_r = Xr.loc[train_mask_r], Xr.loc[test_mask_r]
            ytr_r, yv_r = yr.loc[train_mask_r], yr.loc[test_mask_r]
            if len(yv_r) < 10 or yv_r.nunique() < 2:
                Xtr_r, Xv_r, ytr_r, yv_r = train_test_split(Xr, yr, test_size=0.25, random_state=42)
        else:
            Xtr_r, Xv_r, ytr_r, yv_r = train_test_split(Xr, yr, test_size=0.25, random_state=42)
        models_r = {
            "Ridge": Pipeline([("scale", StandardScaler()), ("model", Ridge(alpha=2.0))]),
            "Random Forest Regressor": RandomForestRegressor(n_estimators=300, min_samples_leaf=3, random_state=42),
        }
        for name, model in models_r.items():
            model.fit(Xtr_r, ytr_r)
            pred_log = model.predict(Xv_r)
            pred = np.maximum(0, np.expm1(pred_log))
            actual = np.maximum(0, np.expm1(yv_r))
            r2 = r2_score(actual, pred)
            rmse = math.sqrt(mean_squared_error(actual, pred))
            out["regression"].append({"model": name, "r2": float(r2), "adjusted_r2": _adj_r2(r2, len(actual), len(reg_features)), "rmse": float(rmse), "mae": float(mean_absolute_error(actual, pred))})
        best_reg = max(out["regression"], key=lambda x: x["r2"])["model"]
        final_reg = models_r[best_reg].fit(Xr, yr)
        out["artifacts"]["best_regressor"] = best_reg
        out["artifacts"]["regressor_model"] = final_reg
        out["artifacts"]["regressor_features"] = reg_features
        out["artifacts"]["regression_target"] = "future 60-day revenue"
    else:
        out["regression_note"] = "Insufficient temporal snapshots for future-revenue regression."
    return out

def model_evaluation(rfm: pd.DataFrame, training: Dict[str, Any]) -> Dict[str, Any]:
    eval_out = {
        "stage": 8,
        "selection": {},
        "predictions": pd.DataFrame(),
        "residuals": pd.DataFrame(),
        "untouched_test_principle": "Churn performance is evaluated on a future temporal snapshot before the final model is refit for current-customer scoring.",
    }
    pred = rfm[["CustomerID", "Churn", "Monetary"]].copy()
    if training["artifacts"].get("classifier_model") is not None:
        m = training["artifacts"]["classifier_model"]
        feats = training["artifacts"]["classifier_features"]
        X = rfm[feats].replace([np.inf, -np.inf], np.nan).fillna(0)
        pred["ChurnProbability"] = m.predict_proba(X)[:, 1]
        pred["PredictedChurn"] = (pred["ChurnProbability"] >= 0.50).astype(int)
        eval_out["selection"]["classifier"] = training["artifacts"].get("best_classifier")
        eval_out["selection"]["churn_horizon_days"] = training["artifacts"].get("churn_horizon_days", 60)
        eval_out["selection"]["evaluation_scheme"] = training["artifacts"].get("evaluation_scheme")
        eval_out["selection"]["probability_calibration"] = training["artifacts"].get("calibration")
    if training["artifacts"].get("regressor_model") is not None:
        m = training["artifacts"]["regressor_model"]
        feats = training["artifacts"]["regressor_features"]
        X = rfm[feats].replace([np.inf, -np.inf], np.nan).fillna(0)
        pred["PredictedFutureRevenue60d"] = np.maximum(0, np.expm1(m.predict(X)))
        pred["RevenueResidual"] = pred["Monetary"] - pred["PredictedFutureRevenue60d"]
        eval_out["selection"]["regressor"] = training["artifacts"].get("best_regressor")
    eval_out["predictions"] = pred
    eval_out["residuals"] = pred[[c for c in ["CustomerID", "RevenueResidual"] if c in pred.columns]].copy()
    return eval_out

def generate_recommendations(rfm: pd.DataFrame, clusters: Dict[str, Any], assoc: Dict[str, Any], training: Dict[str, Any]) -> List[str]:
    recs = []
    churn_rate = float(rfm["Churn"].mean()) if len(rfm) else 0
    recs.append(f"Prioritize retention for the {churn_rate:.1%} of customers crossing the current inactivity threshold, with value-based win-back for high-value at-risk customers.")
    if "BehavioralSet" in rfm.columns:
        top_set = rfm["BehavioralSet"].value_counts().idxmax()
        recs.append(f"Use behavioural sets as the primary customer-persona layer; '{top_set}' is the largest group and should receive a differentiated nurture strategy.")
        if "High Value At Risk" in set(rfm["BehavioralSet"]):
            hv = rfm.loc[rfm["BehavioralSet"] == "High Value At Risk", "Monetary"].sum()
            recs.append(f"Protect high-value revenue at risk: High Value At Risk customers represent {hv:,.0f} in observed monetary value.")
    if clusters.get("available"):
        recs.append(f"Retain the {clusters['selected_k']}-cluster K-Means solution as a secondary discovery view; interpret clusters through RFM and demographic profiles rather than treating numeric cluster IDs as personas.")
    if "Country" in rfm.columns:
        country = rfm.groupby("Country").agg(Customers=("CustomerID", "nunique"), Revenue=("Monetary", "sum"), Churn=("Churn", "mean")).sort_values("Revenue", ascending=False)
        if not country.empty:
            top = country.index[0]
            recs.append(f"Prioritize geographic planning around {top}, while comparing its behavioural mix and churn against other countries before reallocating budget.")
    if assoc.get("available") and not assoc.get("rules", pd.DataFrame()).empty:
        top = assoc["rules"].iloc[0]
        recs.append(f"Use basket recommendations around strong product affinities: top observed lift is {top['lift']:.2f} with confidence {top['confidence']:.1%}.")
    if training.get("regression"):
        best = max(training["regression"], key=lambda x: x["r2"])
        recs.append(f"Use {best['model']} as the current revenue-value benchmark (R² {best['r2']:.3f}, RMSE {best['rmse']:,.2f}).")
    recs.append("Keep raw, processed, engineered, model and evaluation artifacts logically separated so results can be reproduced and audited.")
    return recs


def run_pipeline(raw_bytes: bytes, project_name: str = "Sales Customer Intelligence") -> Dict[str, Any]:
    """Execute the exact eight-stage sequence and return UI-safe outputs."""
    raw = pd.read_csv(io.BytesIO(raw_bytes))
    stage1 = project_management(raw, project_name)
    ingested, stage2 = data_ingestion(raw_bytes)
    stage3, understood = data_understanding(ingested)
    cleaned, stage4 = preprocessing(understood)
    rfm, stage5 = feature_engineering(cleaned)
    rfm_clustered, cluster_diag = clustering(rfm)
    stage5["cluster"] = cluster_diag
    stage5["kmo_bartlett"] = kmo_bartlett(rfm_clustered)
    stage6 = model_proposal(rfm_clustered)
    training = model_training(rfm_clustered, cleaned)
    stage7 = {k: v for k, v in training.items() if k != "artifacts"}
    evaluation = model_evaluation(rfm_clustered, training)
    stage8 = {k: v for k, v in evaluation.items() if k not in ("predictions", "residuals")}
    assoc = association_analysis(cleaned)
    recommendations = generate_recommendations(rfm_clustered, cluster_diag, assoc, training)

    dev = {
        "pipeline": [stage1, stage2, stage3, stage4, stage5, stage6, stage7, stage8],
        "schema_mapping": stage2["schema"],
        "kmo_bartlett": stage5["kmo_bartlett"],
        "cluster": cluster_diag,
        "association": {k: v for k, v in assoc.items() if k not in ("frequent_itemsets", "rules")},
        "classification": training.get("classification", []),
        "regression": training.get("regression", []),
        "selected_models": evaluation["selection"],
        "governance": {
            "lineage": ["raw", "processed", "splits", "engineered", "models", "evaluation", "final candidate"],
            "versioning": "run timestamp + pipeline sequence",
            "approval_gates": ["feature approval", "model approval", "decision tracking"],
        },
    }
    return {
        "cleaned": cleaned,
        "rfm": rfm_clustered,
        "profile": stage3,
        "stages": {"1": stage1, "2": stage2, "3": stage3, "4": stage4, "5": stage5, "6": stage6, "7": stage7, "8": stage8},
        "training": training,
        "evaluation": evaluation,
        "association": assoc,
        "recommendations": recommendations,
        "dev": dev,
    }


def serializable_dev(dev: Dict[str, Any]) -> Dict[str, Any]:
    """Remove model objects/dataframes for a compact developer artifact."""
    def clean(v):
        if isinstance(v, pd.DataFrame):
            return v.head(50).to_dict(orient="records")
        if isinstance(v, dict):
            return {k: clean(x) for k, x in v.items() if k not in {"artifacts"}}
        if isinstance(v, list):
            return [clean(x) for x in v]
        return _json_safe(v)
    return clean(dev)
