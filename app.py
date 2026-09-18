import io
import os
from datetime import datetime

import pandas as pd
import streamlit as st
import pydeck as pdk
import plotly.express as px

from customer_analyzer_backend import run_pipeline, serializable_dev

st.set_page_config(page_title="Sales Data Analyser", page_icon="SA", layout="wide", initial_sidebar_state="expanded")

# ----------------------------- style -----------------------------
st.markdown("""
<style>
.block-container {padding-top: 1.4rem; padding-bottom: 2rem;}
[data-testid="stMetricValue"] {font-size: 1.45rem;}
.pipeline-step {border:1px solid #d9e2ec; border-radius:12px; padding:12px; background:#f8fafc; min-height:120px;}
.small-note {font-size:0.82rem; color:#64748b;}
</style>
""", unsafe_allow_html=True)

# ----------------------------- header -----------------------------
st.title("Sales Data Analyser")
st.caption("End-to-end governed customer intelligence: sales → cleaning → features → models → evaluation → decisions")

with st.sidebar:
    st.header("Project")
    project_name = st.text_input("Project name", "Sales Customer Intelligence")
    st.markdown("**Dev:** Adithya  \\n**Role:** Analyst")
    st.divider()
    uploaded = st.file_uploader("Add sales data", type=["csv"], help="Upload a CSV containing customer/order/sales data.")
    st.caption("Expected concepts: CustomerID, InvoiceDate, TotalAmount, InvoiceNo, Product, Quantity. The mapper accepts common alternatives.")
    if uploaded:
        if st.button("Run / Refresh Analysis", type="primary", use_container_width=True):
            with st.spinner("Running MLOps pipeline stages 1–8..."):
                try:
                    st.session_state["result"] = run_pipeline(uploaded.getvalue(), project_name)
                    st.session_state["filename"] = uploaded.name
                    st.session_state["run_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                except Exception as e:
                    st.error(f"Pipeline failed: {e}")
    if st.session_state.get("result"):
        if st.button("Clear analysis", use_container_width=True):
            st.session_state.pop("result", None)
            st.rerun()

result = st.session_state.get("result")
if not result:
    st.info("Upload a CSV in the sidebar and run the analysis.")
    st.markdown("### What the application produces")
    c1,c2,c3,c4 = st.columns(4)
    for c, title, text in zip([c1,c2,c3,c4], ["Customer view","Churn intelligence","Basket insights","MLOps report"], ["RFM, segments and sales patterns", "Risk scores and retention priorities", "Association rules and product affinities", "LLM-assisted MD/MD report with metrics"]):
        with c:
            st.markdown(f"<div class='pipeline-step'><b>{title}</b><br><span class='small-note'>{text}</span></div>", unsafe_allow_html=True)
    st.stop()

rfm = result["rfm"]
cleaned = result["cleaned"]
profile = result["profile"]
training = result["training"]
eval_out = result["evaluation"]
assoc = result["association"]

# ----------------------------- KPI strip -----------------------------
total_sales = cleaned["AnalyticsAmount"].sum()
customers = rfm["CustomerID"].nunique()
orders = cleaned["InvoiceNo"].nunique()
churn_rate = rfm["Churn"].mean() if len(rfm) else 0
best_r2 = max([x["r2"] for x in training.get("regression", [])], default=float("nan"))
best_f1 = max([x["f1"] for x in training.get("classification", [])], default=float("nan"))

k1,k2,k3,k4,k5,k6 = st.columns(6)
k1.metric("Total Sales", f"{total_sales:,.0f}")
k2.metric("Customers", f"{customers:,}")
k3.metric("Orders", f"{orders:,}")
k4.metric("Current inactivity", f"{churn_rate:.1%}")
k5.metric("Best R²", "—" if pd.isna(best_r2) else f"{best_r2:.3f}")
k6.metric("Best F1", "—" if pd.isna(best_f1) else f"{best_f1:.3f}")

st.caption(f"Churn model: forward 60-day no-purchase target | Source: {st.session_state.get('filename','uploaded CSV')}  |  Run: {st.session_state.get('run_time','')} | Backend sequence is hidden from the main analytical views.")

# ----------------------------- tabs -----------------------------
tabs = st.tabs(["Overview", "Sales & EDA", "Customers & Churn", "Behavioral Segments", "Clusters", "Demographics & Geo", "Associations", "Models & Metrics", "Recommendations", "Report"])

with tabs[0]:
    st.subheader("Executive overview")
    left,right = st.columns([1.35,1])
    with left:
        st.markdown("#### Sales trend")
        trend = cleaned.groupby(cleaned["InvoiceDate"].dt.to_period("M"))["AnalyticsAmount"].sum().reset_index()
        trend["InvoiceDate"] = trend["InvoiceDate"].astype(str)
        st.line_chart(trend.set_index("InvoiceDate")["AnalyticsAmount"])
    with right:
        st.markdown("#### Business signals")
        for rec in result["recommendations"]:
            st.write("•", rec)
        st.markdown("#### Data coverage")
        st.write(f"**{profile['date_min']} → {profile['date_max']}**")
        st.write(f"{profile['rows']:,} transactions | {customers:,} customers | {orders:,} orders")

with tabs[1]:
    st.subheader("Sales & exploratory analysis")
    a,b,c = st.columns(3)
    with a:
        st.markdown("#### Sales by month")
        monthly = cleaned.assign(Month=cleaned["InvoiceDate"].dt.to_period("M").astype(str)).groupby("Month")["AnalyticsAmount"].sum()
        st.bar_chart(monthly)
    with b:
        st.markdown("#### Quantity distribution")
        st.bar_chart(cleaned["Quantity"].clip(lower=0).value_counts().sort_index().head(30))
    with c:
        st.markdown("#### Top products")
        top_products = cleaned.groupby("Product")["AnalyticsAmount"].sum().sort_values(ascending=False).head(10)
        st.bar_chart(top_products)
    st.markdown("#### Data-quality checks")
    q1, q2, q3 = st.columns(3)
    q1.metric("Rows after cleaning", f"{len(cleaned):,}")
    q2.metric("Duplicates removed", f"{result['stages']['4'].get('duplicates_removed', 0):,}")
    q3.metric("Amount mismatches >1%", f"{result['stages']['4'].get('amount_mismatch_rows', 0):,}")
    st.dataframe(profile["missing_by_column"], use_container_width=True, hide_index=True)
    st.markdown("#### Correlations")
    corr = profile.get("correlations")
    if isinstance(corr, pd.DataFrame) and not corr.empty:
        st.dataframe(corr.round(3), use_container_width=True)
    st.markdown("#### Cleaned transaction sample")
    st.dataframe(cleaned.head(100), use_container_width=True, hide_index=True)

with tabs[2]:
    st.subheader("Customer value & churn risk")
    a,b,c,d = st.columns(4)
    a.metric("At-risk customers", f"{int(rfm['Churn'].sum()):,}")
    b.metric("Avg customer value", f"{rfm['Monetary'].mean():,.0f}")
    c.metric("Avg frequency", f"{rfm['Frequency'].mean():.1f}")
    d.metric("Median recency", f"{rfm['Recency'].median():.0f} days")
    st.markdown("#### Predicted 60-day churn risk")
    if "ChurnProbability" in eval_out["predictions"].columns:
        st.bar_chart(eval_out["predictions"]["ChurnProbability"].round(1).value_counts().sort_index())
    else:
        st.bar_chart(rfm["Churn"].value_counts().rename(index={0:"Active",1:"Churn"}))
    st.markdown("#### Highest-value churn-risk customers")
    customer_view = eval_out["predictions"].merge(rfm, on=["CustomerID","Churn","Monetary"], how="left")
    risk_cols = [c for c in ["CustomerID","Monetary","Recency","Frequency","UniqueProducts","ChurnProbability","PredictedChurn"] if c in customer_view]
    if "ChurnProbability" in customer_view:
        customer_view = customer_view.sort_values(["ChurnProbability","Monetary"], ascending=[False,False])
    st.dataframe(customer_view[risk_cols].head(50), use_container_width=True, hide_index=True)

with tabs[3]:
    st.subheader("Behavioural customer sets")
    st.caption("Primary personas are rule-based RFM behavioural sets. K-Means is shown separately as an unsupervised discovery technique.")
    if "BehavioralSet" in rfm.columns:
        seg = rfm.groupby("BehavioralSet").agg(
            Customers=("CustomerID", "nunique"),
            Revenue=("Monetary", "sum"),
            AvgSpend=("Monetary", "mean"),
            AvgRecency=("Recency", "mean"),
            AvgFrequency=("Frequency", "mean"),
            ChurnRate=("Churn", "mean")
        ).reset_index().sort_values("Revenue", ascending=False)
        c1,c2 = st.columns(2)
        with c1:
            st.plotly_chart(px.bar(seg, x="BehavioralSet", y="Customers", title="Customers by behavioural set", labels={"BehavioralSet":"Customer set", "Customers":"Customers"}), use_container_width=True)
        with c2:
            st.plotly_chart(px.bar(seg, x="BehavioralSet", y="Revenue", title="Revenue by behavioural set", labels={"BehavioralSet":"Customer set", "Revenue":"Revenue"}), use_container_width=True)
        st.dataframe(seg.style.format({"Revenue":"{:,.0f}","AvgSpend":"{:,.0f}","AvgRecency":"{:.1f}","AvgFrequency":"{:.1f}","ChurnRate":"{:.1%}"}), use_container_width=True, hide_index=True)
        st.markdown("#### Behavioural definitions")
        definitions = pd.DataFrame([
            ["Champions","Recent + frequent + high value","Protect & reward"],
            ["Loyal High Value","Recent repeat buyers with strong value","Grow share of wallet"],
            ["Potential Loyalists","Recent with room to increase frequency/value","Convert to loyal"],
            ["New / Promising","Recent, limited purchase history","Build purchase habit"],
            ["High Value At Risk","High value but inactive","Immediate win-back"],
            ["Loyal At Risk","Frequent historically, now inactive","Reactivation"],
            ["Hibernating","Low recent and historical engagement","Low-cost reactivation"],
            ["Regular / Needs Nurture","Middle behavioural profile","Nurture and personalize"],
        ], columns=["Set","Definition","Action"])
        st.dataframe(definitions, use_container_width=True, hide_index=True)
        st.markdown("#### Customer-level segment explorer")
        selected = st.multiselect("Behavioural sets", sorted(rfm["BehavioralSet"].dropna().unique()), default=sorted(rfm["BehavioralSet"].dropna().unique()))
        view_cols=[c for c in ["CustomerID","BehavioralSet","R_Score","F_Score","M_Score","RFM_Score","Recency","Frequency","Monetary","Churn","RecommendedAction","Country","City","Age","Gender","IncomeBand"] if c in rfm.columns]
        st.dataframe(rfm[rfm["BehavioralSet"].isin(selected)][view_cols].sort_values("Monetary", ascending=False).head(150), use_container_width=True, hide_index=True)
    else:
        st.warning("Behavioural segmentation could not be generated.")

with tabs[4]:
    st.subheader("Unsupervised customer clusters")
    cluster_diag = result["stages"]["5"].get("cluster", {})
    if cluster_diag.get("available"):
        st.write(f"Selected **{cluster_diag['selected_k']} clusters** using silhouette score (**{cluster_diag['silhouette']:.3f}**).")
        cdf = rfm.groupby("Cluster").agg(Customers=("CustomerID","count"), AvgSpend=("Monetary","mean"), Revenue=("Monetary","sum"), AvgRecency=("Recency","mean"), AvgFrequency=("Frequency","mean"), ChurnRate=("Churn","mean")).reset_index()
        c1,c2=st.columns(2)
        with c1:
            st.plotly_chart(px.bar(cdf, x="Cluster", y="Customers", title="Cluster size"), use_container_width=True)
        with c2:
            st.plotly_chart(px.bar(cdf, x="Cluster", y="Revenue", title="Cluster revenue"), use_container_width=True)
        # Bubble: frequency x average spend, bubble size = customers, colour = cluster.
        st.plotly_chart(px.scatter(cdf, x="AvgFrequency", y="AvgSpend", size="Customers", color="Cluster", hover_data=["Revenue","AvgRecency","ChurnRate"], title="Customer cluster bubble map", labels={"AvgFrequency":"Average purchase frequency","AvgSpend":"Average customer spend"}), use_container_width=True)
        st.dataframe(cdf.style.format({"AvgSpend":"{:,.0f}","Revenue":"{:,.0f}","AvgRecency":"{:.1f}","AvgFrequency":"{:.1f}","ChurnRate":"{:.1%}"}), use_container_width=True, hide_index=True)
        st.info("Cluster IDs are mathematical labels, not customer personas. Use the behavioural sets and demographic profiles below to explain what each cluster contains.")
    else:
        st.warning(cluster_diag.get("reason","Clustering unavailable."))

with tabs[5]:
    st.subheader("Customer demographics & geographic heatzones")
    demo_cols=[c for c in ["Age","Gender","Income","IncomeBand","Country","City"] if c in rfm.columns]
    if not demo_cols:
        st.warning("No demographic fields were detected in the uploaded data.")
    else:
        c1,c2,c3=st.columns(3)
        c1.metric("Demographic fields", len(demo_cols))
        c2.metric("Countries", rfm["Country"].nunique() if "Country" in rfm else 0)
        c3.metric("Cities", rfm["City"].nunique() if "City" in rfm else 0)
        if "Country" in rfm.columns:
            country = rfm.groupby("Country").agg(Customers=("CustomerID","nunique"), Revenue=("Monetary","sum"), AvgSpend=("Monetary","mean"), ChurnRate=("Churn","mean")).reset_index().sort_values("Revenue", ascending=False)
            st.markdown("#### Country performance")
            st.plotly_chart(px.bar(country, x="Country", y="Revenue", title="Revenue by country"), use_container_width=True)
            st.dataframe(country.style.format({"Revenue":"{:,.0f}","AvgSpend":"{:,.0f}","ChurnRate":"{:.1%}"}), use_container_width=True, hide_index=True)

            st.markdown("#### Country × behavioural set")
            cx = pd.crosstab(rfm["Country"], rfm["BehavioralSet"]) if "BehavioralSet" in rfm.columns else pd.DataFrame()
            if not cx.empty:
                st.dataframe(cx, use_container_width=True)
                st.plotly_chart(px.imshow(cx, text_auto=True, aspect="auto", title="Customer behaviour heatmap by country"), use_container_width=True)

            # Country centroids for a defensible country-level heatzone view.
            coords={
                "India":(20.5937,78.9629), "UAE":(23.4241,53.8478), "Singapore":(1.3521,103.8198),
                "UK":(55.3781,-3.4360), "USA":(37.0902,-95.7129), "United Kingdom":(55.3781,-3.4360),
                "United States":(37.0902,-95.7129), "United Arab Emirates":(23.4241,53.8478)
            }
            geo = country.copy()
            geo["lat"] = geo["Country"].map(lambda x: coords.get(x,(None,None))[0])
            geo["lon"] = geo["Country"].map(lambda x: coords.get(x,(None,None))[1])
            geo = geo.dropna(subset=["lat","lon"])
            if not geo.empty:
                st.markdown("#### Revenue heatzones")
                layer=pdk.Layer("HeatmapLayer", data=geo, get_position="[lon, lat]", get_weight="Revenue", radius_pixels=55, intensity=1.4, threshold=0.05)
                view=pdk.ViewState(latitude=20, longitude=75, zoom=1.0)
                st.pydeck_chart(pdk.Deck(layers=[layer], initial_view_state=view, tooltip={"text":"{Country}: {Revenue}"}), use_container_width=True)
                st.caption("Heatzones are country-level revenue concentrations positioned at country centroids; they are not street-level customer locations.")

        # Demographic profile against behavioural sets.
        if "Age" in rfm.columns and "BehavioralSet" in rfm.columns:
            age = rfm.dropna(subset=["Age"]).copy()
            if not age.empty:
                st.markdown("#### Age profile by behavioural set")
                st.plotly_chart(px.box(age, x="BehavioralSet", y="Age", points=False, title="Age distribution across behavioural sets"), use_container_width=True)
        if "IncomeBand" in rfm.columns and "BehavioralSet" in rfm.columns:
            ib = pd.crosstab(rfm["IncomeBand"], rfm["BehavioralSet"])
            st.markdown("#### Income band × behavioural set")
            st.dataframe(ib, use_container_width=True)
        if "Gender" in rfm.columns and "BehavioralSet" in rfm.columns:
            gb = pd.crosstab(rfm["Gender"], rfm["BehavioralSet"])
            st.markdown("#### Gender × behavioural set")
            st.dataframe(gb, use_container_width=True)

with tabs[6]:
    st.subheader("Market basket / association analysis")
    if assoc.get("available"):
        transactions = assoc.get("transactions")
        try:
            transactions_display = f"{int(float(transactions)):,}"
        except (TypeError, ValueError):
            transactions_display = "—"
        st.write(
            f"Transactions analysed: **{transactions_display}** | "
            f"minimum support: **{assoc.get('min_support', 0):.3f}**"
        )
        rules = assoc.get("rules", pd.DataFrame()).copy()
        if not rules.empty:
            display = rules[["antecedents","consequents","support","confidence","lift"]].copy()
            display["antecedents"] = display["antecedents"].apply(lambda x: ", ".join(map(str, x)))
            display["consequents"] = display["consequents"].apply(lambda x: ", ".join(map(str, x)))
            st.dataframe(display.round(3), use_container_width=True, hide_index=True)
            st.markdown("#### Top rules by lift")
            st.bar_chart(rules.head(10).set_index(rules.head(10).apply(lambda r: f"{', '.join(map(str,r['antecedents']))} → {', '.join(map(str,r['consequents']))}", axis=1))["lift"])
        else:
            st.info("No rules met the current support/confidence thresholds.")
    else:
        st.warning(assoc.get("reason","Association analysis unavailable."))

with tabs[7]:
    st.subheader("Model proposal, training and evaluation")
    st.markdown("#### Classification — churn")
    if training.get("classification"):
        cls = pd.DataFrame(training["classification"])
        cols = [c for c in ["model","accuracy","precision","recall","f1","roc_auc","pr_auc","brier_score"] if c in cls.columns]
        st.dataframe(cls[cols].round(3), use_container_width=True, hide_index=True)
        st.info("Evaluation uses historical customer snapshots and a forward 60-day no-purchase target. The latest snapshot is held out as the test period; the selected model is then refit for current-customer scoring. Recency is a predictor, but it is not the churn label.")
    else:
        st.info(training.get("classification_note","Classification unavailable."))
    st.markdown("#### Regression — customer monetary value")
    if training.get("regression"):
        reg = pd.DataFrame(training["regression"])
        st.dataframe(reg[["model","r2","adjusted_r2","rmse","mae"]].round(3), use_container_width=True, hide_index=True)
        best = reg.sort_values("r2", ascending=False).iloc[0]
        m1,m2,m3,m4 = st.columns(4)
        m1.metric("R²", f"{best.r2:.3f}")
        m2.metric("Adjusted R²", f"{best.adjusted_r2:.3f}")
        m3.metric("RMSE", f"{best.rmse:,.2f}")
        m4.metric("MAE", f"{best.mae:,.2f}")
    else:
        st.info(training.get("regression_note","Regression unavailable."))
    if "PredictedFutureRevenue60d" in eval_out["predictions"].columns:
        st.markdown("#### Predicted future 60-day revenue")
        st.dataframe(eval_out["predictions"][["CustomerID","Monetary","PredictedFutureRevenue60d"]].sort_values("PredictedFutureRevenue60d", ascending=False).head(50), use_container_width=True, hide_index=True)

with tabs[8]:
    st.subheader("Recommendations")
    for i, rec in enumerate(result["recommendations"], 1):
        st.markdown(f"**{i}.** {rec}")
    st.markdown("#### Decision table")
    decision = pd.DataFrame([
        ["High churn probability + high monetary value", "Retention / win-back", "Immediate"],
        ["High-frequency customer + low recency", "Loyalty / cross-sell", "High"],
        ["Strong association lift", "Bundle / recommendation", "Medium"],
        ["High-value cluster", "VIP / differentiated offer", "High"],
    ], columns=["Signal","Action","Priority"])
    st.dataframe(decision, use_container_width=True, hide_index=True)

with tabs[9]:
    st.subheader("LLM-assisted management report")
    st.write("Generate a report as Markdown or Word. If a local Ollama Llama model is configured, it writes the narrative; otherwise the application produces a deterministic management report from the computed results.")
    model = st.text_input("Ollama Llama model", os.getenv("OLLAMA_MODEL", "llama3.1"))
    report_format = st.radio("Format", ["Markdown (.md)", "Word (.docx)"], horizontal=True)
    if st.button("Generate report", type="primary"):
        from report_generator import generate_report
        with st.spinner("Generating report..."):
            try:
                content = generate_report(result, model=model)
                if report_format.startswith("Markdown"):
                    st.download_button("Download report.md", content.encode("utf-8"), file_name="sales_customer_analysis_report.md", mime="text/markdown")
                    st.markdown(content)
                else:
                    docx_bytes = generate_report(result, model=model, as_docx=True)
                    st.download_button("Download report.docx", docx_bytes, file_name="sales_customer_analysis_report.docx", mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document")
                    st.success("Word report generated.")
            except Exception as e:
                st.error(f"Report generation failed: {e}")
