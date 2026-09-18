"""Developer-only diagnostics page.
No pipeline implementation is exposed here; only backend results returned by the pipeline are rendered.
"""
import json
import pandas as pd
import streamlit as st

from customer_analyzer_backend import serializable_dev

st.set_page_config(page_title="Dev | Sales Data Analyser", page_icon="DEV", layout="wide")
st.title("Dev — Backend Diagnostics")
st.caption("Adithya · Analyst · concise pipeline/test results")

result = st.session_state.get("result")
if not result:
    st.info("Run an analysis from the Sales Data Analyser page first.")
    st.stop()

dev = result["dev"]
assoc = result["association"]

c1,c2,c3,c4,c5 = st.columns(5)
c1.metric("Pipeline stages", 8)
c2.metric("Mapped fields", len(dev["schema_mapping"]["mapping"]))
c3.metric("KMO", "—" if not dev["kmo_bartlett"].get("available") else f"{dev['kmo_bartlett']['kmo']:.3f}")
c4.metric("Clusters", dev["cluster"].get("selected_k", "—"))
c5.metric("Association rules", len(assoc.get("rules", pd.DataFrame())))

st.subheader("1–8 pipeline result")
stages = ["Project Management","Data Ingestion","Data Understanding","Preprocessing","Feature Engineering","Model Proposal","Model Training","Model Evaluation"]
rows=[]
for i,name in enumerate(stages,1):
    stage=dev["pipeline"][i-1]
    rows.append({"Stage":i,"Module":name,"Result":"Complete","Key output": ", ".join(str(k) for k in stage.keys() if k not in {"stage"})[:140]})
st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

st.subheader("Schema mapping")
st.dataframe(pd.DataFrame([{"Canonical":k,"Source":v,"Confidence":dev["schema_mapping"]["confidence"].get(k)} for k,v in dev["schema_mapping"]["mapping"].items()]), use_container_width=True, hide_index=True)

left,right=st.columns(2)
with left:
    st.subheader("KMO / Bartlett")
    st.json(dev["kmo_bartlett"])
with right:
    st.subheader("Clustering")
    st.json(dev["cluster"])

st.subheader("Classification metrics")
st.dataframe(pd.DataFrame(dev["classification"]).round(3) if dev["classification"] else pd.DataFrame(), use_container_width=True, hide_index=True)
st.subheader("Regression metrics")
st.dataframe(pd.DataFrame(dev["regression"]).round(3) if dev["regression"] else pd.DataFrame(), use_container_width=True, hide_index=True)

st.subheader("Association diagnostics")
st.json(dev["association"])

st.subheader("Governance / lineage")
st.json(dev["governance"])

payload=json.dumps(serializable_dev(dev), indent=2, default=str)
st.download_button("Download complete dev results (JSON)", payload, file_name="dev_backend_results.json", mime="application/json")
