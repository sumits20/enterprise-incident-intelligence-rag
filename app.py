import streamlit as st

from rag.loader import load_incidents
from rag.documents import build_documents
from rag.embeddings import build_faiss_index
from rag.retrieval import retrieve_top_k
from llm.openai_client import generate_answer


def build_evidence_text(results):
    blocks = []
    for _, r in results.iterrows():
        blocks.append(
            f"[{r.get('Incident_ID','')}] "
            f"Date: {r.get('Date','')}, "
            f"Priority: {r.get('Priority','')}, "
            f"Team: {r.get('Responsible_Team','')}\n"
            f"Issue: {r.get('Issue_Description','')}\n"
            f"Root Cause: {r.get('Root_Cause','')}\n"
            f"Resolution: {r.get('Resolution','')}\n"
        )
    return "\n---\n".join(blocks)


st.set_page_config(page_title="Enterprise Incident Intelligence", layout="wide")
st.title("Enterprise Incident Intelligence")
st.caption("User Query → Retrieve similar incidents → Generate evidence-based AI answer (RAG)")

EXCEL_PATH = "data/incidents.xlsx"

try:
    df = load_incidents(EXCEL_PATH)
    st.success(f"Knowledge base: {len(df)} incidents")

    # Build retrieval index (cached)
    df["doc_text"] = build_documents(df)

    @st.cache_resource
    def get_model_and_index(texts):
        return build_faiss_index(texts)

    model, index = get_model_and_index(tuple(df["doc_text"].tolist()))

    # Retrieval UI (TOP K) 
    st.subheader("Semantic Incident Retrieval")
    query = st.text_input("Describe your issue", value="subscription auto renewal failure")
    top_k = st.slider("Top K similar incidents", 1, 7, 5)

    results = None
    if query:
        results = retrieve_top_k(
            df=df,
            query=query,
            model=model,
            index=index,
            k=int(top_k)
        )
        MIN_SIMILARITY = 0.35  # tune this (start 0.30–0.45)

        best = float(results["similarity"].max()) if "similarity" in results.columns else 0.0

        if best < MIN_SIMILARITY:
            st.warning(
                f"No strong matches found (best similarity={best:.2f}). "
                "Try a more specific incident description."
                )
        
        st.success(f"Found {len(results)} similar incidents")

        display_cols = [
            c for c in [
                "Incident_ID",
                "Issue_Type",
                "Priority",
                "Responsible_Team",
                "Typical_Resolution_Hours",
                "Date",
                "similarity",
            ]
            if c in results.columns
        ]

        st.dataframe(results[display_cols], use_container_width=True)

    # LLM Response
    st.divider()
    st.subheader("Request AI (GPT-4o mini) to generate the recommendation")

    if st.button("Generate AI Answer"):
        if best < MIN_SIMILARITY:
            st.warning(
                f"Low similarity can cause hallucination ({best:.2f})."
                )
        if results is None or results.empty:
            st.warning("Please run a search first to retrieve similar incidents.")
        else:
            with st.spinner("Analyzing similar incidents..."):
                evidence_text = build_evidence_text(results)
                answer = generate_answer(query, evidence_text)

            st.markdown("### Answer")
            st.write(answer)

    # Incident list
    st.divider()
    with st.expander("Incident List (Excel Data)", expanded=False):
        st.dataframe(df.drop(columns=["doc_text"], errors="ignore"), use_container_width=True)

except Exception as e:
    st.error("Could not load the Excel file")
    st.code(str(e))
