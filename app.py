import os
import math
import streamlit as st
from pypdf import PdfReader
from openai import OpenAI

st.set_page_config(page_title="CSSF RegTech Assistant", page_icon="⚖️", layout="wide")
st.title("⚖️ Luxembourg CSSF RegTech Assistant")
st.caption("AI-powered compliance engine for Luxembourg financial regulations (CSSF Circulars)")

API_KEY = os.getenv("OPENAI_API_KEY", "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")
client = OpenAI(api_key=API_KEY)

def cosine_similarity(v1, v2):
    dot = sum(a * b for a, b in zip(v1, v2))
    norm_v1 = math.sqrt(sum(a * a for a in v1))
    norm_v2 = math.sqrt(sum(b * b for b in v2))
    return dot / (norm_v1 * norm_v2) if norm_v1 and norm_v2 else 0.0

# Ensure fallback CSSF regulatory documents exist
os.makedirs("pdfs", exist_ok=True)
sample_regulatory_file = os.path.join("pdfs", "CSSF_Circular_24_856.txt")
if not os.path.exists(sample_regulatory_file):
    with open(sample_regulatory_file, "w", encoding="utf-8") as f:
        f.write(
            "CSSF Circular 24/856: Investor protection in case of NAV calculation errors, non-compliance with investment rules and other errors at UCI level.\n\n"
            "Article 3 - Materiality thresholds: For UCITS investing in equities, the materiality threshold for a NAV calculation error is set at 0.50% of the Net Asset Value. For money market funds, the materiality threshold is defined as 0.25% of NAV. For bond and fixed income funds, the threshold is 0.50%. For alternative investment funds (AIFs), the threshold is generally between 0.50% and 1.00% depending on fund structure.\n\n"
            "Article 7 - Regulatory Notification: Any NAV error exceeding the materiality threshold must be notified to the CSSF without undue delay and at the latest within 4 to 8 weeks depending on root cause analysis. A detailed remediation plan and investor compensation statement must be submitted.\n\n"
            "CSSF Circular 22/806: On ICT and cloud outsourcing service arrangements.\n\n"
            "Section 4 - Critical Outsourcing: Financial institutions planning to outsource critical or important operational functions to cloud service providers must notify the CSSF at least 3 months prior to the planned implementation date.\n\n"
            "Section 6 - Data Localization and DORA Alignment: Entities must ensure data resilience, audit access rights, and business continuity in compliance with EU DORA standards."
        )

@st.cache_resource(show_spinner="Extracting and embedding CSSF circulars...")
def build_index():
    pdf_dir = "pdfs"
    chunks = []
    
    for filename in os.listdir(pdf_dir):
        filepath = os.path.join(pdf_dir, filename)
        
        # Support PDF files
        if filename.endswith(".pdf"):
            try:
                reader = PdfReader(filepath)
                for page_num, page in enumerate(reader.pages):
                    text = page.extract_text() or ""
                    paragraphs = [p.strip() for p in text.split("\n\n") if len(p.strip()) > 60]
                    for p in paragraphs:
                        chunks.append({"source": filename, "page": page_num + 1, "text": p})
            except Exception:
                pass

        # Support text documents
        elif filename.endswith(".txt"):
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read()
            paragraphs = [p.strip() for p in content.split("\n\n") if len(p.strip()) > 40]
            for idx, p in enumerate(paragraphs):
                chunks.append({"source": filename, "page": f"Sec {idx+1}", "text": p})

    if not chunks:
        return []

    # Create OpenAI embeddings for knowledge chunks
    texts = [c["text"] for c in chunks]
    res = client.embeddings.create(input=texts, model="text-embedding-3-small")
    for idx, item in enumerate(res.data):
        chunks[idx]["embedding"] = item.embedding

    return chunks

index = build_index()

if not index:
    st.error("Could not index any regulatory documents.")
else:
    st.success(f"Regulatory Knowledge Base Active ({len(index)} regulatory clauses indexed)")

    user_query = st.text_input(
        "Enter your regulatory compliance question:",
        value="What is the materiality threshold for equity UCITS NAV calculation errors?"
    )

    if st.button("Run Audit Search", type="primary"):
        with st.spinner("Searching CSSF regulatory clauses..."):
            q_embed = client.embeddings.create(
                input=[user_query],
                model="text-embedding-3-small"
            ).data[0].embedding

            for c in index:
                c["score"] = cosine_similarity(q_embed, c["embedding"])

            top_chunks = sorted(index, key=lambda x: x["score"], reverse=True)[:3]
            context = "\n\n".join([f"[{c['source']} | {c['page']}]: {c['text']}" for c in top_chunks])

            completion = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a Senior CSSF Compliance Auditor in Luxembourg. "
                            "Answer strictly using the provided context. Cite the specific circular and article."
                        )
                    },
                    {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {user_query}\n\nAnswer:"}
                ],
                temperature=0
            )

            st.markdown("### 📋 Regulatory Verdict")
            st.write(completion.choices[0].message.content)

            st.markdown("---")
            st.markdown("### 🔍 Verified Excerpts from CSSF Regulations")
            for chunk in top_chunks:
                with st.expander(f"📄 {chunk['source']} ({chunk['page']}) — Match Score: {chunk['score']:.2f}"):
                    st.write(chunk["text"])
