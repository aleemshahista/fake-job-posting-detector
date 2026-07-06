"""
app.py
------
Streamlit web interface for the Fake Job Posting Detector.
Run with:  streamlit run app.py
"""

import streamlit as st
from src.predict import predict_job_posting


# ──────────────────────────────────────────────────────────────
# Page config
# ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Fake Job Posting Detector",
    page_icon="🔍",
    layout="centered",
)

# ──────────────────────────────────────────────────────────────
# Custom CSS — premium dark glassmorphic theme
# ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* ---------- Google Font ---------- */
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap');

    /* ---------- Global ---------- */
    html, body, p, h1, h2, h3, h4, h5, h6, span, div, input, textarea, button, label, a {
        font-family: 'Outfit', sans-serif;
    }
    /* Preserve Streamlit's icon fonts */
    .stExpander [data-testid="stExpanderToggleIcon"],
    [data-testid="stIconMaterial"] {
        font-family: 'Material Symbols Rounded', sans-serif !important;
    }
    .stApp {
        background: linear-gradient(145deg, #0f0c29, #1a1a40, #24243e);
    }

    /* ---------- Header ---------- */
    .hero-title {
        text-align: center;
        font-size: 2.6rem;
        font-weight: 700;
        background: linear-gradient(135deg, #a78bfa, #7c3aed, #06b6d4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.15rem;
    }
    .hero-sub {
        text-align: center;
        font-size: 1.05rem;
        color: #a5b4cb;
        margin-bottom: 2rem;
    }

    /* ---------- Result cards ---------- */
    .result-card {
        border-radius: 16px;
        padding: 2rem 2.4rem;
        text-align: center;
        animation: fadeUp 0.5s ease-out;
        margin-bottom: 0.5rem;
    }
    @keyframes fadeUp {
        from { opacity: 0; transform: translateY(18px); }
        to   { opacity: 1; transform: translateY(0); }
    }
    .result-real {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.12), rgba(6, 182, 212, 0.10));
        border: 1px solid rgba(16, 185, 129, 0.35);
        box-shadow: 0 0 32px rgba(16, 185, 129, 0.12);
    }
    .result-fake {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.12), rgba(245, 158, 11, 0.10));
        border: 1px solid rgba(239, 68, 68, 0.35);
        box-shadow: 0 0 32px rgba(239, 68, 68, 0.12);
    }
    .result-emoji { font-size: 3rem; margin-bottom: 0.5rem; }
    .result-label {
        font-size: 1.6rem;
        font-weight: 700;
        margin-bottom: 0.3rem;
    }
    .label-real { color: #34d399; }
    .label-fake { color: #f87171; }
    .result-confidence {
        font-size: 1rem;
        color: #94a3b8;
    }

    /* ---------- Section headers ---------- */
    .section-header {
        font-size: 1.1rem;
        font-weight: 600;
        color: #c4b5fd;
        margin-bottom: 0.4rem;
    }

    /* ---------- Sample button row ---------- */
    .sample-label {
        font-size: 0.85rem;
        color: #94a3b8;
        margin-bottom: 0.3rem;
    }

    /* ---------- Footer ---------- */
    .footer {
        text-align: center;
        color: #475569;
        font-size: 0.8rem;
        margin-top: 3rem;
        padding-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────
# Sample templates (for quick testing)
# ──────────────────────────────────────────────────────────────
SAMPLES = {
    "🟢 Real Job": {
        "title": "Senior Software Engineer",
        "company_profile": (
            "We are a leading fintech company headquartered in New York "
            "with over 500 employees worldwide."
        ),
        "description": (
            "We are seeking an experienced software engineer to join our "
            "backend team. You will design and build scalable microservices, "
            "mentor junior engineers, and participate in architectural reviews."
        ),
        "requirements": (
            "5+ years of experience in Python or Java. "
            "Strong understanding of distributed systems and SQL databases."
        ),
        "benefits": (
            "Competitive salary, equity, health insurance, "
            "401(k) matching, and flexible remote work."
        ),
    },
    "🔴 Fake Job": {
        "title": "Data Entry Clerk - Work From Home",
        "company_profile": "",
        "description": (
            "We are looking for a data entry clerk to work from home. "
            "No experience needed. Earn $5000 per week! "
            "Send your personal and bank details to get started immediately."
        ),
        "requirements": "No requirements needed. Anyone can apply.",
        "benefits": "Unlimited income potential. Be your own boss!",
    },
}


# ──────────────────────────────────────────────────────────────
# Hero header
# ──────────────────────────────────────────────────────────────
st.markdown('<div class="hero-title">🔍 Fake Job Posting Detector</div>',
            unsafe_allow_html=True)
st.markdown(
    '<div class="hero-sub">'
    "Paste a job posting below and our ML model will tell you if it's "
    "legitimate or potentially fraudulent."
    "</div>",
    unsafe_allow_html=True,
)


# ──────────────────────────────────────────────────────────────
# Sample buttons
# ──────────────────────────────────────────────────────────────
st.markdown('<p class="sample-label">⚡ Quick fill with a sample:</p>',
            unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    if st.button("🟢  Load Real Job Sample", use_container_width=True):
        st.session_state["sample"] = SAMPLES["🟢 Real Job"]

with col2:
    if st.button("🔴  Load Fake Job Sample", use_container_width=True):
        st.session_state["sample"] = SAMPLES["🔴 Fake Job"]

# Retrieve current sample (if any)
sample = st.session_state.get("sample", {})


# ──────────────────────────────────────────────────────────────
# Input form
# ──────────────────────────────────────────────────────────────
# Use Streamlit's native container to avoid broken HTML div wrapping
with st.container(border=True):
    st.markdown('<p class="section-header">📝 Job Posting Details</p>',
                unsafe_allow_html=True)

    title = st.text_input("Job Title", value=sample.get("title", ""),
                           placeholder="e.g. Senior Data Analyst")

    company_profile = st.text_area(
        "Company Profile", value=sample.get("company_profile", ""),
        height=80, placeholder="Brief description of the company …",
    )

    description = st.text_area(
        "Job Description", value=sample.get("description", ""),
        height=140, placeholder="Describe the role and responsibilities …",
    )

    requirements = st.text_area(
        "Requirements", value=sample.get("requirements", ""),
        height=100, placeholder="Required skills and qualifications …",
    )

    benefits = st.text_area(
        "Benefits", value=sample.get("benefits", ""),
        height=80, placeholder="Salary, perks, work-from-home policy …",
    )

# ──────────────────────────────────────────────────────────────
# Analyze button & results
# ──────────────────────────────────────────────────────────────
analyze = st.button("🚀  Analyze Posting", type="primary",
                     use_container_width=True)

if analyze:
    # Require at least some text
    if not any([title.strip(), description.strip()]):
        st.warning("Please enter at least a **Job Title** or **Description**.")
    else:
        with st.spinner("Analyzing the job posting …"):
            job_fields = {
                "title": title,
                "company_profile": company_profile,
                "description": description,
                "requirements": requirements,
                "benefits": benefits,
            }

            result = predict_job_posting(job_fields)

        # Build result card
        if result["label"] == "Real":
            card_class = "result-real"
            label_class = "label-real"
            emoji = "✅"
            verdict = "Legitimate Job Posting"
        else:
            card_class = "result-fake"
            label_class = "label-fake"
            emoji = "⚠️"
            verdict = "Potentially Fraudulent Posting"

        confidence_pct = f"{result['confidence']:.1%}"

        st.markdown(
            f"""
            <div class="result-card {card_class}">
                <div class="result-emoji">{emoji}</div>
                <div class="result-label {label_class}">{verdict}</div>
                <div class="result-confidence">
                    Model confidence: <strong>{confidence_pct}</strong>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Expandable detail
        with st.expander("Detailed Scores", icon="📊"):
            st.metric("Prediction", result["label"])
            st.metric("Confidence", confidence_pct)
            st.metric("P(Fake)", f"{result['probability']:.1%}")

# ──────────────────────────────────────────────────────────────
# Footer
# ──────────────────────────────────────────────────────────────
st.markdown(
    '<div class="footer">'
    "Built with Streamlit · ML-powered by scikit-learn · "
    "Fake Job Posting Detector © 2026"
    "</div>",
    unsafe_allow_html=True,
)
