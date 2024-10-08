
import streamlit as st

st.title("About Us")

st.header("Project Scope and Objectives")
st.write("""
The **Resume Matcher with Advanced Retrieval** is designed to streamline the recruitment process by leveraging artificial intelligence and natural language processing technologies. Our application aims to:

- **Automate resume extraction and matching** to job descriptions.
- **Rank candidates** based on relevance and predefined criteria.
- **Enhance efficiency** in the recruitment workflow.
- **Provide insightful summaries** to assist hiring decisions.
""")

st.header("Data Sources")
st.write("""
- **Resumes**: Uploaded manually by users or extracted automatically from a simulated database based on submission dates.
- **Job Descriptions**: Retrieved automatically using job title or code, simulating an integration with an Applicant Tracking System (ATS) or database.
""")

st.header("Features")
st.write("""
- **Automated Resume Extraction**: Fetch resumes submitted within a specified date range.
- **Manual Resume Upload**: Upload resumes directly for processing.
- **Job Description Retrieval**: Obtain job descriptions based on job title or code.
- **Advanced Matching and Ranking**: Use AI models to match and rank resumes.
- **Education Level Filtering**: Filter candidates by minimum education requirements.
- **Email Forwarding**: Send summarized results to hiring managers.
""")
