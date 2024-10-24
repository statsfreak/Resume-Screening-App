import streamlit as st
st.set_page_config(layout="wide")


st.title("About Us")

st.header("Project Scope and Objectives")
st.write("""
The **Automated Screening Tool for Recruitment Operations (ASTRO)** is designed to streamline the recruitment process by leveraging artificial intelligence and natural language processing technologies. Our application aims to:

- **Automate resume extraction and matching** to job descriptions.
- **Rank candidates** based on relevance and predefined criteria.
- **Enhance efficiency** in the recruitment workflow.
- **Provide insightful summaries and visualizations** to assist hiring decisions.
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
- **Advanced Matching and Ranking**: Use AI models to match and rank resumes based on textual similarity and skill matching.
- **Adjustable Skill Weights**: Customize the importance of different skills in the matching process.
- **Education Level Filtering**: Filter candidates by minimum education requirements.
- **Relevance Scoring for Job Histories**: Calculate relevance scores for individual job experiences within a candidate's resume, considering both skill matching and responsibilities similarity.
- **Visualizations of Candidate Experience**: Visualize the relevancy of candidates' past experiences over time with interactive charts.
- **Interactive Hover Information**: Display detailed information when hovering over data points in visualizations, such as candidate name and relevance score.
- **Email Forwarding**: Send summarized results to hiring managers directly from the application.
""")

st.header("Usage Disclaimer")
st.warning("""
**Disclaimer**

IMPORTANT NOTICE: This web application is developed as a proof-of-concept prototype. The information provided here is NOT intended for actual usage and should not be relied upon for making any decisions, especially those related to financial, legal, or healthcare matters.

Furthermore, please be aware that the AI models may generate inaccurate or incorrect information. You assume full responsibility for how you use any generated output.

Always consult with qualified professionals for accurate and personalized advice.
""")

