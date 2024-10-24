import streamlit as st

st.title("Methodology")

st.header("Data Flow and Implementation Details")

st.subheader("Overview")
st.write("""
Our application processes resumes and job descriptions through several key steps to match and rank candidates effectively.
""")

st.subheader("1. Data Input")
st.write("""
- **Automated Resume Extraction**:
  - Users input a job title/code.
  - Job descriptions are retrieved from pre-defined files.
  - Resumes are filtered by submission date.
- **Manual Upload**:
  - Users input a job description manually.
  - Resumes are uploaded directly by the user.
""")

st.subheader("2. Text Extraction")
st.write("""
- Resumes and job descriptions are processed to extract text content.
- Supports multiple file formats: PDF, DOCX, TXT.
""")

st.subheader("3. Data Processing")
st.write("""
- **Semantic Chunking**:
  - Resumes are split into meaningful chunks using `SemanticChunker` for better context understanding.
- **Embedding Generation**:
  - Text embeddings are generated using OpenAI's embedding models.
- **Skill Extraction and Weighting**:
  - Key skills are extracted from the job description and assigned importance weights.
  - Users can adjust skill weights to fine-tune the matching process.
- **Similarity Scoring**:
  - Cosine similarity is calculated between job description and resume embeddings to assess overall textual relevance.
  - Additional similarity is computed between job responsibilities and the job description for each job history.
- **Information Extraction**:
  - An LLM (e.g., GPT-4o-mini) extracts structured data from resumes, such as education, work experience, skills, and highlights.
""")

st.subheader("4. Filtering and Ranking")
st.write("""
- Resumes are filtered based on the minimum education level specified.
- Candidates are ranked according to combined similarity scores, which include:
  - **Textual Similarity**: Overall relevance of the resume to the job description.
  - **Skill Matching**: Alignment of candidate skills with job requirements, adjusted by user-defined weights.
- **Adjustable Skill Weights**:
  - Users can customize the importance of different skills in the matching process.
""")

st.subheader("5. Output and Visualization")
st.write("""
- Results are displayed in a ranked list with detailed summaries.
- Users can adjust filters and view the top N candidates.
- **Visualizations of Candidate Experience**:
  - Interactive charts display the relevancy of candidates' past experiences over time.
  - Hovering over data points shows detailed information such as candidate name and relevance score.
- **Interactive Hover Information**:
  - Provides specific data for the hovered line in visualizations, enhancing data interpretation.
- **Email Forwarding**:
  - Users can send summarized results to hiring managers directly from the application.
""")

st.subheader("Flowcharts")
st.write("The following flowcharts illustrate the process flow for each use case:")

# Include flowchart images (replace with your actual image paths)
st.image('images/flow chart.png', caption='Process Flow', use_column_width=True)
