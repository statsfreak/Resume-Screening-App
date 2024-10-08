
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
- **Similarity Scoring**:
  - Cosine similarity is calculated between job description and resume embeddings to assess relevance.
- **Information Extraction**:
  - An LLM (e.g., GPT-4) extracts structured data from resumes, such as education, work experience, and highlights.
""")

st.subheader("4. Filtering and Ranking")
st.write("""
- Resumes are filtered based on the minimum education level specified.
- Candidates are ranked according to similarity scores.
""")

st.subheader("5. Output and Visualization")
st.write("""
- Results are displayed in a ranked list with detailed summaries.
- Users can adjust filters and view the top N candidates.
- Provides an option to forward the summaries via email.
""")

st.subheader("Flowcharts")
st.write("The following flowcharts illustrate the process flow for each use case:")

# Include flowchart images (replace with your actual image paths)
#st.image('images/manual_upload_flowchart.png', caption='Manual Upload Process Flow', use_column_width=True)
#st.image('images/automated_extraction_flowchart.png', caption='Automated Resume Extraction Process Flow', use_column_width=True)
