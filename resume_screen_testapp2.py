
import streamlit as st
import pandas as pd
import docx2txt
import PyPDF2
import json
import io
import re
import openai
import os
import tempfile
import base64
from dotenv import load_dotenv
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from langchain_experimental.text_splitter import SemanticChunker
from langchain.docstore.document import Document
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Load environment variables from the .env file
load_dotenv('.env')

# Fetch the OpenAI API key from the environment variables
openai_api_key = os.getenv('OPENAI_API_KEY')

# Initialize OpenAI client
openai.api_key = openai_api_key

# Function to extract text from PDF files
def extract_text_from_pdf(file):
    text = ""
    reader = PyPDF2.PdfReader(file)
    for page in reader.pages:
        text += page.extract_text()
    return text

# Function to extract text from DOCX files
def extract_text_from_docx(file):
    return docx2txt.process(file)

# Function to extract text from TXT files
def extract_text_from_txt(file):
    return file.read().decode("utf-8")

# Function to extract text from different file types
def extract_text(uploaded_file):
    if uploaded_file.type == "application/pdf":
        return extract_text_from_pdf(uploaded_file)
    elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        return extract_text_from_docx(uploaded_file)
    elif uploaded_file.type == "text/plain":
        return extract_text_from_txt(uploaded_file)
    else:
        return ""
    

# Streamlit app
st.title("Resume Matcher with Advanced Retrieval")

# Input for job description
job_description = st.text_area("Enter Job Description")

# Upload resumes
uploaded_files = st.file_uploader("Upload Resumes", accept_multiple_files=True, type=["pdf", "docx", "txt"])

# Input for number of top resumes to display
top_n = st.number_input("Enter the number of top resumes to display", min_value=1, value=5, step=1)

# Add a button to run the resume matching
if st.button("Match Resumes"):
    if job_description and uploaded_files:
        resume_texts = []
        filenames = []
        file_links = []  # List to store file links

        for uploaded_file in uploaded_files:
            try:
                file_content = extract_text(uploaded_file)
                resume_texts.append(file_content)
                filenames.append(uploaded_file.name)

                # Save the file to a temporary location
                with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
                    tmp_file.write(uploaded_file.getbuffer())
                    tmp_file_path = tmp_file.name

                # Generate a download link for the file
                with open(tmp_file_path, 'rb') as f:
                    file_bytes = f.read()

                # Create a download link and store the HTML
                b64 = base64.b64encode(file_bytes).decode()
                href = f'<a href="data:file/{uploaded_file.type};base64,{b64}" download="{uploaded_file.name}">{uploaded_file.name}</a>'
                file_links.append(href)

            except Exception as e:
                st.error(f"Error processing file {uploaded_file.name}: {e}")

        # Use the SemanticChunker to split resumes into meaningful chunks
        embeddings_model = OpenAIEmbeddings(model='text-embedding-3-small')
        text_splitter = SemanticChunker(embeddings_model)  # Semantic chunker instead of token-based

        # Wrap text content into Document objects for LangChain processing
        documents = [Document(page_content=text, metadata={"filename": filenames[i]}) for i, text in enumerate(resume_texts)]
        semantic_chunks = []
        for doc in documents:
            chunks = text_splitter.split_documents([doc])
            semantic_chunks.extend(chunks)

        # Store chunks and embeddings in Chroma for retrieval
        vectordb = Chroma.from_documents(semantic_chunks, embeddings_model)

        # Create a LangChain RetrievalQA pipeline
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        qa_chain = RetrievalQA.from_chain_type(
            retriever=vectordb.as_retriever(),
            llm=llm
        )

        # Embed the job description for similarity scoring
        job_desc_embedding = embeddings_model.embed_query(job_description)

        # Generate summaries and calculate similarity scores
        results = []
        for idx, resume_text in enumerate(resume_texts):
            # Get the embedding for the current resume
            resume_embedding = embeddings_model.embed_query(resume_text)

            # Calculate cosine similarity between job description and resume
            similarity_score = cosine_similarity(
                [job_desc_embedding], [resume_embedding]
            )[0][0]

            # Modify the prompt to request JSON-only output
            result = qa_chain.run(f"""
            For the following resume, extract and return the following details in JSON format only. Do not include any additional text or explanations—just the JSON data.

            Required details:
            - Name
            - Area of study
            - Summary
            - Work experiences (list of job experiences), where each job experience includes:
                - Job title
                - Company name
                - Start date
                - End date

            Example output:
            {{
                "Name": "John Doe",
                "Area of study": "Bachelor in Computer Science",
                "Summary": "Experienced software engineer with expertise in Python and machine learning.",
                "Work experiences": [
                    {{"Job title": "Software Engineer", "Company name": "ABC Corp", "Start date": "Jan 2020", "End date": "Present"}},
                    {{"Job title": "Junior Developer", "Company name": "XYZ Ltd", "Start date": "May 2017", "End date": "Dec 2019"}}
                ]
            }}
            Return only the JSON data. Do not include any additional text.

            Resume: {resume_text[:1000]}...
            """)

            # Try parsing the result as JSON
            try:
                # Extract the JSON object from the result
                json_match = re.search(r'\{.*\}', result, re.DOTALL)
                if json_match:
                    json_str = json_match.group()
                    parsed_result = json.loads(json_str)
                else:
                    raise ValueError("No JSON object found in the model's output.")
            except Exception as e:
                st.error(f"Failed to parse result for {filenames[idx]}: {e}")
                st.write("Model's raw output:")
                st.write(result)
                parsed_result = {}

            name = parsed_result.get('Name', 'N/A')
            area_of_study = parsed_result.get('Area of study', 'N/A')
            skills_summary = parsed_result.get('Summary', 'N/A')
            job_experiences = parsed_result.get('Work experiences', [])

            # Proceed with processing job_experiences as before
            formatted_roles = []
            for job in job_experiences:
                job_title = job.get('Job title', 'N/A')
                company = job.get('Company name', 'N/A')
                start_date = job.get('Start date', 'N/A')
                end_date = job.get('End date', 'N/A')
                role_text = f"{job_title} in {company} ({start_date} - {end_date})"

                # Calculate relevance score for this specific job role
                role_embedding = embeddings_model.embed_query(role_text)
                relevance_score = cosine_similarity(
                    [job_desc_embedding], [role_embedding]
                )[0][0] * 10  # Scale the relevance score (0-10 scale)

                # Format the role with relevance score
                formatted_roles.append(f"{job_title} in {company} [Relevant score: {relevance_score:.1f}], {start_date} - {end_date}")

            # Append the summary, similarity score, and relevant roles to the results
            results.append({
                "Filename": file_links[idx],
                "Name": name,
                "Area of Study": area_of_study,
                "Relevant Years of Experience": '<br>'.join(formatted_roles),
                "Summary": skills_summary,
                "Similarity Score": similarity_score
            })

        # Sort results by similarity score in descending order
        results = sorted(results, key=lambda x: x["Similarity Score"], reverse=True)

        # Display the results in a DataFrame
        df_results = pd.DataFrame(results)
        df_results.index = df_results.index + 1
        st.write(f"Top {top_n} Resumes")

        # Adjust pandas display options
        pd.set_option('display.max_colwidth', None)

        # Apply styling to adjust cell height and allow HTML content
        styled_df = df_results.style.set_properties(**{
            'white-space': 'pre-wrap',
            'word-wrap': 'break-word',
            'text-align': 'left',
            'vertical-align': 'top'
        })

        # Render the DataFrame with clickable links
        st.write(styled_df.to_html(escape=False, index=False), unsafe_allow_html=True)

    else:
        st.error("Please enter a job description and upload at least one resume.")
