# Job history score calculated based on similarity of skill with job description
# #__import__('pysqlite3')
import sys

# Try to import pysqlite3 if available
try:
    pysqlite3 = __import__('pysqlite3')
    # Only swap sqlite3 with pysqlite3 if it's actually in sys.modules
    if 'pysqlite3' in sys.modules:
        sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
    print("Using pysqlite3 for SQLite3.")
except ImportError:
    print("pysqlite3 not found, using the built-in sqlite3.")

import streamlit as st
import pandas as pd
import docx2txt
import PyPDF2
import json
import re
import openai
import os
import datetime
from dotenv import load_dotenv
from PIL import Image
import numpy as np
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQA
#from langchain.chat_models import ChatOpenAI
from langchain_openai import ChatOpenAI
#from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain.docstore.document import Document
from sklearn.metrics.pairwise import cosine_similarity
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.application import MIMEApplication
import tiktoken
from helper_functions.utility import check_password 
import plotly.express as px 
#sys.modules['sqlite3'] = sys.modules.pop('pysqlite3') # in order for chromadb to run need to set this

# Load environment variables from the .env file
if load_dotenv('.env'):
    # For local development
    openai_api_key = os.getenv('OPENAI_API_KEY')
    sender_email = os.getenv('SENDER_EMAIL')
    sender_password = os.getenv('SENDER_PASSWORD')
else:
    openai_api_key = st.secrets['OPENAI_API_KEY']
    sender_email = st.secrets['SENDER_EMAIL']
    sender_password = st.secrets['SENDER_PASSWORD']

# Initialize OpenAI client
openai.api_key = openai_api_key

# **Initialize Session State Variables**
if 'original_skill_weights' not in st.session_state:
    st.session_state.original_skill_weights = {}
if 'adjusted_skill_weights' not in st.session_state:
    st.session_state.adjusted_skill_weights = {}
if 'embeddings_model' not in st.session_state:
    st.session_state.embeddings_model = OpenAIEmbeddings(model='text-embedding-3-small')
if 'vectordb' not in st.session_state:
    st.session_state.vectordb = None
if 'skill_embeddings' not in st.session_state:
    st.session_state.skill_embeddings = {}
if 'job_desc_embedding' not in st.session_state:
    st.session_state.job_desc_embedding = None
if 'matched_resumes' not in st.session_state:
    st.session_state.matched_resumes = False
if 'results' not in st.session_state:
    st.session_state.results = None
#if 'df_results' not in st.session_state:
#    st.session_state.df_results = None
if 'top_n_filtered' not in st.session_state:
    st.session_state.top_n_filtered = None
if 'resume_texts' not in st.session_state:
    st.session_state.resume_texts = []
if 'filenames' not in st.session_state:
    st.session_state.filenames = []

# ** Caching Expensive Operations**
@st.cache_resource
def get_embeddings_model():
    return OpenAIEmbeddings(model='text-embedding-3-small')

@st.cache_resource
def get_llm():
    return ChatOpenAI(model="gpt-4o-mini", temperature=0)

@st.cache_resource
def create_vector_store(_semantic_chunks, _embeddings_model):
    return Chroma.from_documents(_semantic_chunks, _embeddings_model)

@st.cache_resource
def get_qa_chain(_vectordb, _llm):
    """

    Initializes and returns a RetrievalQA chain using the provided vector store and LLM.
    """
    return RetrievalQA.from_chain_type(
        llm=_llm,
        chain_type="stuff",  # You can choose other chain types based on your needs
        retriever=_vectordb.as_retriever()
    )


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
    if uploaded_file.name.lower().endswith(".pdf"):
        return extract_text_from_pdf(uploaded_file)
    elif uploaded_file.name.lower().endswith(".docx"):
        return extract_text_from_docx(uploaded_file)
    elif uploaded_file.name.lower().endswith(".txt"):
        return extract_text_from_txt(uploaded_file)
    else:
        return ""

# Function to send email with attachment
def send_email(sender_email, sender_password, recipient_email, subject, message, attachment=None, attachment_filename='attachment.csv'):
    try:
        # Create the email message
        msg = MIMEMultipart()
        msg['From'] = sender_email
        msg['To'] = recipient_email
        msg['Subject'] = subject

        # Add message body
        msg.attach(MIMEText(message, 'plain'))

        # Attach the CSV file if provided
        if attachment is not None:
            part = MIMEApplication(attachment, Name=attachment_filename)
            part['Content-Disposition'] = f'attachment; filename="{attachment_filename}"'
            msg.attach(part)

        # Connect to SMTP server using STARTTLS
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(sender_email, sender_password)

        # Send the email
        server.sendmail(sender_email, recipient_email, msg.as_string())
        server.quit()
        return True
    except Exception as e:
        st.error(f"Error sending email: {e}")
        return False

def extract_and_rank_skills(job_description, llm):
    """
    Extracts skills from the job description and assigns importance weights.
    
    Args:
        job_description (str): The job description text.
        llm: The language model to use for extraction.
    
    Returns:
        tuple: A tuple containing a list of extracted skills and a dictionary mapping skills to their weights.
    """
    prompt = f"""
    Extract a list of key skills required for the following job description. For each skill, assign an importance score between 1 and 5, where 5 is most important.

    Job Description:
    {job_description}

    Provide the output in JSON format with each skill and its corresponding weight ranging from 1 to 10. Return the following details in JSON format only. Do not include any additional text or explanations—just the JSON data.Example:

    {{
        "Python": 10,
        "Machine Learning": 10,
        "Data Analysis": 8,
        "SQL": 6,
        "Data Visualization": 4
    }}
    Return only the JSON data. Do not include any additional text.
    """
    messages = [
        {"role": "system", "content": "You are an assistant that extracts and ranks skills from job descriptions."},
        {"role": "user", "content": prompt}
    ]
    response = llm(messages)

    # Extract the content from the AIMessage object
    if hasattr(response, 'content'):
        response_text = response.content
    else:
        st.error("Unexpected response format from the language model.")
        st.write("Response object:", response)
        return [], {}

    try:
        skills_data = json.loads(response_text)
    except json.JSONDecodeError:
        st.error("Failed to parse skills extraction response.")
        st.write("Model's raw output:")
        st.write(response)
        skills_data = {}

    skills = list(skills_data.keys())
    skill_weights = {skill.lower(): weight for skill, weight in skills_data.items()}

    return skills, skill_weights

def process_resumes_initial(job_description, resume_texts, filenames):
    """
    Initial processing: embedding, chunking, retrieval setup.
    Extract and store embeddings and vector store.
    """
    embeddings_model = st.session_state.embeddings_model
    llm = get_llm()
    
    # Wrap text content into Document objects for LangChain processing
    documents = [Document(page_content=text, metadata={"filename": filenames[i]}) for i, text in enumerate(resume_texts)]
    
    # Use the SemanticChunker to split resumes into meaningful chunks
    text_splitter = SemanticChunker(embeddings_model)  # Semantic chunker instead of token-based
    semantic_chunks = []
    for doc in documents:
        chunks = text_splitter.split_documents([doc])
        semantic_chunks.extend(chunks)
    
    # Store chunks and embeddings in Chroma for retrieval
    vectordb = create_vector_store(_semantic_chunks=semantic_chunks, _embeddings_model=embeddings_model)
    st.session_state.vectordb = vectordb

    # Create the QA chain and store it
    qa_chain = get_qa_chain(_vectordb=vectordb, _llm=llm)
    st.session_state.qa_chain = qa_chain
    
    # Embed the job description for similarity scoring
    job_desc_embedding = embeddings_model.embed_query(job_description)
    st.session_state.job_desc_embedding = job_desc_embedding

    # **Store the job_description in session state**
    st.session_state.job_description = job_description
    
    # Extract and rank skills
    job_skills, skill_weights = extract_and_rank_skills(job_description, llm)
    st.session_state.original_skill_weights = skill_weights.copy()
    st.session_state.adjusted_skill_weights = skill_weights.copy()
    
    # Generate embeddings for job skills 
    skill_embeddings = {skill.lower(): embeddings_model.embed_query(skill) for skill in job_skills}
    st.session_state.skill_embeddings = skill_embeddings
    
    return job_skills

def calculate_similarity_scores(resume_texts, filenames, job_desc_embedding, skill_embeddings, adjusted_skill_weights, embeddings_model, qa_chain):
    """
    Calculate similarity scores using adjusted skill weights.
    """
    results = []
    education_level_mapping = {
        'O level or equivalent': 1,
        'A level or equivalent': 2,
        'Diploma': 3,
        'Degree': 4,
        'Graduate Degree': 5,
        'Other': 0
    }
    
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
        - Education (list all tertiary degrees and above obtained, including degree name and field of study. If no tertiary degree then state the highest education achieved)
        - Map the highest obtained education level to one of the following categories: "O level or equivalent", "A level or equivalent", "Diploma", "Degree", "Graduate Degree", "Other"
        - Provide the mapped education level as "Mapped Education Level"
        - Highlights of Career (2-3 standout sentences from the resume)
        - Skills (list all relevant technical and soft skills)
        - Work experiences (list of job experiences), where each job experience includes:
            - Job title
            - Company name
            - Start date
            - End date
            - Responsibilities (list of key responsibilities or achievements)
            - Job Skills (list of skills used in this job)


    
        Example output:
        {{
            "Name": "John Doe",
            "Education": [
                "Bachelor in Computer Science",
                "Master of Business Administration"
            ],
            "Mapped Education Level": "Graduate Degree",
            "Highlights of Career": [
                "Led a team of developers to create a cutting-edge AI application.",
                "Increased company revenue by 20% through innovative solutions."
            ],
            "Skills": [
                "Python",
                "Machine Learning",
                "Data Analysis",
                "SQL",
                "Data Visualization"
            ],
            "Work experiences": [
                {{"Job title": "Software Engineer", "Company name": "ABC Corp", "Start date": "Jan 2020", "End date": "Present", "Responsibilities": ["Developed AI models for predictive analytics.", "Collaborated with cross-functional teams to implement data-driven solutions."], "Job Skills": ["Python", "Machine Learning", "TensorFlow"]}},
                {{"Job title": "Junior Developer", "Company name": "XYZ Ltd", "Start date": "May 2017", "End date": "Dec 2019", "Responsibilities": ["Assisted in developing web applications.", "Maintained databases and ensured data integrity."], "Job Skills": ["JavaScript", "React.js", "SQL"]}}
            ]
        }}
        Return only the JSON data. Do not include any additional text.
    
        Resume: {resume_text}...
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
        education = parsed_result.get('Education', [])
        mapped_education_level = parsed_result.get('Mapped Education Level', 'Other')
        highlights = parsed_result.get('Highlights of Career', [])
        skills = parsed_result.get('Skills', [])  # Extracted skills
        job_experiences = parsed_result.get('Work experiences', [])
    
        # Calculate Weighted Skill Similarity Score Using Adjusted Weights
        matched_skills = set()
        resume_skill_embeddings = {skill: embeddings_model.embed_query(skill) for skill in skills}
    
        for job_skill, job_emb in skill_embeddings.items():
            job_emb_array = np.array(job_emb).reshape(1, -1)
            for resume_skill, resume_emb in resume_skill_embeddings.items():
                resume_emb_array = np.array(resume_emb).reshape(1, -1)
                similarity = cosine_similarity(job_emb_array, resume_emb_array)[0][0]
                if similarity >= 0.8:  # Threshold for considering a match
                    matched_skills.add(job_skill)
                    break  # Move to the next job_skill after a match is found
    
        skill_score = 0
        for skill in matched_skills:
            weight = adjusted_skill_weights.get(skill.lower(), 1)  # Use adjusted weights
            skill_score += weight
    
        # Normalize skill_score by the maximum possible score
        max_skill_score = sum(adjusted_skill_weights.values())
        normalized_skill_score = skill_score / max_skill_score  # Value between 0 and 1
    
        # Combined Similarity Score
        overall_weight = 0.5  # Weight for textual similarity
        skill_weight = 0.5    # Weight for skill similarity
        combined_similarity = (overall_weight * similarity_score) + (skill_weight * normalized_skill_score)
    
        # Map candidate's education level to numeric value
        candidate_education_level = education_level_mapping.get(mapped_education_level, 0)
    
        # Format work experiences
        formatted_roles = []
        for job in job_experiences:
            job_title_role = job.get('Job title', 'N/A')
            company = job.get('Company name', 'N/A')
            start_date_role = job.get('Start date', 'N/A')
            end_date_role = job.get('End date', 'N/A')
            responsibilities = job.get('Responsibilities', [])  # Extract responsibilities
            responsibilities_text = ' '.join(responsibilities)
            #role_text = f"{job_title_role} in {company} ({start_date_role} - {end_date_role}) {responsibilities_text}"
            job_skills = job.get('Job Skills', [])  # Extract job-specific skills

            # Identify matched skills for this job based on embeddings
            matched_skills_in_job = {}
            for skill in job_skills:
                if skill.lower() in skill_embeddings:
                    # Compute similarity with all resume skills
                    job_emb = skill_embeddings[skill.lower()]
                    job_emb_array = np.array(job_emb).reshape(1, -1)
                    for resume_skill, resume_emb in resume_skill_embeddings.items():
                        resume_emb_array = np.array(resume_emb).reshape(1, -1)
                        similarity = cosine_similarity(job_emb_array, resume_emb_array)[0][0]
                        if similarity >= 0.5:  # Threshold for considering a match
                            matched_skills_in_job[skill] = {'similarity': similarity}
                            break  # Move to the next job_skill after a match is found

            # Calculate skill score for the job based on matched skills
            skill_score_job = 0
            for skill, match_info in matched_skills_in_job.items():
                weight = adjusted_skill_weights.get(skill.lower(), 1)  # Use adjusted weights
                skill_score_job += weight * match_info['similarity']
    
            # Normalize skill score
            max_skill_score_job = sum(adjusted_skill_weights.values())
            #normalized_skill_score_job = skill_score_job / max_skill_score_job  # Value between 0 and 1
            if max_skill_score_job > 0:
                normalized_skill_score_job = skill_score_job / max_skill_score_job  # Value between 0 and 1
            else:
                normalized_skill_score_job = 0  # Avoid division by zero
    
            # Compute responsibilities embedding
            if responsibilities_text.strip():
                responsibilities_embedding = embeddings_model.embed_query(responsibilities_text)
                responsibilities_embedding_array = np.array(responsibilities_embedding).reshape(1, -1)

                # Compute cosine similarity between responsibilities and job description
                job_desc_embedding_array = np.array(job_desc_embedding).reshape(1, -1)
                responsibilities_similarity = cosine_similarity(job_desc_embedding_array, responsibilities_embedding_array)[0][0]
            else:
                responsibilities_similarity = 0  # No responsibilities text

            # Now combine the skill-based score and the responsibilities similarity
            weight_skill_score = 0.5
            weight_responsibilities_similarity = 0.5

            combined_relevance_score = (weight_skill_score * normalized_skill_score_job) + (weight_responsibilities_similarity * responsibilities_similarity)

            # Scale to 0-10
            relevance_score = combined_relevance_score * 10

            # Format the role with relevance score
            formatted_roles.append(f"{job_title_role} in {company} [Relevant score: {relevance_score:.1f}], {start_date_role} - {end_date_role}")

            
            # Calculate relevance score for this specific job role
            #role_embedding = embeddings_model.embed_query(role_text)
            #role_embedding = embeddings_model.embed_query(responsibilities_text)
            #relevance_score = cosine_similarity(
            #    [job_desc_embedding], [role_embedding]
            #)[0][0] * 10  # Scale the relevance score (0-10 scale)
    
            # Format the role with relevance score
            #formatted_roles.append(f"{job_title_role} in {company} [Relevant score: {relevance_score:.1f}], {start_date_role} - {end_date_role}")
    
        # Append the summary, similarity score, and relevant roles to the results
        results.append({
            "Filename": filenames[idx],
            "Name": name,
            "Education": education,
            "Mapped Education Level": mapped_education_level,
            "Candidate Education Level": candidate_education_level,
            "Highlights": highlights,
            "Skills": skills,  # Include extracted skills
            "Relevant Years of Experience": '\n'.join(formatted_roles),
            "Similarity Score": combined_similarity,
            "Resume Text": resume_text  # Include the resume text
        })
    
    return results

def parse_relevant_experience(experience_str, candidate_id):
    """
    Parses the 'Relevant Years of Experience' string to extract job details.
    
    Args:
        experience_str (str): The experience string from the DataFrame.
        candidate_id (str): Unique identifier for the candidate (e.g., Filename).
    
    Returns:
        list of dicts: Each dict contains job_title, company, relevance_score, start_date, end_date, candidate_id.
    """
    pattern = r'(?P<JobTitle>.*?) in (?P<Company>.*?) \[Relevant score: (?P<Score>\d+\.?\d*)\], (?P<StartDate>.*?) - (?P<EndDate>.*?)$'
    experiences = []
    for line in experience_str.split('\n'):
        match = re.match(pattern, line.strip())
        if match:
            job_title = match.group('JobTitle').strip()
            company = match.group('Company').strip()
            score = float(match.group('Score'))
            start_date_str = match.group('StartDate').strip()
            end_date_str = match.group('EndDate').strip()
            
            # Parse dates to datetime objects
            start_date = parse_date(start_date_str)
            end_date = parse_date(end_date_str)

            # Extract responsibilities from the line if available
            responsibilities_match = re.search(r'Responsibilities: (.*)', line, re.IGNORECASE)
            if responsibilities_match:
                responsibilities = responsibilities_match.group(1).split(';')  # Assuming responsibilities are separated by semicolons
                responsibilities = [resp.strip() for resp in responsibilities]
            else:
                responsibilities = []

            # Extract job-specific skills from the line if available
            job_skills_match = re.search(r'Job Skills: (.*)', line, re.IGNORECASE)
            if job_skills_match:
                job_skills = job_skills_match.group(1).split(';')  # Assuming skills are separated by semicolons
                job_skills = [skill.strip() for skill in job_skills]
            else:
                job_skills = []
            
            experiences.append({
                'Job Title': job_title,
                'Company': company,
                'Relevance Score': score,
                'Start Date': start_date,
                'End Date': end_date,
                'Responsibilities': responsibilities,  # Add responsibilities
                'Job Skills': job_skills,  # Add job-specific skills
                'Candidate ID': candidate_id  # Add candidate identifier
            })
    return experiences

def parse_date(date_str):
    """
    Parses a date string into a datetime object.
    
    Args:
        date_str (str): Date string (e.g., 'Jan 2020', 'Present').
    
    Returns:
        datetime: Parsed datetime object. If 'Present', returns current date.
    """
    if date_str.lower() == 'present':
        return datetime.datetime.now()
    try:
        return datetime.datetime.strptime(date_str, '%b %Y')
    except ValueError:
        try:
            return datetime.datetime.strptime(date_str, '%B %Y')
        except ValueError:
            st.warning(f"Unrecognized date format: {date_str}")
            return None

def map_roles_to_years_max(experiences):
    """
    Maps each role's relevance score to the years it spans, taking the maximum score per candidate per year.
    
    Args:
        experiences (list of dicts): Parsed job experiences.
    
    Returns:
        pd.DataFrame: DataFrame with Year, Candidate ID, Relevance Score.
    """
    records = []
    for exp in experiences:
        start_date = exp['Start Date']
        end_date = exp['End Date']
        candidate_id = exp['Candidate ID']
        relevance_score = exp['Relevance Score']
        if not start_date or not end_date:
            continue  # Skip if dates are not properly parsed
        for year in range(start_date.year, end_date.year + 1):
            records.append({
                'Year': year,
                'Candidate ID': candidate_id,
                'Relevance Score': relevance_score
            })
    df = pd.DataFrame(records)
    # Aggregate by Candidate ID and Year, taking the maximum relevance score
    df_max = df.groupby(['Candidate ID', 'Year'], as_index=False)['Relevance Score'].max()
    return df_max



# Set the page layout to wide
st.set_page_config(layout="wide")

# Streamlit app
st.title("Automated Screening Tool For Recruitment Operations (ASTRO)")

# Check if the password is correct.
if not check_password():
    st.stop()


# Creating the instructions with explicit "Step 1", "Step 2" labels
instructions = """
<div style="background-color: #f0f8ff; padding: 10px; border-radius: 5px;">
    <h4><strong>Instructions:</strong></h4>
    <p><strong>Step 1:  Select Resume Extraction Mode: </strong> 
        <ul>
            <li>For automated resume extraction, enter 'Data Scientist JD01' or 'Marketing Executive JD02' and select September as the date range for the 5 preloaded resumes and job description. Click 'Extract Resume' when done.</li>
            <li>For manual upload, enter job description and upload the resume (uploading function does not work on GSIB laptop. Click 'Extract Resume' when done.</li>
        </ul>
    </p>
    <p><strong>Step 2:  Adjust Skill Weights. </strong> Set the skill weights according to your preference and click 'Confirm Skill Weights'.</p>
    <p><strong>Step 3:  Filter Results. </strong> Choose the top N results and filter by education level. Then, click 'Show Results' to display summaries.</p>
    <p><strong>Step 4:  Re-adjusting Weights and Filters. </strong> 
        <ul>
            <li>To adjust the skill weights, repeat Step 2 and Step 3.</li>
            <li>To change the filters, repeat Step 3.</li>
        </ul>
    </p>
</div>
"""

# Display the instructions in Streamlit using markdown with embedded HTML
st.markdown(instructions, unsafe_allow_html=True)


# Initialize session state variables
if 'matched_resumes' not in st.session_state:
    st.session_state.matched_resumes = False
if 'results' not in st.session_state:
    st.session_state.results = None

# Sample resumes with submission dates (simulate automated extraction)
sample_resumes = [
    {'filename': 'resume1_kenny sim.docx', 'submission_date': datetime.date(2024, 9, 1)},
    {'filename': 'resume2_crystal.docx', 'submission_date': datetime.date(2024, 9, 5)},
    {'filename': 'resume3_mark.docx', 'submission_date': datetime.date(2024, 9, 10)},
    {'filename': 'resume4_david.docx', 'submission_date': datetime.date(2024, 9, 15)},
    {'filename': 'resume5_linus_ds.docx', 'submission_date': datetime.date(2024, 9, 30)}
]

# Create tabs for resume source selection
tabs = st.tabs(["Automated Resume Extraction", "Manual Upload"])
# -------------------- Automated Resume Extraction Tab --------------------
with tabs[0]:
    st.markdown("### Automated Resume Extraction")
    with st.form("automated_resume_extraction"):
        #st.markdown("### Automated Resume Extraction Inputs")
        
        # Input for job title/job code
        job_title = st.text_input("Enter Job Title/Code", value='e.g. Data Scientist JD01')

        # Date range picker for resume submission date
        date_range = st.date_input("Select Resume Submission Date Range", value=(datetime.date(2024, 9, 1), datetime.date(2024, 9, 30)))
        if isinstance(date_range, tuple):
            start_date, end_date = date_range
        else:
            start_date = end_date = date_range

        # Submit button
        submitted_automated = st.form_submit_button("Extract Resume")

    # -------------------- Handle "Extract Resume" Submission --------------------
    # Handle "Extract Resume" submission from Automated Resume Extraction tab
    if submitted_automated:
        # Reset previous results
        st.session_state.pop('df_results', None)
        st.session_state.pop('top_n_filtered', None)
        st.session_state.pop('original_skill_weights', None)
        st.session_state.pop('adjusted_skill_weights', None)
        st.session_state.pop('skill_embeddings', None)
        st.session_state.pop('job_desc_embedding', None)

        # Automated Resume Extraction logic
        if job_title:
            # Attempt to read job description from file
            job_description_file = os.path.join('job_descriptions', f"{job_title}.txt")
            if os.path.exists(job_description_file):
                with open(job_description_file, 'r', encoding='utf-8') as f:
                    job_description = f.read()
                # Proceed to filter resumes based on submission date
                filtered_resumes_info = [res for res in sample_resumes if start_date <= res['submission_date'] <= end_date]
                if not filtered_resumes_info:
                    st.warning("No resumes found for the selected date range.")
                else:
                    # Extract and process resumes
                    filenames = [res['filename'] for res in filtered_resumes_info]
                    resume_texts = []
                    for res_info in filtered_resumes_info:
                        filename = res_info['filename']
                        file_path = os.path.join('resumes', filename)
                        try:
                            with open(file_path, 'rb') as f:
                                file_content = extract_text(f)
                                resume_texts.append(file_content)
                        except Exception as e:
                            st.error(f"Error reading file {filename}: {e}")
                    # Process resumes and store results
                    with st.spinner('Extracting resumes...'):
                        results = process_resumes_initial(job_description, resume_texts, filenames)
                    st.success("Resumes extracted successfully!")

                    # **Store 'resume_texts' and 'filenames' in session state before setting 'matched_resumes'**
                    st.session_state.resume_texts = resume_texts
                    st.session_state.filenames = filenames

                    st.session_state.matched_resumes = True
    #                with st.spinner('Processing resumes...'):
    #                    st.session_state.results = calculate_similarity_scores(
    #                        resume_texts,
    #                        filenames,
    #                        st.session_state.job_desc_embedding,
    #                        st.session_state.skill_embeddings,
    #                        st.session_state.adjusted_skill_weights,
    #                        st.session_state.embeddings_model,
    #                        st.session_state.qa_chain  # Pass the chain object here
    #                    )
    #                    st.session_state.job_description = job_description
    #                    st.session_state.resume_texts = resume_texts
    #                    st.session_state.filenames = filenames
    #                st.success("Resumes processed successfully!")
            else:
                st.error(f"Job description file for '{job_title}' not found.")
        else:
            st.error("Please enter a job title/code.")

# -------------------- Manual Upload Tab --------------------
with tabs[1]:
    st.markdown("### Manual Upload")
    with st.form("manual_upload"):
        #st.markdown("### Manual Upload Inputs")
        
        # Input for job description
        job_description = st.text_area("Enter Job Description")

        # Upload resumes
        uploaded_files = st.file_uploader("Upload Resumes", accept_multiple_files=True, type=["pdf", "docx", "txt"])

        # Submit button
        submitted_manual = st.form_submit_button("Extract Resume")

    # -------------------- Handle "Extract Resume" Submission from Manual Upload tab --------------------
    if submitted_manual:
        # Reset previous results
        st.session_state.pop('df_results', None)
        st.session_state.pop('top_n_filtered', None)
        st.session_state.pop('original_skill_weights', None)
        st.session_state.pop('adjusted_skill_weights', None)
        st.session_state.pop('skill_embeddings', None)
        st.session_state.pop('job_desc_embedding', None)

        # Manual Upload logic
        if job_description and uploaded_files:
            # Extract text from uploaded files
            resume_texts = []
            filenames = []
            for uploaded_file in uploaded_files:
                try:
                    file_content = extract_text(uploaded_file)
                    resume_texts.append(file_content)
                    filenames.append(uploaded_file.name)
                except Exception as e:
                    st.error(f"Error processing file {uploaded_file.name}: {e}")
            # Process resumes and store results
            with st.spinner('Extracting resumes...'):
                job_skills = process_resumes_initial(job_description, resume_texts, filenames)
            st.success("Resumes uploaded and extracted successfully!")

            # **Store 'resume_texts' and 'filenames' in session state before setting 'matched_resumes'**
            st.session_state.resume_texts = resume_texts
            st.session_state.filenames = filenames

            st.session_state.matched_resumes = True
    #        with st.spinner('Processing resumes...'):
    #            st.session_state.results = calculate_similarity_scores(
    #                resume_texts,
    #                filenames,
    #                st.session_state.job_desc_embedding,
    #                st.session_state.skill_embeddings,
    #                st.session_state.adjusted_skill_weights,
    #                st.session_state.embeddings_model,
    #                st.session_state.qa_chain  # Pass the chain object here
    #            )
    #            st.session_state.job_description = job_description
    #            st.session_state.resume_texts = resume_texts
    #            st.session_state.filenames = filenames
    #        st.success("Resumes processed successfully!")
        else:
            st.error("Please enter a job description and upload at least one resume.")




# -------------------- Display and Adjust Skill Weights --------------------
# Create two columns: one for numbering and one for the expander
col1, col2 = st.columns([6, 4])  # Adjust the ratio as needed for layout

# Display the counter in the first column
with col1:
    if st.session_state.matched_resumes:
        st.subheader("Adjust Skill Weights")
        st.markdown("The skills and their respective weights listed below are generated by the AI based on the provided job description.")
        st.markdown("Please adjust the weights as needed and click 'Confirm Skill Weights'.")
        job_skills = list(st.session_state.skill_embeddings.keys())
        for skill in job_skills:
            current_weight = st.session_state.adjusted_skill_weights.get(skill.lower(), 1)
            adjusted_weight = st.slider(
                label=f"{skill}",
                min_value=1,
                max_value=10,
                value=current_weight,
                key=f"weight_{skill.lower()}"
            )
            st.session_state.adjusted_skill_weights[skill.lower()] = adjusted_weight

        # **Introduce the "Confirm Skill Weights" button**
        if st.button("Confirm Skill Weights"):
            with st.spinner('Calculating similarity scores...'):
                adjusted_results = calculate_similarity_scores(
                    st.session_state.resume_texts,
                    st.session_state.filenames,
                    st.session_state.job_desc_embedding,
                    st.session_state.skill_embeddings,
                    st.session_state.adjusted_skill_weights,
                    st.session_state.embeddings_model,
                    st.session_state.qa_chain  # Pass the chain object here
                )
                st.session_state.results = adjusted_results  # Store the results

            st.success("Similarity scores calculated based on adjusted weights!")

# -------------------- Filters Section --------------------
st.markdown("### Filters")
st.markdown("Adjust the filters and click 'Show Result' to view the summaries")

with st.form("Filters"):
    # Input for number of top resumes to display
    top_n = st.number_input("Enter the number of top resumes to display", min_value=1, value=5, step=1)

    # Dropdown for minimum education level
    education_levels = ['None', 'O level or equivalent', 'A level or equivalent', 'Diploma', 'Degree', 'Graduate Degree']
    min_education = st.selectbox("Select minimum education level required:", education_levels, index=0)

    # Submit button for filters
    show_results = st.form_submit_button("Show Results")






# -------------------- Display and Adjust Skill Weights --------------------

# Create two columns: one for numbering and one for the expander
#col1, col2 = st.columns([6, 4])  # Adjust the ratio as needed for layout

# Display the counter in the first column
#with col1:
#    if st.session_state.matched_resumes:
#        job_skills = list(st.session_state.skill_embeddings.keys())
#        st.subheader("Adjust Skill Weights")
#        for skill in job_skills:
#            current_weight = st.session_state.adjusted_skill_weights.get(skill.lower(), 1)
#            adjusted_weight = st.slider(
#                label=f"{skill}",
#                min_value=1,
#                max_value=10,
#                value=current_weight,
#                key=f"weight_{skill.lower()}"
#            )
#            st.session_state.adjusted_skill_weights[skill.lower()] = adjusted_weight

# -------------------- Handle "Show Results" Submission --------------------
if show_results:
    if st.session_state.matched_resumes and st.session_state.results is not None:
#display skill and weight
#        job_skills = list(st.session_state.skill_embeddings.keys())
#        st.subheader("Adjust Skill Weights")
#        for skill in job_skills:
#            current_weight = st.session_state.adjusted_skill_weights.get(skill.lower(), 1)
#            adjusted_weight = st.slider(
#                label=f"{skill}",
#                min_value=1,
#                max_value=5,
#                value=current_weight,
#                key=f"weight_{skill.lower()}"
#            )
#            st.session_state.adjusted_skill_weights[skill.lower()] = adjusted_weight


        results = st.session_state.results

        # Map education levels to numeric values
        education_level_mapping = {
            'O level or equivalent': 1,
            'A level or equivalent': 2,
            'Diploma': 3,
            'Degree': 4,
            'Graduate Degree': 5,
            'Other': 0
        }

        # Map user's selected minimum education level
        selected_min_education_level = education_level_mapping.get(min_education, 0)

        # Filter resumes based on minimum education level
        filtered_results = [result for result in results if result['Candidate Education Level'] >= selected_min_education_level]

        # Check if any resumes meet the criteria
        if not filtered_results:
            st.warning("No resumes meet the minimum education level requirement.")
        else:
            # Sort filtered results by similarity score in descending order
            filtered_results = sorted(filtered_results, key=lambda x: x["Similarity Score"], reverse=True)

            # Convert filtered results to DataFrame
            df_results = pd.DataFrame(filtered_results)

            # Add a column to number each row starting from 1
            df_results.insert(0, 'No.', pd.Series(range(1, len(df_results) + 1)))

            # For columns that are lists (Education, Highlights), join them into strings
            df_results['Education'] = df_results['Education'].apply(lambda x: '; '.join(x) if isinstance(x, list) else x)
            df_results['Highlights'] = df_results['Highlights'].apply(lambda x: '; '.join(x) if isinstance(x, list) else x)
            df_results['Skills'] = df_results['Skills'].apply(lambda x: '; '.join(x) if isinstance(x, list) else x)
            df_results['Relevant Years of Experience'] = df_results['Relevant Years of Experience'].str.replace('\n','<br>')


            # Reorder columns if necessary
            df_results = df_results[['No.', 'Filename', 'Name', 'Education', 'Mapped Education Level', 'Highlights','Skills', 'Relevant Years of Experience', 'Similarity Score', 'Resume Text']]

            # Apply the top_n filter to both Resume Rankings and Detailed Resume Summaries
            top_n_filtered = min(top_n, len(df_results))  # Ensure top_n does not exceed available resumes

            # Store results in session state
            st.session_state['df_results'] = df_results
            st.session_state['top_n_filtered'] = top_n_filtered
    else:
        if st.session_state.matched_resumes:
            st.info("Processing resumes. Please wait...")
        else:
            st.info("Please fill out the form and submit to process the resumes.")

# -------------------- Handle "Recalculate Similarity with Adjusted Weights" Button --------------------
#if st.session_state.matched_resumes:
#    job_skills = list(st.session_state.skill_embeddings.keys())
#    st.subheader("Adjust Skill Weights")
#    for skill in job_skills:
#        current_weight = st.session_state.adjusted_skill_weights.get(skill.lower(), 1)
#        adjusted_weight = st.slider(
#            label=f"{skill}",
#            min_value=1,
#            max_value=5,
#            value=current_weight,
#            key=f"weight_{skill.lower()}"
#        )
#        st.session_state.adjusted_skill_weights[skill.lower()] = adjusted_weight

# -------------------- Recalculate Similarity with Adjusted Weights Button --------------------
#if st.session_state.matched_resumes and st.session_state.results is not None:
#    if st.button("Recalculate Similarity with Adjusted Weights"):
#        with st.spinner('Recalculating similarity scores...'):
#            # Generate embeddings for job skills ####MOVED
#            skill_embeddings = {skill: embeddings_model.embed_query(skill) for skill in job_skills}
#            st.session_state.skill_embeddings = skill_embeddings
#
#            # Recalculate similarity using adjusted weights
#            adjusted_results = calculate_similarity_scores(
#                st.session_state.resume_texts,
#                st.session_state.filenames,
#                st.session_state.job_desc_embedding,
#                st.session_state.skill_embeddings,
#                st.session_state.adjusted_skill_weights,
#                st.session_state.embeddings_model,
#                st.session_state.qa_chain  # Pass the chain object here
#            )
#            st.session_state.results = adjusted_results  # Correctly indented
#
#            # Map education levels to numeric values
#            education_level_mapping = {
#                'O level or equivalent': 1,
#                'A level or equivalent': 2,
#                'Diploma': 3,
#                'Degree': 4,
#                'Graduate Degree': 5,
#                'Other': 0
#            }
#
#            # Map user's selected minimum education level
#            selected_min_education_level = education_level_mapping.get(min_education, 0)
#
#            # Filter resumes based on minimum education level
#            filtered_results = [result for result in adjusted_results if result['Candidate Education Level'] >= selected_min_education_level]
#
#            # Check if any resumes meet the criteria
#            if not filtered_results:
#                st.warning("No resumes meet the minimum education level requirement.")
#            else:
#                # Sort filtered results by similarity score in descending order
#                filtered_results = sorted(filtered_results, key=lambda x: x["Similarity Score"], reverse=True)
#
#                # Convert filtered results to DataFrame
#                df_results = pd.DataFrame(filtered_results)
#
#                # Add a column to number each row starting from 1
#                df_results.insert(0, 'No.', pd.Series(range(1, len(df_results) + 1)))
#
#                # For columns that are lists (Education, Highlights), join them into strings
#                df_results['Education'] = df_results['Education'].apply(lambda x: '; '.join(x) if isinstance(x, list) else x)
#                df_results['Highlights'] = df_results['Highlights'].apply(lambda x: '; '.join(x) if isinstance(x, list) else x)
#                df_results['Skills'] = df_results['Skills'].apply(lambda x: '; '.join(x) if isinstance(x, list) else x)
#                df_results['Relevant Years of Experience'] = df_results['Relevant Years of Experience'].str.replace('\n','<br>')
#
#                # Reorder columns if necessary
#                df_results = df_results[['No.', 'Filename', 'Name', 'Education', 'Mapped Education Level', 'Highlights','Skills', 'Relevant Years of Experience', 'Similarity Score', 'Resume Text']]
#
#               # Apply the top_n filter to both Resume Rankings and Detailed Resume Summaries
#                top_n_filtered = min(top_n, len(df_results))  # Ensure top_n does not exceed available resumes
#
#                # Store results in session state
#                st.session_state['df_results'] = df_results
#                st.session_state['top_n_filtered'] = top_n_filtered
#        st.success("Similarity scores recalculated based on adjusted weights!")
#else:
#    if st.session_state.matched_resumes:
#        st.info("Processing resumes. Please wait...")
#    else:
#        st.info("Please fill out the form and submit to process the resumes.")



# -------------------- Display Results and Handle Email Sending --------------------
# Check if df_results is available in session state
#if st.session_state.df_results is not None:
if 'df_results' in st.session_state:
    df_results = st.session_state['df_results']
    top_n_filtered = st.session_state['top_n_filtered']

    # Create a brief ranking above the table
    st.subheader("Resume Rankings")
    for idx, row in df_results.iterrows():
        if idx >= top_n_filtered:
            break

        # Use the 'No.' column for the counter/numbering
        counter = row['No.']  # Refers to the numbering column

        # Create two columns: one for numbering and one for the expander
        col1, col2 = st.columns([0.5, 9.5])  # Adjust the ratio as needed for layout

        # Display the counter in the first column
        with col1:
            st.write(f"{counter}")

        # Display the expander in the second column
        with col2:
            with st.expander(f"{row['Filename']} - Similarity Score: {row['Similarity Score']:.2f}"):
                st.write(row['Resume Text'])

    # Display the DataFrame without modifying the 'Filename' column
    st.subheader("Detailed Resume Summaries")
    # Exclude 'Resume Text' from the displayed DataFrame and apply top_n filter
    df_display = df_results.drop(columns=['Resume Text']).head(top_n_filtered)

    st.write(df_display.to_html(escape=False, index=False), unsafe_allow_html=True)

    # Input field for hiring manager's email
    recipient_email = st.text_input("Enter hiring manager's email:")

    # Button to send the summary table
    if st.button("Forward Summary to Hiring Manager"):
        if recipient_email:
            # Prepare DataFrame for CSV
            # Replace '<br>' with newline characters '\n' if necessary
            df_results['Relevant Years of Experience'] = df_results['Relevant Years of Experience'].str.replace('<br>', '\n')

            df_csv = df_results.drop(columns=['Resume Text'])
            csv_data = df_csv.to_csv(index=False)

            # Prepare the email content
            subject = "Resume Summaries"
            message = "Dear Hiring Manager,\n\nPlease find attached the summaries of the top resumes.\n\nBest regards,\nYour Name"

            # Send the email with attachment
            success = send_email(
                sender_email, sender_password, recipient_email,
                subject, message,
                attachment=csv_data.encode('utf-8-sig'),
                attachment_filename='resume_summaries.csv'
            )
            if success:
                st.success(f"Email sent to {recipient_email}")
            else:
                st.error(f"Failed to send email to {recipient_email}")
        else:
            st.error("Please enter a recipient email address.")

#    st.info("Please fill out the form and submit to process the resumes.")

# Assuming 'results' is stored in session_state after processing
    st.subheader("Relevancy of Past Experience")
#if st.button("Visualise Relevancy of Past Experience"):
    #if 'results' in st.session_state:
    #    results = st.session_state.results
    #    df_results = st.session_state['df_results']
    #    top_n_filtered = st.session_state['top_n_filtered']
    all_experiences = []
    
    # Iterate through each resume's results
    for idx, row in df_results.head(top_n_filtered).iterrows():
        # Replace '<br>' with '\n' to ensure proper parsing
        experience_str = row.get('Relevant Years of Experience', '').replace('<br>', '\n')
        candidate_name = row.get('Name', f"Candidate_{idx+1}")
        candidate_filename = row.get('Filename', f"Candidate_{idx+1}")
        candidate_id = candidate_name if candidate_name != 'N/A' else candidate_filename
        parsed_experiences = parse_relevant_experience(experience_str, candidate_id)
        all_experiences.extend(parsed_experiences)

    if all_experiences:
        # Create a DataFrame from all experiences with relevance scores
        df_experiences = map_roles_to_years_max(all_experiences)
        
        # Drop entries with missing years
        df_experiences = df_experiences.dropna(subset=['Year'])
        
        # Sort the DataFrame by Candidate ID and Year to ensure chronological order
        df_experiences = df_experiences.sort_values(by=['Candidate ID', 'Year'])

        # Since we're only displaying the top N candidates, we don't need a selection filter
        df_filtered = df_experiences
        
        # Create the line chart
        fig_line = px.line(
            df_filtered,
            x='Year',
            y='Relevance Score',
            color='Candidate ID',
            markers=True,
            title='Candidate Relevance Score Progression Over Years',
            labels={
                'Year': 'Year',
                'Relevance Score': 'Relevance Score',
                'Candidate ID': 'Candidate'
            },
            hover_data={'Candidate ID': True, 'Relevance Score': True, 'Year': True}
        )

        # Update hover mode to show data only for the hovered line
        fig_line.update_layout(hovermode='closest')

        # Customize the hover template to display desired information
        fig_line.update_traces(
            hovertemplate='<b>Candidate:</b> %{fullData.name}<br><b>Year:</b> %{x}<br><b>Relevance Score:</b> %{y}<extra></extra>'
        )

        
        # Display the line chart in Streamlit
        st.plotly_chart(fig_line, use_container_width=True)
    else:
        st.info("No relevant experiences found to plot.")


st.warning("""
Disclaimer

IMPORTANT NOTICE: This web application is developed as a proof-of-concept prototype. 
The information provided here is NOT intended for actual usage and should not be relied 
upon for making any decisions, especially those related to financial, legal, or healthcare matters.

Furthermore, please be aware that the LLM may generate inaccurate or incorrect information. 
You assume full responsibility for how you use any generated output.

Always consult with qualified professionals for accurate and personalized advice.
""")
