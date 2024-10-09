__import__('pysqlite3')
import sys
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
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3') # in order for chromadb to run need to set this

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

    Provide the output in JSON format with each skill and its corresponding weight. Return the following details in JSON format only. Do not include any additional text or explanations—just the JSON data.Example:

    {{
        "Python": 5,
        "Machine Learning": 5,
        "Data Analysis": 4,
        "SQL": 3,
        "Data Visualization": 2
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

def normalize_skills(skills):
    """
    Normalizes the extracted skills to handle synonyms and variations.

    Args:
        skills (list): A list of extracted skills.

    Returns:
        list: A list of normalized skills.
    """
    synonym_mapping = {
        'ml': 'machine learning',
        'data viz': 'data visualization',
        'sql server': 'sql',
        'data science': 'data science',
        'python programming': 'python',
        'statistics analysis': 'statistics'
        # Add more synonyms as needed
    }
    normalized = []
    for skill in skills:
        normalized_skill = synonym_mapping.get(skill.lower(), skill.lower())
        normalized.append(normalized_skill)
    return normalized



# Set the page layout to wide
st.set_page_config(layout="wide")

# Streamlit app
st.title("Resume Matcher with Advanced Retrieval")

# Check if the password is correct.
if not check_password():
    st.stop()

# Initialize session state variables
if 'matched_resumes' not in st.session_state:
    st.session_state.matched_resumes = False
if 'results' not in st.session_state:
    st.session_state.results = None

# Sample resumes with submission dates (simulate automated extraction)
sample_resumes = [
    {'filename': 'resume1_KENNY SIM.docx', 'submission_date': datetime.date(2023, 9, 1)},
    {'filename': 'resume2_crystal.docx', 'submission_date': datetime.date(2023, 9, 5)},
    {'filename': 'resume3_mark.docx', 'submission_date': datetime.date(2023, 9, 10)},
    {'filename': 'resume4_david.docx', 'submission_date': datetime.date(2023, 9, 15)},
    {'filename': 'resume5_linus_ds.docx', 'submission_date': datetime.date(2023, 10, 15)}
]

# Create tabs for resume source selection
tabs = st.tabs(["Automated Resume Extraction", "Manual Upload"])

# -------------------- Filters Section --------------------
st.markdown("### Filters")

with st.form("Filters"):
    # Input for number of top resumes to display
    top_n = st.number_input("Enter the number of top resumes to display", min_value=1, value=5, step=1)

    # Dropdown for minimum education level
    education_levels = ['None', 'O level or equivalent', 'A level or equivalent', 'Diploma', 'Degree', 'Graduate Degree']
    min_education = st.selectbox("Select minimum education level required:", education_levels, index=0)

    # Submit button for filters
    show_results = st.form_submit_button("Show Results")

# -------------------- Automated Resume Extraction Tab --------------------
with tabs[0]:
    st.markdown("### Automated Resume Extraction")
    with st.form("automated_resume_extraction"):
        #st.markdown("### Automated Resume Extraction Inputs")
        
        # Input for job title/job code
        job_title = st.text_input("Enter Job Title/Code", value='e.g. Data Scientist JD01')

        # Date range picker for resume submission date
        date_range = st.date_input("Select Resume Submission Date Range", value=(datetime.date(2023, 9, 1), datetime.date(2023, 9, 30)))
        if isinstance(date_range, tuple):
            start_date, end_date = date_range
        else:
            start_date = end_date = date_range

        # Submit button
        submitted_automated = st.form_submit_button("Extract Resume")

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

# Function to process resumes (unchanged)
def process_resumes(job_description, resume_texts, filenames):
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
    embeddings_model2 = OpenAIEmbeddings(model='text-embedding-3-small')  # Use the same model
    job_desc_embedding = embeddings_model2.embed_query(job_description)

    # **New: Extract and Rank Skills from Job Description**
    job_skills, skill_weights = extract_and_rank_skills(job_description, llm)  # Function to extract and rank skills

    # **New: Generate Embeddings for Job Skills**
    skill_embeddings = {skill: embeddings_model.embed_query(skill) for skill in job_skills}

    # Initialize a list to store skill similarity scores
    skill_similarity_scores = []

    # Generate summaries and calculate similarity scores
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
                {{"Job title": "Software Engineer", "Company name": "ABC Corp", "Start date": "Jan 2020", "End date": "Present"}},
                {{"Job title": "Junior Developer", "Company name": "XYZ Ltd", "Start date": "May 2017", "End date": "Dec 2019"}}
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

##############################################REPLACED###########################################################
        # **New: Calculate Weighted Skill Similarity Score**
        #matched_skills = set(job_skills).intersection(set(skills))
        #skill_score = 0
        #for skill in matched_skills:
        #    weight = skill_weights.get(skill.lower(), 1)  # Default weight is 1 if not specified
        #    skill_score += weight

        # Normalize skill_score by the maximum possible score
        #max_skill_score = sum(skill_weights.values())
        #normalized_skill_score = skill_score / max_skill_score  # Value between 0 and 1
##############################################################################################################

# **Modified: Calculate Weighted Skill Similarity Score Using Embeddings**
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
            weight = skill_weights.get(skill.lower(), 1)  # Default weight is 1 if not specified
            skill_score += weight

        # Normalize skill_score by the maximum possible score
        max_skill_score = sum(skill_weights.values())
        normalized_skill_score = skill_score / max_skill_score  # Value between 0 and 1
############################################################################################################
        # **New: Combined Similarity Score**
        # Assign weights to overall similarity and skill similarity
        overall_weight = 0.5  # Weight for textual similarity
        skill_weight = 0.5    # Weight for skill similarity
        combined_similarity = (overall_weight * similarity_score) + (skill_weight * normalized_skill_score)


        # Map candidate's education level to numeric value
        candidate_education_level = education_level_mapping.get(mapped_education_level, 0)

        # Proceed with processing job_experiences as before
        formatted_roles = []
        for job in job_experiences:
            job_title_role = job.get('Job title', 'N/A')
            company = job.get('Company name', 'N/A')
            start_date_role = job.get('Start date', 'N/A')
            end_date_role = job.get('End date', 'N/A')
            role_text = f"{job_title_role} in {company} ({start_date_role} - {end_date_role})"

            # Calculate relevance score for this specific job role
            role_embedding = embeddings_model.embed_query(role_text)
            relevance_score = cosine_similarity(
                [job_desc_embedding], [role_embedding]
            )[0][0] * 10  # Scale the relevance score (0-10 scale)

            # Format the role with relevance score
            formatted_roles.append(f"{job_title_role} in {company} [Relevant score: {relevance_score:.1f}], {start_date_role} - {end_date_role}")

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
# -------------------- Handle "Extract Resume" Submission --------------------
# Handle "Extract Resume" submission from Automated Resume Extraction tab
if submitted_automated:
    # Reset previous results
    st.session_state.pop('df_results', None)
    st.session_state.pop('top_n_filtered', None)

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
                with st.spinner('Processing resumes...'):
                    results = process_resumes(job_description, resume_texts, filenames)
                st.success("Resumes extracted and processed successfully!")
                st.session_state.matched_resumes = True
                st.session_state.results = results
        else:
            st.error(f"Job description file for '{job_title}' not found.")
    else:
        st.error("Please enter a job title/code.")

# Handle "Extract Resume" submission from Manual Upload tab
if submitted_manual:
    # Reset previous results
    st.session_state.pop('df_results', None)
    st.session_state.pop('top_n_filtered', None)

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
        with st.spinner('Processing resumes...'):
            results = process_resumes(job_description, resume_texts, filenames)
        st.success("Resumes uploaded and processed successfully!")
        st.session_state.matched_resumes = True
        st.session_state.results = results
    else:
        st.error("Please enter a job description and upload at least one resume.")

# -------------------- Handle "Show Results" Submission --------------------
if show_results:
    if st.session_state.matched_resumes and st.session_state.results is not None:
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

# -------------------- Display Results and Handle Email Sending --------------------
# Check if df_results is available in session state
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
else:
    st.info("Please fill out the form and submit to process the resumes.")
