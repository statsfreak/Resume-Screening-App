# helper_functions/llm.py

import openai
import re
import json
import os

# Initialize OpenAI client
openai.api_key = os.getenv('OPENAI_API_KEY')

def extract_information_with_llm(qa_chain, resume_text, filename):
    """
    Extracts structured information from a resume using an LLM.

    Parameters:
        qa_chain: The QA chain object for LLM interaction.
        resume_text: The text content of the resume.
        filename: The name of the resume file.

    Returns:
        parsed_result: A dictionary containing extracted information.
        error_message: An error message if parsing fails.
        raw_output: The raw output from the LLM if parsing fails.
    """
    # Modify the prompt to request JSON-only output
    result = qa_chain.run(f"""
    For the following resume, extract and return the following details in JSON format only. Do not include any additional text or explanations—just the JSON data.

    Required details:
    - Name
    - Education (list all tertiary degrees and above obtained, including degree name and field of study. If no tertiary degree then state the highest education achieved)
    - Map the highest obtained education level to one of the following categories: "O level or equivalent", "A level or equivalent", "Diploma", "Degree", "Graduate Degree", "Other"
    - Provide the mapped education level as "Mapped Education Level"
    - Highlights of Career (2-3 standout sentences from the resume)

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
        return parsed_result, None, None
    except Exception as e:
        error_message = f"Failed to parse result for {filename}: {e}"
        raw_output = result
        return {}, error_message, raw_output
