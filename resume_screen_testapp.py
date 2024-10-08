
import streamlit as st
import base64
# ASTRO - automated system for talent recruitment optimisation
#automated screening tool for recruitment operations
#AURA - Automated Utility for Resume Assessment

def add_bg_from_url():
    st.markdown(
         f"""
         <style>
         .stApp {{
             #background-image: url("https://cdn.pixabay.com/photo/2016/10/05/03/36/blue-1716030_1280.jpg");
             background-image: linear-gradient(rgba(255, 255, 255, 0.5), rgba(255, 255, 255, 0.75)), url("https://cdn.pixabay.com/photo/2016/10/29/10/12/purple-1780371_1280.png");
             background-attachment: fixed;
             background-size: cover
         }}
         </style>
         """,
         unsafe_allow_html=True
     )

add_bg_from_url() 


# Upload resumes
uploaded_files = st.file_uploader("Upload Resumes", accept_multiple_files=True, type=["pdf", "docx", "txt"])

if uploaded_files:
    st.write("Files uploaded successfully.")
    for uploaded_file in uploaded_files:
        try:
            st.write("Filename: ", uploaded_file.name)
            st.write("File type: ", uploaded_file.type)
            st.write("File size: ", uploaded_file.size)
        except Exception as e:
            st.error(f"Error processing file {uploaded_file.name}: {e}")
