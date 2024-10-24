
import streamlit as st
from PIL import Image

st.set_page_config(
    page_title="Resume Matcher",
    page_icon=":page_with_curl:",
    layout="wide",
    initial_sidebar_state="expanded",
)


def add_bg_from_url():
    st.markdown(
         f"""
         <style>
         .stApp {{
             #background-image: url("https://cdn.pixabay.com/photo/2016/10/29/10/12/purple-1780371_1280.png");
             background-image: linear-gradient(rgba(0, 0, 0, 0.5), rgba(255, 255, 255, 0.75)), url("https://cdn.pixabay.com/photo/2016/10/29/10/12/purple-1780371_1280.png");
             background-attachment: fixed;
             background-size: cover
         }}
         /* Ensure the font color stays black */
         .stApp * {{
             color: black !important;
         }}
         </style>
         """,
         unsafe_allow_html=True
     )

add_bg_from_url() 


# Path to your header image
#header_image_path = "C:/Users/weichee/Documents/ai bootcamp/headerc.png"
header_image_path = "headerc.png"
header_image = Image.open(header_image_path)
#header_image_path2 = "C:/Users/weichee/Documents/ai bootcamp/screenshot2.png"
header_image_path2 = "screenshot2.png"
header_image2 = Image.open(header_image_path2)

# Display the header image
col1 = st.columns(1)[0]
with col1:
    st.image(header_image, caption="", use_column_width=True)
    
col1 = st.columns(1)[0]
with col1:
    st.image(header_image2, caption="", use_column_width=True)


#st.subheader("""
#Welcome to the Resume Matcher application. Use the navigation menu on the left to access different sections of the app.
#""")
