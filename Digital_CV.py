import streamlit as st
from PIL import Image

# --- PATH_SETTINGS ---
PATH_CSS = f"{st.secrets.PATH_CONFIGURATION.path_css}/main.css"
PATH_IMAGES = st.secrets.PATH_CONFIGURATION.path_images

INFO_GEN = st.secrets.general

# --GENERAL SETTINGS ---
SOCIAL_MEDIA = {
    "LinkedIn" : "https://www.linkedin.com/in/saputraafryan/",
    "GitHub" : "https://github.com/SaputraAfryan/",
    "Streamlit" : "https://digital-cv-afryan.streamlit.app/",
}

PROJECTS = {
    "Aspect-Based Sentiment Analysis Using Recurrent Neural Networks (RNN) on Social Media Twitter" : "https://ieeexplore.ieee.org/document/10276768",

}

SKILLS = {
    "Programming":["Python", 
                   "GoLang",
    ],
    "Data Processing/Wrangling":[
        "SQL", 
        "Pandas", 
        "NumPy"
    ],
    "Data Visualization":[
        "Matplotlib",
        "Seaborn",
        "Plotly"
    ],
    "Machine Learning": [
        "Scikit-Learn",
    ],
    "Deep Learning": [
        "TensorFlow",
        "PyTorch",
        "Keras",
    ],
    "Mobile Development": [
      "Flutter"  
    ],
    "Web Development":[
        "Laravel",
        "HTML",
        "CSS",
    ],
    "Model Deployment":[
        "Streamlit",
        "Heroku",
    ]
}


st.set_page_config(
    page_title="Digital CV | Afryan",
    layout="wide",
)

# --- LOAD RESOURCE ---
with open(PATH_CSS) as f:
    st.markdown("<style>{}</style>".format(f.read()), unsafe_allow_html=True)
profile_pic = Image.open(f"{PATH_IMAGES}/profile-pic.png")

arrow_icon = """
<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" fill="currentColor" class="bi bi-arrow-left-circle-fill" viewBox="0 0 16 16">
  <path d="M8 0a8 8 0 1 0 0 16A8 8 0 0 0 8 0m3.5 7.5a.5.5 0 0 1 0 1H5.707l2.147 2.146a.5.5 0 0 1-.708.708l-3-3a.5.5 0 0 1 0-.708l3-3a.5.5 0 1 1 .708.708L5.707 7.5z"></path>
</svg>
"""

# --- SIDE BAR ---
with st.sidebar:
    scol1, scol2 = st.columns([1, 2])
    with scol1:
        st.image(profile_pic, width=80)
    with scol2:
        st.write("## Muhammad Afryan Saputra's")
    st.write("---")
    st.markdown('''
                <ul style="list-style-type:none;">
                    <li>
                        <button class="sidebar-button">
                            <a href="#summary" target="_self" class="sidebar-link">
                                Personal Information</a>
                        </button>
                    </li>
                    <li>
                        <button class="sidebar-button">
                            <a href="#summary" target="_self" class="sidebar-link">
                                Summary
                            </a>
                        </button>
                    </li>
                    <li>
                        <button class="sidebar-button">
                            <a href="https://digital-cv-afryan.streamlit.app/~/+/#education" target="_self" class="sidebar-link">
                                Education
                            </a>
                        </button>
                    </li>
                    <li>
                        <button class="sidebar-button">
                            <a href="https://digital-cv-afryan.streamlit.app/~/+/#work-experience" target="_self" class="sidebar-link">
                                Work Experience
                            </a>
                        </button>
                    </li>
                    <li>
                        <button class="sidebar-button">
                            <a href="https://digital-cv-afryan.streamlit.app/~/+/#skills" target="_self" class="sidebar-link">
                                Skills
                            </a>
                        </button>
                    </li>
                </ul>
                '''
    , unsafe_allow_html=True)

# --- MAIN PAGE ---
col1, col2 = st.columns([1, 2])
with st.container():
    with col1:
        st.image(profile_pic, width=250)
    with col2:
        st.write("# Muhammad Afryan Saputra")
        st.write(f"**{INFO_GEN.address}**")
        st.write(f"🎂 : **{INFO_GEN.birth}**")
        st.write(f"📧 : **{INFO_GEN.email}**")
        st.write(f"📞 : **{INFO_GEN.phone}**")

# --- SOCIAL MEDIA ---
st.write("#")
cols = st.columns(len(SOCIAL_MEDIA))
for i, (platform, link) in enumerate(SOCIAL_MEDIA.items()):
    cols[i].markdown(f'''
        <a class="center" href="{link}"> {platform} <span style="color: rgba(0, 0, 0, 0); margin-right: 0px">a</span>{arrow_icon}</a>
    ''', unsafe_allow_html=True)

# --- SUMMARY ---
with st.container():
    st.write("---")
    st.markdown("<h1 id='summary'>Summary</h1>", unsafe_allow_html=True)
    st.write('''
    Bachelor's degree in Informatics Engineering at Telkom University with a Data Science concentration and `240+` hours of internship experience at UPTD PSDA WS Cisadea-Cibareno, `500+` hours working on projects in data science and data analysis throughout 2023.
''')

def col_text(a, b):
    col1, col2 = st.columns([4, 1])
    with col1:
        st.write(a)
    with col2:
        st.write(b)

def s_cols(x: dict):
    s_col1, s_col2 = st.columns([1, 4])
    with s_col1:
        for key in x.keys():
            st.write(f"{key}")
    with s_col2:
        for items in x.values():
            item = ', '.join([f"`{values}`" for values in items])
            st.write(item)


# --- EDUCATION ---
with st.container():
    st.write("#")
    st.write("# **Education**")
    col_text("**Bachelor of Informatics Engineering** (Informatics Engineering), *Telkom University*, Indonesia", "2019-2023")
    st.write('''
    - GPA: `3.04`
    - EPRT: `510`
    - Final-Project reseach entitled: `Aspect-Based Sentiment Analysis Using Recurrent Neural Networks (RNN) on Social Media Twitter`
    - Presenting Final-Project in `The 6th International Conference on Data Science and Its Applications 2023`
''')
    

# --- WORK EXPERIENCES ---
with st.container():
    st.write("#")
    st.write("# **Work Experience**")
    col_text("**Mobile Application Developer**, UPTD PSDA WS Cisadea-Cibareno, Indonesia | Internship", "07/2022 - 09/2022")
    st.write('''
- Utilized Python scripting to efficiently organize and process employee data, resulting in a 22% reduction in data processing time
- Designed and implemented an SQL database system to store employee information, improving data accessibility and retrieval by 43%.
- Created visually appealing Employee of the Month displays using Canva, contributing to a 17% increase in employee morale and engagement.
- Successfully coordinated projects involving Python, MySQL, Canva, and Flutter, demonstrating proficiency in integrating technologies for a seamless workflow.
''')
    col_text("**Assistant Lecturer**, Telkom University, Indonesia", "09/2020 - 03/2021")
    st.write('''
- Conducted regular assessments, resulting in a 35% decrease in students struggling with foundational programming principles.
- Achieved a 15% increase in the number of students completing advanced programming projects successfully.
- Implemented a training session support system resulting in a 30% decrease in lecturer workload.
''')

     
# --- SKILLS ---
    st.write("#")
    st.write("# **Skills**")
    s_cols(SKILLS)


