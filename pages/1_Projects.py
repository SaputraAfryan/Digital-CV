import streamlit as st
from PIL import Image

# --- PATH_SETTINGS ---
PATH_CSS = f"{st.secrets.PATH_CONFIGURATION.path_css}/main.css"
PATH_IMAGES = st.secrets.PATH_CONFIGURATION.path_images

INFO_GEN = st.secrets.general

# --GENERAL SETTINGS ---
SOCIAL_MEDIA = {
    "LinkedIn" : "https://www.linkedin.com/in/saputraafryan/",
    "GitHub" : "https://github.com/SaputraAfryan/"
}

PROJECTS = {
    "Aspect-Based Sentiment Analysis Using Recurrent Neural Networks (RNN) on Social Media Twitter" : "https://ieeexplore.ieee.org/document/10276768",

}


st.set_page_config(
    page_title="Digital CV | Afryan",
    layout="wide",
)

# --- LOAD RESOURCE ---
with open(PATH_CSS) as f:
    st.markdown("<style>{}</style>".format(f.read()), unsafe_allow_html=True)
profile_pic = Image.open(f"{PATH_IMAGES}/profile-pic.png")

# --- SIDE BAR ---
with st.sidebar:
    scol1, scol2 = st.columns([1, 2])
    with scol1:
        st.image(profile_pic, width=80)
    with scol2:
        st.write("## Muhammad Afryan Saputra's")
    st.write("---")
    # st.markdown('''
    #             <ul style="list-style-type:none;">
    #                 <li>
    #                     <button class="sidebar-button">
    #                         <a href="/Projects#aspect-based-sentiment-analysis-using-recurrent-neural-networks-rnn-on-social-media-twitter" target="_self" class="sidebar-link">
    #                             Project 1</a>
    #                     </button>
    #                 </li>
    #                 <li>
    #                     <button class="sidebar-button">
    #                         <a href="/Projects#coming-soon" target="_self" class="sidebar-link">
    #                             Project 2</a>
    #                     </button>
    #                 </li>
    #             </ul>
    #             '''
    # , unsafe_allow_html=True)

# --- TABS ---
tab1, tab2 = st.tabs(["Project 1", "Project 2"])
with tab1:
    with st.container():
        st.write("#")
        st.write("# **Aspect-Based Sentiment Analysis Using Recurrent Neural Networks (RNN) on Social Media Twitter**")
        st.write("""Author : 
- Muhammad Afryan Saputra 
- Erwin Budi Setiawan""")
        
with tab2:
    with st.container():
        st.write("#")
        st.write("# **Coming Soon...**")
