import matplotlib.pyplot as plt
import streamlit as st
from PIL import Image
import pandas as pd
from eda import ExploratoryDataAnalysis

# --- PATH_SETTINGS ---
PATH_CSS = f"{st.secrets.PATH_CONFIGURATION.path_css}/main.css"
PATH_IMAGES = st.secrets.PATH_CONFIGURATION.path_images
PATH_DATASET = st.secrets.PATH_CONFIGURATION.path_dataset

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

@st.cache_data
def load_pict(filename):
    img = Image.open(filename)
    return img

profile_pic = load_pict(f"{PATH_IMAGES}/profile-pic.png")

# --- SIDE BAR ---
with st.sidebar:
    scol1, scol2 = st.columns([1, 2])
    with scol1:
        st.image(profile_pic, width=80)
    with scol2:
        st.write("## Muhammad Afryan Saputra's")
    st.write("---")
    
# --- FUNCTION ---
@st.cache_data(show_spinner=False)
def load_pickle(filename):
    df = pd.read_pickle(filename)
    return df

@st.cache_data(show_spinner=False)
def transform_df(data:pd.DataFrame, arr :list):
    df = data.filter(arr)
    return df

@st.cache_data(show_spinner=False)
def get_fig(_func:ExploratoryDataAnalysis):
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(15, 5), layout="constrained")
    _func.Freq_char(axes[0, 0])
    _func.Freq_word(axes[0, 1])
    _func.Freq_mean(axes[1, 0])
    _func.Freq_unique(axes[1, 1])
    return fig
    

@st.cache_data(show_spinner=False)
def load_eda()->ExploratoryDataAnalysis:
    data = load_pickle(f'{PATH_DATASET}sentiment.pkl')
    data = transform_df(data, ['cleaned', 'sentiment'])
    eda = ExploratoryDataAnalysis(data)
    return eda



# --- TABS ---
tab1, tab2 = st.tabs(["Project 1", "Project 2"])
with tab1:
    with st.container():
        st.write("#")
        st.write("# **SentimentCine: Sentiment Analysis on Movie Reviews using Natural Language Processing (NLP) Technology**")
        st.write("""Author : Muhammad Afryan Saputra""")
        st.write("---")
        st.write("""The goal of the project is to develop SentimentCine, a system that will perform sentimental analysis on film reviews using natural language processing (NLP) technology and machine learning. 
                 It will help producers, directors, and screenwriters understand how the audience responds to their work and provide stakeholders in the film industry with insights that can be used for strategic decision-making.""")
        
    ## --- Dataset ---
    with st.container():
        df_tab1 = load_pickle(f'{PATH_DATASET}preprocessed.pkl') 
        df_tab1 = transform_df(df_tab1, ['judul', 'tweet', 'detokenize'])

        st.write("## Datasets")
        st.write(f"""The datasets used for the training and test data on this project are `{df_tab1.shape[0]: ,}` Indonesian language reviews with `{df_tab1['judul'].nunique():,}` different titles taken through the `Twitter` platform.""")
        st.dataframe(
            df_tab1,
            column_config={
                "judul" : "Tittle",
                "tweet" : "Reviews",
                "detokenize" : "Preprocessed"
            }, 
            use_container_width=True,
            hide_index=False)

    ## --- Preprocessing ---
    with st.container():
        st.write('## Preprocessing')
        st.write("""The techniques used in the preprocessing phase for training dataset above are:
- Removing numbers, hashtags, mention, and special characters
- Tokenization
- Normalization
- Stopwords Removal
- Stemming
- Lemmatization
""")

    ## --- Exploratory Data Analysis ---
    with st.container():
        eda = load_eda()
        st.write('## Exploratory Data Analysis (EDA)')

        eda_col1, eda_col2 = st.columns([2,1])
        with eda_col1:            
            fig, ax = plt.subplots(1, 1, figsize=(7.5, 5))
            eda.Class_plot(ax)
            st.pyplot(fig)
        with eda_col2:
            st.markdown("The datasets used as training data have a linear distribution of the sentiment values.")

        with st.container():
            st.pyplot(get_fig(eda))

    
with tab2:
    with st.container():
        st.write("#")
        st.write("# **Coming Soon...**")
