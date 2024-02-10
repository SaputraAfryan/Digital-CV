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
    

@st.cache_resource
def load_eda()->ExploratoryDataAnalysis:
    data = load_pickle(f'{PATH_DATASET}sentiment.pkl')
    data = transform_df(data, ['cleaned', 'sentiment'])
    eda = ExploratoryDataAnalysis(data)
    return eda

@st.cache_data(show_spinner=False)
def plot_ngrams(_func:ExploratoryDataAnalysis, cat, n):
    fig = plt.figure(figsize=(15,5))
    _func.Freq_ngrams(cat, n)
    return fig



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

        _, center, __ = st.columns([0.5,1,0.5])
        with center:            
            fig, ax = plt.subplots(1, 1, figsize=(7.5, 5))
            eda.Class_plot(ax)
            st.pyplot(fig)
            st.markdown("<p class='eda-text'>The datasets used as training data have a linear distribution of the sentiment values.</p>", unsafe_allow_html=True)

        with st.container():
            st.pyplot(get_fig(eda))
            st.write("""From the above exploration can be inferred that the data has it's peak frequency at:
- Number of Words : `11`
- Number of Characters: `70` 
- Number of Unique Characters: `22` 
- Average Word Length : `5`
""")
        
        with st.container():
            sbox, slide = st.columns([0.4, 0.6])
            with sbox:
                sent_val = st.select_slider("Select Sentiment Values", ["Positive", "Neutral", 'Negative'])
            with slide:
                N = st.slider("How Many N-Grams", 1, 5)
            st.pyplot(plot_ngrams(eda, sent_val, N))
            st.write("""Some titles like `Kimi no Nawa` become tokens that often appear on the whole sentiment value when n-grams = `3`. 
                     This is because the review on the dataset has three aspects of discussion, namely: `Plot`, `Actor`, and also `Director`. 
                     While this project is intended to do sentiment analysis at document level.""")

    
with tab2:
    with st.container():
        st.write("#")
        st.write("# **Coming Soon...**")
