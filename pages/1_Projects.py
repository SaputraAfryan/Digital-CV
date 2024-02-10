import joblib
from numpy import vectorize
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from PIL import Image
from eda import ExploratoryDataAnalysis
from preprocessing import TextPreparation

# --- PATH_SETTINGS ---
PATH_CSS = f"{st.secrets.PATH_CONFIGURATION.path_css}/main.css"
PATH_IMAGES = st.secrets.PATH_CONFIGURATION.path_images
PATH_DATASET = st.secrets.PATH_CONFIGURATION.path_dataset
PATH_MODEL = st.secrets.PATH_CONFIGURATION.path_model

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

@st.cache_resource
def load_model():
    model = joblib.load(f'{PATH_MODEL}best_svm_model.joblib')
    return model

@st.cache_resource
def load_vec():
    vectorizer = joblib.load(f'{PATH_MODEL}tfidf_vectorizer.joblib')
    return vectorizer

@st.cache_data(show_spinner=False)
def predict(str):
    vec = load_vec()
    model = load_model()
    tp = TextPreparation()

    text = tp.preprocess_text(str)
    test = vec.transform([text])
    return model.predict(test)[0]



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

        st.write('#')
        st.write("## Datasets")
        st.write(f"""The datasets used for the training and test data on this project are `{df_tab1.shape[0]: ,}` Indonesian language reviews with `{df_tab1['judul'].nunique():,}` different titles taken through the `Twitter` platform.""")
        dcol1, dcol2 = st.columns([0.65, 0.35])
        with dcol1:
            st.dataframe(
                df_tab1,
                column_config={
                    "judul" : "Tittle",
                    "tweet" : "Reviews",
                    "detokenize" : "Preprocessed"
                }, 
                use_container_width=True,
                hide_index=False)
        with dcol2:
            ## --- Preprocessing ---
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
        st.write('#')
        st.write('---')
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
        
    ## --- N-Grams Distribution ---
    with st.container():
        st.write('#')
        st.write('---')
        st.write('## N-Grams Distribution')
        sbox, slide = st.columns([0.4, 0.6])
        with sbox:
            sent_val = st.select_slider("Select Sentiment Values", ["Positive", "Neutral", 'Negative'])
        with slide:
            n_val = st.slider("How Many N-Grams", 1, 5)
        st.pyplot(plot_ngrams(eda, sent_val, n_val))
        st.write("""Some titles like `Kimi no Nawa` become tokens that often appear on the whole sentiment value when n-grams = `3`. 
                    This is because the review on the dataset has three aspects of discussion, namely: `Plot`, `Actor`, and also `Director`. 
                    While this project is intended to do sentiment analysis at document level.""")

    ## --- Models ---
        st.write('#')
        st.write('---')
        st.write('## **Models**')
        st.write("""Support Vector Machine (SVM) is used alongside with TF-IDF because SVM has proven to be effective in handling complex and non-linear data. 
                 This is beneficial for sentimental analysis processing because the data used is often complex and not necessarily linear. 
                 In addition, SVM can also handle high-dimensional data and separate classes well in the space.""")
        st.write("""TF-IDF gives weight to words based on the frequency in a document and how rarely the word appears throughout the dataset.
                 Words that appear frequently in the document but rarely appear generally will have a higher weight.""")
        st.write('#### Try This Out...')
        inp_col1, inp_col2 = st.columns([2,1])
        with inp_col1:
            def callbacks():
                predict(st.session_state.review)
            if 'review' not in st.session_state:
                st.session_state.review = 'bagus banget ceritanya'
            else:
                pc = "Example: bagus banget ceritanya"
                st.text_input("Reviews (Indonesian)", placeholder=pc, 
                          key='review', 
                          on_change=callbacks())
        with inp_col2:
            if 'pred' not in st.session_state:
                st.session_state.pred = "Positif"
            st.text_input("Predicted Sentiment", key="pred", value=f"{predict(st.session_state.review)}", disabled=True)

    
with tab2:
    with st.container():
        st.write("#")
        st.write("# **Coming Soon...**")
