import swifter
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from nltk import ngrams
from nltk import FreqDist


class ExploratoryDataAnalysis:
    def __init__(self, df :pd.DataFrame):
        self.df = self.preproccess(df)

    def preproccess(self, eda_df:pd.DataFrame):
        eda_df['sentiment'] = eda_df['sentiment'].swifter.apply(lambda x: x.replace('Positif', 'Positive').replace('Netral', 'Neutral').replace('Negatif', 'Negative'))
        eda_df['char'] = eda_df['cleaned'].swifter.apply(lambda x: len(' '.join(x)))
        eda_df['word'] = eda_df['cleaned'].swifter.apply(lambda x: len(x))
        eda_df['mean'] = eda_df['cleaned'].swifter.apply(lambda x : [len(i) for i in x]).map(lambda x: np.mean(x))
        eda_df['unique'] = eda_df['cleaned'].swifter.apply(lambda x: len(set(' '.join(x))))
        return eda_df

    def Class_plot(self, ax):
        plot = sns.countplot(x='sentiment', data=self.df, palette='viridis', ax=ax)
        plot = plot.set(title="Sentiment Distribution", xlabel="Sentiment", ylabel="Counts")
        return plot

    def Freq_char(self, ax):
        plot = sns.histplot(data=self.df, x='char', bins=np.arange(0, 260, 10), hue='sentiment', kde=True, palette='hls', ax=ax)
        plot = plot.set(title="Frequency Distribution Number of Characters per Data Based on Sentiment", xlabel="Number of Characters", ylabel="Frequency")
        return plot
    
    def Freq_word(self, ax):
        plot = sns.histplot(data=self.df, x='word', bins=np.arange(0, 50), hue='sentiment', kde=True, palette='hls', ax=ax)
        plot = plot.set(title="Frequency Distribution Number of Words per Data Based on Sentiment", xlabel="Number of Words", ylabel="Frequency")
        return plot
    
    def Freq_mean(self, ax):
        plot = sns.histplot(data=self.df, x='mean', hue='sentiment', kde=True, palette='hls', ax=ax)
        plot = plot.set(title="Average Word Length Frequency Distribution Based on Sentiment", xlabel="Average Word Length", ylabel="Frequency")
        return plot

    def Freq_unique(self, ax):
        plot = sns.histplot(data=self.df, x='unique', bins=range(self.df['unique'].min(), self.df['unique'].max() + 1), hue='sentiment', kde=True, palette='hls', ax=ax)
        plot = plot.set(title="Frequency Distribution Number of Unique Characters Based on Sentiment", xlabel="Number of Unique Characters", ylabel="Frequency")
        return plot

    def Freq_ngrams(self, cat_sentiment, n=3):
        sentiment_text = [word for tweet in self.df['cleaned'] for word in tweet]
        ngram_text = list(ngrams(sentiment_text, n))
        fqdist = FreqDist(ngram_text)

        return fqdist.plot(30, cumulative=False, title=f'{n}-gram Distribution on {cat_sentiment} Sentiment')



if __name__ == "__main___":
    data = pd.read_pickle("assets/datasets/sentiment.pkl")
    eda = ExploratoryDataAnalysis(data)
    
