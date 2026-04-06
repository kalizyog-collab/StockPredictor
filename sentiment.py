from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import pandas as pd

# Create the sentiment analyzer once
analyzer = SentimentIntensityAnalyzer()


def get_sentiment_score(text):
    """
    Convert a headline into one sentiment score.

    Output:
    - closer to 1   -> positive
    - closer to -1  -> negative
    - near 0        -> neutral
    """

    # If text is empty, return neutral sentiment
    if not isinstance(text, str) or text.strip() == "":
        return 0.0

    # VADER returns several values, we use 'compound'
    scores = analyzer.polarity_scores(text)
    return scores["compound"]


def create_demo_headlines(df):
    """
    Create simple demo headlines based on stock movement.

    This lets the project run without needing a real news API.
    These headlines are only for demonstration.
    """

    # Work on a copy so the original data is not changed directly
    df = df.copy()

    headlines = []

    for i in range(len(df)):
        # First row has no previous day to compare with
        if i == 0:
            headlines.append("Market opens with stable sentiment")
        else:
            # If today's close is higher than yesterday -> positive headline
            if df.loc[i, "Close"] > df.loc[i - 1, "Close"]:
                headlines.append("Positive investor sentiment after stock gains")

            # If today's close is lower than yesterday -> negative headline
            elif df.loc[i, "Close"] < df.loc[i - 1, "Close"]:
                headlines.append("Negative investor sentiment after stock decline")

            # If same close -> neutral headline
            else:
                headlines.append("Neutral market sentiment as stock stays flat")

    # Add the headlines to the DataFrame
    df["Headline"] = headlines

    return df


def add_sentiment_column(df, headline_col="Headline"):
    """
    Turn each headline into a sentiment score
    and store it in a new Sentiment column.
    """

    df["Sentiment"] = df[headline_col].apply(get_sentiment_score)
    return df