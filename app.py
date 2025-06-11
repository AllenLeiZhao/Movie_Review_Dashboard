import streamlit as st
import pandas as pd
import plotly.express as px
from textblob import TextBlob
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt

# --- Load and preprocess data ---
df = pd.read_csv("IMDB_dataset_320.000_reviews.csv", nrows=10000)
df["review_length"] = df["review"].astype(str).apply(len)
df["sentiment_polarity"] = df["review"].astype(str).apply(lambda x: TextBlob(x).sentiment.polarity)

def get_label(p):
    if p > 0.1:
        return "Positive"
    elif p < -0.1:
        return "Negative"
    else:
        return "Neutral"

df["sentiment_label"] = df["sentiment_polarity"].apply(get_label)

# --- Page config ---
st.set_page_config(page_title="IMDB Sentiment Dashboard", layout="wide")
st.title("🎬 IMDB Movie Review Sentiment Dashboard")
st.markdown("**Group Members:** Lei Zhao, Ting Hsuan Yen, Huifan Lin")

st.markdown("""
Explore 10,000 IMDB reviews through interactive sentiment analysis.  
Use the filters in the sidebar to drill into specific genres, keywords, and emotional profiles.
""")

# ======================
# 🔍 SIDEBAR FILTERS
# ======================
st.sidebar.header("🔎 Filters")

sentiment_filter = st.sidebar.multiselect(
    "Sentiment Category", options=["Positive", "Neutral", "Negative"],
    default=["Positive", "Neutral", "Negative"]
)

min_len, max_len = int(df["review_length"].min()), int(df["review_length"].max())
review_len_range = st.sidebar.slider("Review Length", min_len, max_len, (min_len, max_len))

keyword = st.sidebar.text_input("Keyword (optional)").lower()

pol_min, pol_max = float(df["sentiment_polarity"].min()), float(df["sentiment_polarity"].max())
pol_range = st.sidebar.slider("Sentiment Polarity", pol_min, pol_max, (pol_min, pol_max))

genres = df["category"].dropna().unique().tolist()
selected_genres = st.sidebar.multiselect("Genre", options=sorted(genres), default=genres)

titles = df["original title"].dropna().unique().tolist()
selected_title = st.sidebar.selectbox("Movie Title", options=["All"] + sorted(titles))

authors = df["author"].dropna().unique().tolist()
selected_author = st.sidebar.selectbox("Reviewer", options=["All"] + sorted(authors))

# ======================
# ✅ APPLY FILTERS
# ======================
filtered_df = df[
    df["sentiment_label"].isin(sentiment_filter) &
    df["review_length"].between(review_len_range[0], review_len_range[1]) &
    df["sentiment_polarity"].between(pol_range[0], pol_range[1]) &
    df["category"].isin(selected_genres)
]

if selected_title != "All":
    filtered_df = filtered_df[filtered_df["original title"] == selected_title]
if selected_author != "All":
    filtered_df = filtered_df[filtered_df["author"] == selected_author]
if keyword:
    filtered_df = filtered_df[filtered_df["review"].str.lower().str.contains(keyword)]

# ======================
# 🔢 KPIs
# ======================
col1, col2, col3, col4 = st.columns(4)
col1.metric("📊 Total Reviews", f"{len(filtered_df):,}")
col2.metric("❤️ Avg Polarity", round(filtered_df["sentiment_polarity"].mean(), 3))
col3.metric("📝 Max Length", f"{filtered_df['review_length'].max():,} chars")
col4.metric("🙂 Positive Count", filtered_df["sentiment_label"].value_counts().get("Positive", 0))
st.markdown("---")

# ======================
# 📊 CHARTS
# ======================
col5, col6 = st.columns(2)
sentiment_counts = filtered_df["sentiment_label"].value_counts().reset_index()
sentiment_counts.columns = ["sentiment_label", "count"]

with col5:
    fig1 = px.pie(sentiment_counts, names="sentiment_label", values="count", title="Sentiment Distribution")
    st.plotly_chart(fig1, use_container_width=True)

with col6:
    fig2 = px.scatter(filtered_df, x="sentiment_polarity", y="review_length",
                      color="sentiment_label", title="Polarity vs. Review Length",
                      opacity=0.5)
    st.plotly_chart(fig2, use_container_width=True)

col7, col8 = st.columns(2)
with col7:
    fig3 = px.histogram(filtered_df, x="review_length", nbins=50, color="sentiment_label",
                        title="Review Length Distribution")
    st.plotly_chart(fig3, use_container_width=True)

with col8:
    fig4 = px.box(filtered_df, x="sentiment_label", y="review_length",
                  title="Review Length by Sentiment")
    st.plotly_chart(fig4, use_container_width=True)

st.markdown("### 📈 Sentiment Polarity Distribution")
fig5 = px.histogram(filtered_df, x="sentiment_polarity", nbins=50, marginal="rug",
                    color="sentiment_label", opacity=0.6, histnorm='probability density')
st.plotly_chart(fig5, use_container_width=True)

st.markdown("### ☁️ Word Cloud of Selected Reviews")
text = " ".join(filtered_df["review"].astype(str).tolist())
wordcloud = WordCloud(stopwords=STOPWORDS, background_color="white", width=800, height=400).generate(text)
fig_wc, ax = plt.subplots(figsize=(10, 5))
ax.imshow(wordcloud, interpolation="bilinear")
ax.axis("off")
st.pyplot(fig_wc)
