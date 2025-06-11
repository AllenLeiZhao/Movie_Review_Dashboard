import pandas as pd

df = pd.read_csv("IMDB_dataset_320.000_reviews.csv")
df.head(10000).to_csv("IMDB_10k.csv", index=False)
