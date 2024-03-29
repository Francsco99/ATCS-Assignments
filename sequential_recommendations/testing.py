import pandas as pd
from itertools import islice
df = pd.read_csv("group-reccomandations/data/csv/final.csv")

pivot_df = df.pivot(index='movieId', columns='userId', values='rating')
pivot_df['mean'] = pivot_df.mean(axis=1)
pivot_df['min'] = pivot_df.min(axis=1)
pivot_df.reset_index(inplace=True)
pivot_df.dropna(inplace=True)
print(pivot_df)
map ={}
for index, row in pivot_df.iterrows():
    movie_id = int(row['movieId'])
    mean_rating = float(row['mean'])
    min_rating = float(row['min'])
    map[movie_id]=[mean_rating,min_rating]
for k,v in islice(map.items(),10):
    print(k,v)