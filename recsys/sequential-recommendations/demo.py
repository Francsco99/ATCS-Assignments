import recommender as rec
import sequential as seq
import os

"""File paths"""
data_dir = os.path.dirname(__file__)
folder = os.path.basename(data_dir)
base_path = data_dir.replace(folder,"")

ratings_dir = os.path.join(base_path,"data","ml-latest-small","ratings.csv")

df = rec.open_file(ratings_dir)
group=[]
group_size=3
for i in range(group_size):
    group.append(int(input("Insert userId (1 to 610) ")))
iterations=int(input("Insert number of iterations "))
seq.sequential_recommender(group,iterations,df,10)