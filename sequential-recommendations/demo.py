import recommender as rec
import sequential as seq

df = rec.open_file('data/ml-latest-small/ratings.csv')
group=[]
group_size=3
for i in range(group_size):
    group.append(int(input("Insert userId (1 to 610) ")))
iterations=int(input("Insert number of iterations "))
seq.sequential_recommender(group,iterations,df,10)