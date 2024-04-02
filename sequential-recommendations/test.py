import recommender as rc
import sequential as seq
import plots as pt
import csv
df = rc.open_file('data/ml-latest-small/ratings.csv')
group=[2,10,241]
group_sat=seq.sequential_recommender(group,4,df,10)
seq.save_to_csv(group,group_sat,"data/csv/hete.csv")
pt.plot_subgraphs('data/csv/hete.csv','data/plots/hete-sat.png')
pt.plot_group_scores('data/csv/hete.csv','data/plots/hete-dis-sat.png')
