import recommender as rc
import sequential as seq
import plots as pt
df = rc.open_file('data/ml-latest-small/ratings.csv')
group=[493, 26, 57]
group_sat=seq.sequential_recommender(group,4,df,10)
seq.save_to_csv(group,group_sat,"data/csv/homo.csv")
pt.plot_subgraphs('data/csv/homo.csv','data/plots/homo-sat.png')
pt.plot_group_scores('data/csv/homo.csv','data/plots/homo-dis-sat.png')
