import recommender as rc
import sequential as seq
import plots as pt

df = rc.open_file('group-reccomandations/data/ml-latest-small/ratings.csv')
group=[23,88,530]
#group_sat=seq.sequential_recommender(group,5,df,10)
#seq.save_to_csv(group,group_sat,"group-reccomandations/data/csv/out.csv")
pt.plot_subgraphs('group-reccomandations/data/csv/out.csv')
pt.plot_group_scores('group-reccomandations/data/csv/out.csv')