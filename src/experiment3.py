import sys
from collections import Counter
import pandas as pd

#Experiment 3 - Highest and lowest valence sentences
TOP_PCT = 10

transformer = sys.argv[2]
projection_df = pd.read_csv(f'../output/projection_products_permutations_'+transformer+'.csv',index_col=0)

phrases = projection_df.index.tolist()
idxs = [i for i in projection_df.index.tolist() if len(i.split(' ')) == 7] #Only take five-word sentences
five_df = projection_df.loc[idxs]
top_ = int(len(five_df.index.tolist())/TOP_PCT)

#Top Projections
top_proj = five_df.nlargest(top_,'projection')

print("*****************")
print("most pleasant\n")
#Count by group
k = [i.split(' ') for i in top_proj.index.tolist()]
j = [item for sublist in k for item in sublist]
c = Counter(j)
print(Counter(j))

#Print percentages by group
for k in c.keys():
    count = c[k]
    pct = count/top_
    print(k)
    print(pct)


#Get smallest projection products
small_proj = five_df.nsmallest(top_,'projection')

print("*****************")
print("most unpleasant\n")

#Count by group
k = [i.split(' ') for i in small_proj.index.tolist()]
j = [item for sublist in k for item in sublist]
c = Counter(j)
print(Counter(j))

#Print percentages by group
for k in c.keys():
    count = c[k]
    pct = count/top_
    print(k)
    print(pct)