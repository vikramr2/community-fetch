import pandas as pd

csv_file = '../data/network/oc_mini_edgelist.csv'

df = pd.read_csv(csv_file)
tsv_file = '../data/network/oc_mini_edgelist.tsv'
df.to_csv(tsv_file, sep='\t', index=False, header=False)

