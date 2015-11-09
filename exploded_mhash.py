# coding: utf-8

import re
import pandas as pd

# Read the MeSH tree and Create 2 dataframes for quick access.
# Using Dictionary might be better. Should check this out later.
df = pd.read_csv("data/mtrees2015.bin", sep=";", header=None)
df_ids = df.groupby(0).agg(lambda x: list(x.values))
df_names = df.groupby(1).first()

# Function which returns all the parents of a given MeSH term k. 
def get_exploded(k):
    regex = []
    temp_mesh_items = set()
    if k not in df_ids.index:
        return temp_mesh_items
    # For each MeSH id append to the regex all its parent IDs
    for mid in df_ids.ix[k,1]:
        tags = mid.split(".")
        tags = ["\.".join(tags[:t]) for t in range(1,len(tags)+1)]
        regex.extend(tags)
    # Create a big regex for quick searching.
    regex = "|".join(set(regex))
    regex = re.compile(r"^("+regex+")$")
    # Find all matching parents in the ID_name DF. 
    for key in df_names.index:
        if regex.match(key) is not None:
            temp_mesh_items.add(df_names.ix[key, 0])
    return temp_mesh_items
        

# Print to file
with open("out/exploded_mhash.tsv", "wb+") as fp:
    for k in df_ids.index:
        exploded_mt = "|".join(get_exploded(k))
        if exploded_mt == '':
            continue
        print >> fp, "{0}\t{1}".format(k,exploded_mt)
        
