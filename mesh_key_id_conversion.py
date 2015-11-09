# coding: utf-8
####
# Convert MeSH names to ids and keys
# Store the results in auxilary files for later usage.
#####



import pandas as pd
df = pd.read_csv("data/mtrees2015.bin", sep=";", header=None)
df_ids = df.groupby(0).agg(lambda x: list(x.values))
df_ids.head()
mesh_keys = {k: i for i,k in enumerate(df[0].unique())}
mesh_key2name = df[0].unique()
temp = [mesh_keys[k] for k in df_ids.index]
df_ids["keys"] = temp
df_ids["joined"] = df_ids[1].apply(lambda x: "|".join(x))
df_ids.sort("keys").to_csv("out/mesh_id_key.tsv", sep="\t",header=False, columns=["keys","joined"])
df["keys"] = df[0].apply(lambda x: mesh_keys[x])
df_keys = df[[1,"keys"]].groupby(1).first()
df_keys.sort("keys").to_csv("out/mesh_id2key.tsv", sep="\t",header=False)
