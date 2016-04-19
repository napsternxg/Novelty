# coding: utf-8

# Run using pyspark. 
# Look at spark_statup.sh file for details

import glob
import numpy as np
import pandas as pd
import fit_curve as fc
from pyspark import SparkContext, SparkConf

conf = SparkConf()
conf.setMaster("local[30]")
conf.setAppName("Novelty analysis")
conf.set("spark.local.dir", "./tmp")
sc = SparkContext(conf=conf)

def pmid_mesh(x):
    # Row to pmid year mesh
    x = x.split("\t")
    return (x[0], int(x[1]), tuple(x[2].split('|')))

def exploded_names(x):
    # Explode the mesh parents column into into list
    x = x.split('\t')
    return (x[0], tuple(x[1].split('|'))) # Return exploded mesh terms as tuple

mesh_exploded = sc.broadcast(sc.textFile("out/exploded_mhash.tsv").map(exploded_names).collectAsMap())
def get_exploded(k):
    # Get the parents of a mesh term.
    # If mesh is not in dict simply return the mesh
    try:
        return mesh_exploded.value[k]
    except:
        return (k,)

def get_exploded_counts(x):
    # Return list of exploded mesh year count tuples
    total_mesh = set(reduce(lambda a, b: a+b, map(get_exploded, x[2])))
    return (((k,x[1]),1) for k in total_mesh)

def get_exploded_pair_counts(x):
    # Return list of exploded mesh pair year count tuples
    total_mesh = set(reduce(lambda a, b: a+b, map(get_exploded, x[2])))
    total_mesh = sorted(total_mesh, key=lambda x: x.lower)
    mesh_pairs = [(k1,k2) for i,k1 in enumerate(total_mesh) for k2 in total_mesh[i+1:]]
    return (((k1,k2,x[1]),1) for k1,k2 in mesh_pairs)

# LOAD DATA FROM THE FILE
pmid_data = sc.textFile("data/pmid_year_mesh.noheader.tsv").map(pmid_mesh)


# WRITE DATA FOR INDIVIDUAL MESH TERMS. ONLY EXPLODED COUNTS
mesh_exploded_year_count = pmid_data.flatMap(get_exploded_counts).reduceByKey(lambda x,y: x+y)
csv_data = mesh_exploded_year_count.map(lambda x: "%s\t%s\t%s" % (x[0][0], x[0][1], x[1]))
csv_data.saveAsTextFile("out/mesh_exploded_year")

# WRITE DATA FOR PAIRWISE MESH TERMS. ONLY EXPLODED COUNTS
meshpair_exploded_year_count = pmid_data.flatMap(get_exploded_pair_counts).reduceByKey(lambda x,y: x+y)
csv_data = meshpair_exploded_year_count.map(lambda x: "%s\t%s\t%s\t%s" % (x[0][0], x[0][1], x[0][2], x[1]))
csv_data.saveAsTextFile("out/meshpair_exploded_year")

# CALCULATE SCORES FOR PAIRWISE MESH TERMS, ONLY EXPLODED COUNTS
def mesh_pair_scores(x):
  year, counts = zip(*sorted(x, key=lambda k: k[0]))
  year = np.array(year)
  counts = np.array(counts)
  TFirstP = year - year.min()
  VolFirstP = counts.cumsum()
  return ((year[i], counts[i], TFirstP[i], VolFirstP[i]) for i in xrange(len(year)))


def read_mesh_pair(x):
  x = x.split("\t")
  return (x[0], x[1], int(x[2]), int(x[3]))

meshpair_exploded_year_count = sc.textFile(",".join(glob.glob("out/meshpair_exploded_year/part-*"))).map(read_mesh_pair)
meshpair_scores_data = meshpair_exploded_year_count.map(lambda x: ((x[0], x[1]), (x[2], x[3]))).groupByKey().flatMapValues(mesh_pair_scores)
csv_data = meshpair_scores_data.map(lambda x: "%s\t%s\t%s\t%s\t%s\t%s" % (x[0][0], x[0][1], x[1][0], x[1][1], x[1][2], x[1][3]))
csv_data.saveAsTextFile("out/meshpair_exploded_scores")


# Calculate the PMID per Year, In this case we READ it using the file generated from PUBMED2015.Articles table. 
# pmid_per_year = pmid_data.map(lambda x: (x[1], 1)).reduceByKey(lambda a,b: a+b).collectAsMap()
pmid_per_year = pd.read_csv("out/PMID_PER_YEAR.tsv", sep="\t", index_col="year", dtype = {'Year':int, 'TotalPMID':float})
pmid_per_year_norm = pmid_per_year / pmid_per_year.TotalPMID.mean()


# CALCULATE MODEL AND EMPIRICAL SCORES FOR INDIVIDUAL MESH TERMS, ONLY EXPLODED COUNTS
def mesh_scores(x):
  year, counts = zip(*sorted(x, key=lambda k: k[0]))
  # Calculate Empirical scores
  df = pd.DataFrame(data=list(counts), index=list(year), dtype=float)
  df.columns = ["y"]
  TFirstP = df.index - df.index.min()
  VolFirstP = df["y"].values.cumsum()
  # Calculate model scores
  year_min, year_max = df.index.min() - 5, df.index.max() + 1
  year_range = np.arange(year_min, year_max)
  df_y = pd.DataFrame(data=np.zeros_like(year_range), index=year_range, dtype=float)
  
  normalizer = pmid_per_year_norm.ix[df.index, "TotalPMID"]
  df_counts = np.log((df[df.index.isin(year_range)]["y"] / normalizer) + 1) # LOG shoud be base exp as per formula
  df_y["x"] = year_range - year_min
  df_y["y"] = df_counts
  df_y["counts"] = df[df.index.isin(year_range)]["y"]
  df_y = df_y.fillna(0)
  x, y = df_y["x"], df_y["y"]
  # We now have x and y and we need to fit the curve.
  try:
    popt,pcov = fc.getFitCurve(x,y)
  except RuntimeError:
    popt, pcov = (0.0,0.0,0.0), 0.
  # After fitting the curve we only need values for the original years
  #x_org = df_y.ix[df.index, "x"].values
  #predicted_val = fc.func(x_org, popt[0],popt[1],popt[2])
  #velocity = fc.f_der1(x_org, popt)
  #acceleration = fc.f_der2(x_org, popt)
  #df_y.ix[df.index, "predicted"] = predicted_val
  #df_y.ix[df.index, "velocity"] = velocity
  #df_y.ix[df.index, "acceleration"] = acceleration
  #df_y.ix[df.index, "predicted_actual"] = (np.power(10, df_y.ix[df.index, "predicted"]) - 1) * normalizer
  #predicted_actual = df_y.ix[df.index, "predicted_actual"].values
  df["x_fit"] = df.index - year_min
  df["y_norm"] = np.log((df["y"] / normalizer) + 1) # LOG shoud be base exp as per formula
  df["pred"] = fc.func(df["x_fit"], popt[0],popt[1],popt[2])
  df["pred_actual"] = (np.power(10, df["pred"]) - 1) * normalizer
  df["velocity"] = fc.f_der1(df["x_fit"], popt)
  df["acceleration"] = fc.f_der2(df["x_fit"], popt)
  return ((year[i], counts[i], TFirstP[i], VolFirstP[i], \
      df["pred_actual"].values[i], df["velocity"].values[i], df["acceleration"].values[i]) for i in xrange(len(year)))


def read_mesh(x):
  x = x.split("\t")
  return (x[0], int(x[1]), int(x[2]))


mesh_exploded_year_count = sc.textFile(",".join(glob.glob("out/mesh_exploded_year/part-*"))).map(read_mesh)
mesh_scores_data = mesh_exploded_year_count.map(lambda x: ((x[0]), (x[1], x[2]))).groupByKey().flatMapValues(mesh_scores)
csv_data = mesh_scores_data.map(lambda x: "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s" \
    % (x[0], x[1][0], x[1][1], x[1][2], x[1][3], x[1][4], x[1][5], x[1][6]))
csv_data.saveAsTextFile("out/mesh_exploded_scores")




# GEN SCORES FOR ALL PMID
# Make sure to load the data in the database before running the following commands.
# First run the following in ipython
# %run -i scores_helper.py


# RUN INDIVIDUAL SCORES
sc.textFile("data/pmid_year_mesh.noheader.tsv").map(gen_scores)\
    .map(lambda x: "\t".join(["%s"]*len(x)) % x)\
    .saveAsTextFile("out/pmid_novelty_scores")


# RUN PAIR SCORES
sc.textFile("data/pmid_year_mesh.noheader.tsv").map(gen_pair_scores)\
    .map(lambda x: "\t".join(["%s"]*len(x)) % x)\
    .saveAsTextFile("out/pmid_novelty_pair_scores")

# JOIN INDIVIDUAL AND PAIR DATA
individual_scores = sc.textFile("out/pmid_novelty_scores").map(lambda x: x.split()).map(lambda x: (int(x[0]), tuple(x)))
pair_scores = sc.textFile("out/pmid_novelty_pair_scores").map(lambda x: x.split()).map(lambda x: (int(x[0]), tuple(x[2:])))

individual_scores.leftOuterJoin(pair_scores).map(lambda x: x[1][0] + x[1][1])\
  .map(lambda x: "\t".join(["%s"]*len(x)) % x)\
  .saveAsTextFile("out/pmid_novelty_all_scores")


mesh_count = sc.textFile("data/pmid_year_mesh.noheader.tsv").map(get_mesh_counts)
sc.textFile("out/pmid_novelty_all_scores").map(lambda x: x.split("\t"))\
    .map(lambda x: (int(x[0]), tuple(x))).leftOuterJoin(mesh_count)\
    .map(lambda x: x[1][0] + x[1][1])\
    .map(lambda x: "\t".join(["%s"]*len(x)) % x)\
    .saveAsTextFile("out/pmid_novelty_all_scores_mesh_c")

# cat out/pmid_novelty_all_scores_mesh_c/part-* > out/pmid_novelty_all_scores_mesh_c.txt

