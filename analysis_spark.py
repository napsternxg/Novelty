import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import ScalarFormatter

import numpy as np
import pandas as pd

from pyspark import SparkContext, SparkConf
from pyspark.sql import SQLContext
from pyspark.sql import functions as F
from pyspark.sql import types as T

import seaborn as sns
sns.set_style('ticks')
sns.set_context('paper')

matplotlib.rcParams['font.size'] = 10
matplotlib.rcParams['axes.labelsize'] = 10
matplotlib.rcParams['xtick.labelsize'] = 10
matplotlib.rcParams['ytick.labelsize'] = 10
matplotlib.rcParams['axes.titlesize'] = 10
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['savefig.dpi'] = 600
matplotlib.rcParams['lines.markersize'] = 3
matplotlib.rcParams['lines.linewidth'] = 1

pd.options.display.width = 200

conf = SparkConf()
conf.setMaster("local[30]")
conf.setAppName("Author Roles")
conf.set("spark.local.dir", "../tmp")
conf.set("spark.executor.memory", "50g")
conf.set("spark.driver.maxResultSize", "50g")
conf.set("spark.shuffle.consolidateFiles", "true")
#conf.set("spark.serializer", "org.apache.spark.serializer.KryoSerializer")

sc = SparkContext(conf=conf)
sqlContext = SQLContext(sc)

DATA_DIR = "data"
PLOT_DIR = "plots"
OUT_DIR = "output"


columns = ["PMID", "Year", "AbsVal", "TFirstP", "VolFirstP", "acc_pos_vel_min", "acc_pos_vel_max", "acc_neg_vel_max", "acc_neg_vel_min", "Pair_AbsVal", "Pair_TFirstP", "Pair_VolFirstP", "Mesh_counts", "Exploded_Mesh_counts"]

df = sqlContext.read.format("csv").options(header='false', inferschema='true', delimiter='\t').load("out/pmid_novelty_all_scores_mesh_c")
df = df.selectExpr(*("%s as %s" % (df.columns[i], k) for i,k in enumerate(columns)))


for k in ["acc_pos_vel_min", "acc_pos_vel_max", "acc_neg_vel_max", "acc_neg_vel_min", "Pair_AbsVal", "Pair_TFirstP", "Pair_VolFirstP"]:
  df = df.withColumn(k, df[k].cast(T.DoubleType()))



# Get data for distribution of Scores
score_types = ["TFirstP", "VolFirstP", "Pair_TFirstP", "Pair_VolFirstP"]
bins = [range(60), reduce(lambda x, y: x + y, [[0]] + [(10**k*np.arange(1,10)).tolist() for k in range(6)])]
data = []
for i, k in enumerate(score_types):
  print "Processing %s" % k
  j = i % 2
  x, y = df[(df["Year"] >= 1985)].rdd.map(lambda x: x[k]).histogram(bins[j])
  data.append((x,y))



# Plot distribion data
plt.close("all")
plt.clf()
fig, ax = plt.subplots(1,2, figsize=(6,3.2))

y_label = "Cumulative proportion of papers\nin MEDLINE since 1985"
x_labels = ["Years since first publication", "Papers since first publication"]
labels = ["Individual concept", "Pair of concepts"]
colors = ["black", "red"]
markers = ["s", "o"]

for i, k in enumerate(score_types):
  print "Plotting %s" % k
  j = i % 2
  x, y = data[i]
  x = np.array(x[:-1])
  if "VolFirstP" in k:
    x = x + 1.0
  y = np.cumsum(y) * 1.0 / np.sum(y)
  ax[j].plot(x, y, marker=markers[i/2], color=colors[i/2], label=labels[i/2], lw=2)

ax[0].set_ylabel(y_label)
ax[0].set_xlabel(x_labels[0])
ax[1].set_xscale("log")
ax[1].set_xlabel(x_labels[1])
lgd = fig.legend(*ax[0].get_legend_handles_labels(),
    loc = 'upper center',bbox_to_anchor=(0.5, 1.2),
    title = "Type of Novelty Score",
    ncol=2, frameon=True, fancybox=True)
plt.savefig("%s/ScoreDistribution.pdf" % PLOT_DIR, bbox_inches='tight', bbox_extra_artists=[lgd])

# Growth data
score_types = ["acc_pos_vel_min", "acc_neg_vel_max"]
bins = np.arange(0,2.01,0.01).tolist()

data_acc_growth = df[(df["Year"] >= 1985) & (~df["acc_pos_vel_min"].isNull())].rdd.map(lambda x: x["acc_pos_vel_min"]).histogram(bins)
data_dec_growth = df[(df["Year"] >= 1985) & (df["acc_pos_vel_min"].isNull())].rdd.map(lambda x: x["acc_neg_vel_max"]).histogram(bins)

data = []
for i, k in enumerate(score_types):
  print "Processing %s" % k
  j = i % 2
  x, y = df[(df["Year"] >= 1985)].rdd.map(lambda x: x[k]).histogram(bins[j])
  data.append((x,y))


df_mesh = pd.read_csv("../data/MeSHProfiles.txt", sep="\t")
df_pmid = pd.read_csv("../data/PMID_PER_YEAR.2015.txt", sep="\t")
df_pmid["norm"] = df_pmid["TotalPMID"] / df_pmid["TotalPMID"].mean()

mterm = "Neoplasms"

plt.clf()
plt.close("all")
fig, ax = plt.subplots(4,1,figsize=(5,5), sharex=True)
for mterm in df_mesh.MeshTerm.unique():
  print "Plotting %s" % mterm
  for i, k in enumerate(["VolFirstP", "PredVal", "Velocity", "Acceleration"]):
    x, y = df_mesh[df_mesh["MeshTerm"] == mterm].Year.values, df_mesh[df_mesh["MeshTerm"] == mterm][k].values
    if k == "Velocity":
      y = y / y.max()
    elif k == "Acceleration":
      y = y / (y.max() - y.min())
    ax[i].plot(x,y, label=mterm)

ax[0].set_ylim([1,1e6])
ax[1].set_ylim([1,1e6])
ax[3].set_xlim([1920,2010])
ax[0].set_yscale("log")
ax[1].set_yscale("log")
plt.savefig("../plots/ALLMeshProfile.pdf", bbox_inches="tight")

