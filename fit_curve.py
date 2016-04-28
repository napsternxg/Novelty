import numpy as np
from scipy import interpolate, optimize
import pandas as pd

import connectDB as cdb

# Fit function
#def func(x, b, c, d): 
def func(x, b, q, p):
  #return (b/(1.0+np.exp(-(c+d*x))))
  return (b/(1.0+np.exp(-(x-q)/p)))

# First derivative
def f_der1(x, popt):
  b,q,p = popt[:]
  c = -q/p
  d = 1./p
  #e = 2+ np.exp(-(c+d*x)) + np.exp(c+d*x)
  #e = np.exp(-(c + d*x)) / np.power((1 + np.exp(-(c+d*x))), 2)
  e = np.exp(-(x-q)/p) / np.power((1 + np.exp(-(x-q)/p)), 2)
  #return b*d/e
  return b*e/p

# Second derivative
def f_der2(x, popt):
  b,q,p = popt[:]
  c = -q/p
  d = 1./p
  #t1 = b*d/np.power((2+ np.exp(-(c+d*x)) + np.exp(c+d*x)),2)
  #t2 = (d*np.exp(-(c+d*x)) - d*np.exp(c+d*x))
  #t1 = b*d*d/np.power((1 + np.exp(-(c+d*x))), 3)
  #t2 = np.exp(-(c+d*x)) * (np.exp(-(c+d*x)) - 1)
  t1 = (b/p**2)/np.power((1 + np.exp(-(x-q)/p)), 3)
  t2 = np.exp(-(x-q)/p) * (np.exp(-(x-q)/p) - 1)
  return t1*t2

# Fit the curve for the data
def getFitCurve(x1,y):
  #return optimize.curve_fit(func,x1,y, p0=[0.5,-2,0.1])
  return optimize.curve_fit(func,x1,y, p0=[1,-1,1])
  #return optimize.curve_fit(func,x1,y)


pmid_per_year = pd.read_csv("PMID_PER_YEAR.tsv", sep="\t", index_col="year", dtype = {'Year':int, 'TotalPMID':float})
pmid_per_year_norm = pmid_per_year / pmid_per_year.TotalPMID.mean()


def getData(mesh_term=None, normalized=True):
  mesh_data = None
  if mesh_term is not None:
    query = "SELECT Year,AbsVal FROM mesh_scores WHERE MeshTerm = :mterm"
    r = cdb.session.execute(query, {"mterm": mesh_term})
    mesh_data = r.fetchall()
  return mesh_data

# CALCULATE MODEL AND EMPIRICAL SCORES FOR INDIVIDUAL MESH TERMS, ONLY EXPLODED COUNTS
def mesh_scores(x):
  year, counts = zip(*sorted(x, key=lambda k: k[0]))
  # Calculate Empirical scores
  
  #df = pd.DataFrame(data=list(counts), index=list(year), dtype=float)
  #df.columns = ["y"]
  # Collect all year and counts from spelling errors
  df = pd.DataFrame({"x": list(year), "y": list(counts)})
  df = df.groupby("x").sum()
  year, counts = df.index.values, df["y"].values

  # Start processing
  df["y"] = df["y"].astype("float")
  TFirstP = df.index - df.index.min()
  VolFirstP = df["y"].values.cumsum()
  # Calculate model scores
  #year_min, year_max = df.index.min() - 5, df.index.max() + 1
  popt, pcov = None, None
  for offset in [0,2,5,10]:
      #print offset
      #year_min, year_max = df.index.min()-offset, min(df.index.max() + 1, 2015)
      modified_min_year = max(df.index.min(), 1966)
      year_min, year_max = modified_min_year-offset, min(df.index.max() + 1, 2015)
      #if df.index.min() < 1966:
      # Don't fit curve for any MeSH term which was started before 1966
      #  break
      #print offset
      #year_range = np.arange(year_min, year_max)
      year_range = range(year_min, modified_min_year) + df.index[(df.index <= year_max) & (df.index>=modified_min_year)].values.tolist()
      #year_min, year_max = df.index.min()-offset, min(df.index.max() + 1, 2015)
      #year_range = np.arange(year_min, year_max)
      df_y = pd.DataFrame(data=np.zeros_like(year_range), index=year_range, dtype=float)
      normalizer = pmid_per_year_norm.ix[df.index, "TotalPMID"]
      df_counts = np.log((df[df.index.isin(year_range)]["y"] / normalizer) + 1) # LOG shoud be base exp as per formula
      df_y["x"] = np.array(year_range) - year_min
      df_y["y"] = df_counts
      df_y["counts"] = df[df.index.isin(year_range)]["y"]
      df_y = df_y.fillna(0)
      x, y = df_y["x"], df_y["y"]
      # We now have x and y and we need to fit the curve.
      if df.shape[0] < 2:
          # Too few instances to fit
          break
      if len(year_range) < 3:
          continue
      try:
          popt,pcov = getFitCurve(x,y)
          #print popt
          if popt[0]*popt[2] > 0:
              break
      except RuntimeError:
          popt, pcov = None, None
  # After fitting the curve we only need values for the original years
  #x_org = df_y.ix[df.index, "x"].values
  #predicted_val = fc.func(x_org, popt[0],popt[1],popt[2])
  #velocity = fc.f_der1(x_org, popt)
  df["x_fit"] = df.index - year_min
  x_inflex = None
  df["y_norm"] = np.log((df["y"] / normalizer) + 1) # LOG shoud be base exp as per formula
  if popt is not None and popt[0]*popt[2] > 0:
      #x_inflex = -popt[1]/popt[2]
      x_inflex = int(popt[1] + year_min + 1)
      df["pred"] = func(df["x_fit"], popt[0],popt[1],popt[2])
      df["pred_actual"] = (np.exp(df["pred"]) - 1) * normalizer
      df["velocity"] = f_der1(df["x_fit"], popt)
      df["acceleration"] = f_der2(df["x_fit"], popt)
  else:
      df["pred"] = np.nan
      df["pred_actual"] = np.nan
      df["velocity"] = np.nan
      df["acceleration"] = np.nan
  df = df.fillna("\\N")
  return popt, offset, x_inflex, ((year[i], counts[i], TFirstP[i], VolFirstP[i], df["y_norm"].values[i], df["pred"].values[i],\
      df["pred_actual"].values[i], df["velocity"].values[i], df["acceleration"].values[i]) for i in xrange(len(year)))


def get_plots(gen_x):
  colors = {
      'Black': '#000000',
      'Red':   '#FF0000',
      'Green': '#00FF00',
      'Blue':  '#0000FF',
  }
  year, counts, y_norm, pred, pred_actual, velocity, acceleration = zip(
      *((k[:2] + k[4:]) for k in gen_x))

  plot_size = 800
  plot_height = 400
  TOOLS = 'box_zoom,box_select,crosshair,resize,reset,hover,save'
  colors_i = [colors["Blue"], colors["Black"]]
  count_fig = figure(x_axis_label = "Year", y_axis_label = "# Articles per year",\
      plot_width=plot_size, plot_height=plot_height, y_axis_type="log", tools=TOOLS)
  count_fig.line(year, pred_actual, color=colors_i[0], line_width=2, legend="predicted")
  count_fig.scatter(year, counts, color=colors_i[0], line_width=2, legend="actual")
  count_fig.legend.orientation = "top_left"
  
  norm_fig = figure(x_axis_label = "Year", y_axis_label = "Normed count",\
      plot_width=plot_size, plot_height=plot_height, tools=TOOLS)
  norm_fig.line(year, pred, color=colors_i[0], line_width=2, legend="predicted")
  norm_fig.line(year, np.convolve(y_norm, np.ones(5)/5., "same"), color=colors_i[1], line_width=2, legend="Mov. Avg.")
  norm_fig.scatter(year, y_norm, color=colors_i[0], line_width=2, legend="actual")
  norm_fig.set(y_range=Range1d(0, max(y_norm)))
  norm_fig.legend.orientation = "top_left"

  velocity_fig = figure(x_axis_label = "Year", y_axis_label = "Velocity",\
      plot_width=plot_size, plot_height=plot_height, tools=TOOLS)
  velocity_fig.line(year, velocity, color=colors_i[0], line_width=2)

  acc_fig = figure(x_axis_label = "Year", y_axis_label = "Acceleration",\
      plot_width=plot_size, plot_height=plot_height, tools=TOOLS)
  acc_fig.line(year, acceleration, color=colors_i[0], line_width=2)
  
  fig = vplot(count_fig, norm_fig, velocity_fig, acc_fig)
  df = pd.DataFrame({"year": year, "counts": counts, "y_norm": y_norm, "pred": pred,
    "pred_actual": pred_actual, "velocity": velocity, "acceleration": acceleration})
  return fig, df

def plot_to_file(gen_x, x_inflex):
  colors = {
      'Black': '#000000',
      'Red':   '#FF0000',
      'Green': '#00FF00',
      'Blue':  '#0000FF',
  }
  year, counts, y_norm, pred, pred_actual, velocity, acceleration = zip(
      *((k[:2] + k[4:]) for k in gen_x))
  x = np.array(year)
  plot_size = 800
  plot_height = 400
  colors_i = [colors["Black"], colors["Blue"]]
  fig, ax = plt.subplots(4,1, sharex="col", figsize=(8,8))

  ## Count plot
  ax[0].plot(year, pred_actual, color=colors_i[0], label="predicted")
  ax[0].plot(year, counts, color=colors_i[0], marker="o", linestyle="None", label="actual")
  ax[0].axvline(x=x_inflex,linestyle=":", color="red")
  ax[0].axvspan(x[(x <= x_inflex) & (np.array(velocity) > 0.1)].min(), x_inflex, facecolor="red", alpha=0.2, label="Acc. growth")
  ax[0].axvspan(x_inflex, x[(x >= x_inflex) & (np.array(velocity) > 0.1)].max(), facecolor="blue", alpha=0.2, label="Dec. growth")
  ax[0].set_yscale("log")
  ax[0].set_ylabel("# Articles per year")
  ax[0].legend(loc="lower right", ncol=2,frameon=True, fancybox=True)

  ## Normed fig
  ax[1].plot(year, pred, color=colors_i[0], label="predicted")
  ax[1].plot(year, y_norm, color=colors_i[0], marker="o", linestyle="None", label="actual")
  ax[1].axvline(x=x_inflex,linestyle=":", color="red")
  ax[1].axvspan(x[(x <= x_inflex) & (np.array(velocity) > 0.1)].min(), x_inflex, facecolor="red", alpha=0.2, label="Acc. growth")
  ax[1].axvspan(x_inflex, x[(x >= x_inflex) & (np.array(velocity) > 0.1)].max(), facecolor="blue", alpha=0.2, label="Dec. growth")
  ax[1].set_ylim([0, max(y_norm)])
  ax[1].set_ylabel("Normed count")
  ax[1].legend(loc="lower right", ncol=2,frameon=True, fancybox=True)
  ## Velocity fig
  ax[2].plot(year, velocity, color=colors_i[0])
  ax[2].axvline(x=x_inflex,linestyle=":", color="red")
  ax[2].set_ylabel("Velocity")
  ## Acceleration fig
  ax[3].plot(year, acceleration, color=colors_i[0])
  ax[3].axvline(x=x_inflex,linestyle=":", color="red")
  ax[3].set_xlabel("Year")
  ax[3].set_ylabel("Acceleration")
  sns.despine(trim=True, offset=2)
  fig.tight_layout()
  df = pd.DataFrame({"year": year, "counts": counts, "y_norm": y_norm, "pred": pred,
    "pred_actual": pred_actual, "velocity": velocity, "acceleration": acceleration})
  return fig, df


def main(filename="Mesh_plot.pdf", mesh_term="Neoplasms", normalized=True, saveFigure=True, debug=False, offset=0):
  mesh_data = getData(mesh_term = mesh_term)
  popt, offset, x_inflex, gen_x = mesh_scores(mesh_data)
  fig, df = plot_to_file(gen_x, x_inflex)
  plt.savefig(filename, bbox_inches="tight")
  print df


if __name__ == "__main__":
  import argparse
  parser = argparse.ArgumentParser(description="Create Plots for Mesh Term\
                                   Data")
  parser.add_argument("-f",help="Name of file to get data from. Should\
                      contain\
                      2 columns only seperated by tab. Year and Count")
  parser.add_argument("-m", type=str, help="Full name of MeSH term overrides filename")
  parser.add_argument("-n", type=bool, help="Normalize counts. Set to activate")
  parser.add_argument("-s", type=bool, help="Save figure and not show. Set to activate")
  parser.add_argument("-t", type=bool, help="Running from terminal. Set to activate")
  parser.add_argument("--offset", type=int, help="Offset from beginning", default=2)
  parser.add_argument("-d", type=bool, help="Debug", default=False)
  args = parser.parse_args()
  filename = args.f
  mesh_term = args.m
  normalized = args.n
  saveFigure  = args.s
  debug = args.d
  offset = args.offset 
  if filename is None:
      print parser.print_help()
      exit(-1)
  if normalized is None:
      normalized == False
  if saveFigure is None:
      saveFigure == False
  if args.t is not None:
      import matplotlib
      matplotlib.use('Agg')
      saveFigure = True

  import matplotlib.pyplot as plt 
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

  print "Filename: %s, MeSHTerm: %s, Normalized: %s, saveFigure: %s, debug: %s" % (filename, mesh_term, normalized, saveFigure, debug)
  main(filename, mesh_term, normalized, saveFigure, debug, offset)

