#!/usr/bin/env python

import os

import numpy as np
from scipy import interpolate, optimize

import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns


"""
Year wise cumulative count of
pubmed ids
"""

def normalizeAndLogCount(k):
    global tyd
    #print "Initial k: ", k
    delta  = tyd.TotalPMID.mean()
    normalizeF =  1.0*(tyd[tyd.year == k["x"]].TotalPMID.values[0])
    k["y"] = (k["y"]*delta)/normalizeF
    k["y"] = np.log(k["y"]+1)
    #print "Final k: ", k
    return k

def prepareData(df,normalized):
    global tyd
    df[["x","y"]] =df[["x","y"]].astype(float)
    df["y_org"] = df["y"]
    if normalized:
        #print "Using Normalized routine"
        tyd = pd.read_csv("PMID_PER_YEAR.tsv", sep="\t")
        #tyd = tyd[(tyd.year > 1901) & (tyd.year <= 2011)] 
        df = df.apply(normalizeAndLogCount, axis=1)
        df[["x"]] = df[["x"]].astype(int)
    else:
        df.y = df.y.apply(lambda y: np.log(y+1))

    #print df

    min_year = df.x.min() - 5
    max_year = df.x.max() -1
    x = range(min_year,max_year)
    x1 = range(0,max_year-min_year)
    y = []
    y_org = []
    for i in x:
        if i not in df.x.values:
            y.append(0)
            y_org.append(0)
        else:
            y.append(df[df.x == i].y.values[0])
            y_org.append(df[df.x == i].y_org.values[0])

    #y = [0 if i not in df.x else df[df.x == i].y for i in x]
    #print df.y
    #print x,x1,y
    return x,x1,y, y_org

def getData(filename, normalized):
    global tyd
    column_names = ["x","y"]
    # Important step. TotalPMID should be of type float or else there will be error in normalized calculations
    df = pd.read_csv(filename,sep="\t", dtype = {'Year':int, 'TotalPMID':float})
    df.columns = column_names
    return prepareData(df, normalized)


def plot(x,y,x2,r,y2,y3,y4,y_org, saveFigure=False, filename="None", ax = None, color=sns.xkcd_rgb["black"],marker='o', linestyle='-',label='',alpha=1.0,normalized=True,**kwargs):
    #print x,y,x2,r,y2,y3,y4
    sns.set_style("whitegrid")
    title  = "Curve fitting for MeshTerm"
    ax_defined = True
    if normalized:
        title += " [Normalized]"
        filename += "_Normalized"
    if ax is None:
      ax_defined = False
      fig, ax = plt.subplots(4,1, sharex=True)
      fig.set_size_inches(15,12)
      title = label
      label = 'Data'
    ax[0].plot(x,y_org,marker=marker,markersize=7,linestyle='None',\
        color=color, alpha=alpha,label=label,**kwargs)
    #ax[0].plot(x,r,"-")
    ax[0].set_yscale('log')
    ax[0].set_ylabel("Original Counts")

    ax[1].plot(x,y,marker=marker,markersize=7,linestyle='None',\
        color=color,alpha=alpha,label=label,**kwargs)
    ax[1].plot(x2,y2,linestyle=linestyle,color=color,\
        alpha=alpha,label="Fit "+label, **kwargs)
    ax[1].set_ylabel("Log Normalized Counts")

    ax[2].plot(x2,y3,linestyle=linestyle,color=color,alpha=alpha,label=label,**kwargs)
    ax[2].set_ylabel("Growth")

    ax[3].plot(x2,y4,linestyle=linestyle,color=color,alpha=alpha,label=label, **kwargs)
    ax[3].set_ylabel("Acceleration")
    ax[3].set_xlabel("Year")
    
    if saveFigure and (not ax_defined):
        ax[1].legend(loc="upper left")
        #filename = filename + ".png"
        ax[0].set_title(title)
        plt.savefig(filename+ ".png", bbox_inches = "tight")
        plt.savefig(filename+ ".eps", bbox_inches = "tight")
        print "Figure saved as: %s" % filename
    elif not ax_defined:
        plt.show()

# Fit function
def func(x, b, c, d):
    return (b/(1.0+np.exp(-(c+d*x))))

# First derivative
def f_der1(x, popt):
    b,c,d = popt[:]
    e = 2+ np.exp(-(c+d*x)) + np.exp(c+d*x)
    return b*d/e

# Second derivative
def f_der2(x, popt):
    b,c,d = popt[:]
    t1 = b*d/np.power((2+ np.exp(-(c+d*x)) + np.exp(c+d*x)),2)
    t2 = (d*np.exp(-(c+d*x)) - d*np.exp(c+d*x))
    return t1*t2
# Fit the curve for the data
def getFitCurve(x1,y):
  return optimize.curve_fit(func,x1,y)

def genFitData(filename, normalized=False):
    x,x1,y,y_org = getData(filename, normalized)
    #print x,x1,y,y_org 
    #Interpolate using spline
    s = interpolate.PchipInterpolator(x,y)
    r = s(x)
    
    #Fit curve to logistic regression and get coefficients
    popt,pcov = getFitCurve(x1,y)
    print popt, pcov
    #print "Coefficients: %s" % popt
    x2 = np.arange(min(x1),max(x1)+1, 0.2)
    #y2 = [func(i,popt[0],popt[1],popt[2],popt[3]) for i in x2]
    y2 = [func(i,popt[0],popt[1],popt[2]) for i in x2]
    y3 = [f_der1(i,popt) for i in x2]
    y4 = [f_der2(i,popt) for i in x2]
    x2 = [min(x)+k for k in x2]
    return x,y,x2,r,y2,y3,y4,y_org


def main(filename, normalized=False, saveFigure=False):
    x,y,x2,r,y2,y3,y4,y_org = genFitData(filename, normalized=normalized)
    dir_name = os.path.dirname(filename)
    base_name = os.path.basename(filename)
    filename = os.path.join(dir_name, base_name.split('.')[0])
    plot(x,y,x2,r,y2,y3,y4,y_org, saveFigure = saveFigure, filename = filename)





"""
yp = []
for i in range(3,8):
    z = np.polyfit(x1,y,i)
    f = np.poly1d(z)
    pl.plot(x,f(x1),"-",linewidth=2,label="{0}-D Poly Fit curve".format(i))
"""

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Create Plots for Mesh Term\
                                     Data")
    parser.add_argument("-f",help="Name of file to get data from. Should\
                        contain\
                        2 columns only seperated by tab. Year and Count")
    parser.add_argument("-n", type=bool, help="Normalize counts. Set to activate")
    parser.add_argument("-s", type=bool, help="Save figure and not show. Set to activate")
    parser.add_argument("-t", type=bool, help="Running from terminal. Set to activate")
    args = parser.parse_args()
    filename = args.f
    normalized = args.n
    saveFigure  = args.s
    print "Normalized:", normalized
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

    main(filename, normalized, saveFigure)

