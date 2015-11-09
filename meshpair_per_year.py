#!/usr/bin/env python

import csv
import re
"""
Create dataset for count of mesh pair term per year.
"""

exploded_hash = {} # Maintain Hash of exploded terms for each meshterm to save computation time.
mesh_index = []
with open("out/exploded_mhash.tsv") as fp:
  for line in fp.readlines():
    data = line.replace("\n","").split("\t")
    exploded_hash[data[0]] = data[1].split("|")
    mesh_index.append(data[0])

mesh_index.sort(key=str.lower)

mesh_count = {}

def getIndex(mStr):
  global mesh_index
  try:
    return mesh_index.index(mStr)
  except ValueError:
    return -1

with open("data/pmid_year_mesh.tsv") as fp:
#with open("temp.tsv") as fp:
  # pmid_year_mesh.tsv: PMID    year    mesh
  #datareader = csv.DictReader(fp, delimiter = "\t")
  datareader = csv.reader(fp, delimiter = "\t")
  col_i = {k: i for i, k in enumerate(datareader.next())}
  for row in datareader:
    mesh_items = row[2].split("|")
    mesh_items = filter(lambda x: x!="-", mesh_items)
    if len(mesh_items) == 0:
      continue
    year = int(row[1])
    print mesh_items, year
    mesh_items.sort(key=str.lower)
#    print mesh_items
    for i in range(len(mesh_items)):
      item = mesh_items[i]
      if item not in exploded_hash:
        continue
      #all_mesh.add(item)
      for item1 in mesh_items[i+1:]:
        if item1 not in exploded_hash:
          continue
        if item1 is item:
          continue
#        if item1 in mesh_count and item in mesh_count[item1]:
#          item1,item = item,item1
        tKey = (getIndex(item),getIndex(item1))
        if tKey not in mesh_count:
          mesh_count[tKey] = {}
        if year not in mesh_count[tKey]:
          mesh_count[tKey][year] = 0
        mesh_count[tKey][year] += 1
#        print "{0}\t{1}\t{2}".format(item, item1, year)
# routine for exploded counts
  print "Done Reading all Data"


with open("out/meshpair_per_year.tsv", "w+") as fp:
  for tKey in mesh_count:
    for year in mesh_count[tKey]:
      total = mesh_count[tKey][year]
      mesh1 = mesh_index[tKey[0]]
      mesh2 = mesh_index[tKey[1]]
      if total > 0:
        fp.write("{0}\t{1}\t{2}\t{3}\n".\
            format(mesh1,mesh2,year,total))
  print "Done Non Exploded Data writing to file"


# Now work on exploded data
ex_mesh_count = {}
with open("out/meshpair_per_year.tsv") as fp:
  datareader = csv.reader(fp,delimiter="\t" )
  for row in datareader:

# Better routine for Exploded data. Adds all counts in specific year in batch to exploded mesh terms
    k = row[:2]
    mesh1 = k[0]
    mesh2 = k[1]
    y = row[2]
    count = int(row[3])
    if mesh1 not in exploded_hash:
      continue
    mesh_items = exploded_hash[mesh1]
    # print "exploded_hash[{0}]={1}".format(k,mesh_items)

    if mesh2 not in exploded_hash:
      continue
    mesh_items1 = exploded_hash[mesh2]
    # print "exploded_hash[{0}]={1}".format(k2,mesh_items1)

    #print "mesh_count[{0}][{1}][{2}]={3};Exploded={4}".format(mesh1,mesh2,y,count,'N')
    for item1 in mesh_items:
      temp = item1
      for item2 in mesh_items1:
        if item1 == item2:
          continue
        if item1.lower() > item2.lower():
          item1,item2 = item2,item1
        ex_key = "%s\t%s\t%s" %(item1,item2,y)
        """
        if mesh1 not in ex_mesh_count:
          ex_mesh_count[mesh1] = {}
        if mesh2 not in ex_mesh_count[mesh1]:
          ex_mesh_count[mesh1][mesh2] = {}
        """
        if ex_key not in ex_mesh_count:
          ex_mesh_count[ex_key] = 0
        ex_mesh_count[ex_key] += count
          # print "[{0}][{1}][{2}]->[{3}][{4}][{5}]={6}".format(k,k2,y,item1,item2,y,ex_mesh_count[item1][item2][y])
        item1 = temp
        #print item1,item2
    #print "mesh_count[{0}][{1}]".format(mesh1,mesh2)
  print "Done Exploded counts"

with open("out/meshpair_per_year_exploded.tsv", "wb+") as fp:
  for ex_key in ex_mesh_count:
    total = ex_mesh_count[ex_key]
    ex_key = ex_key.split("\t")
    mesh1 = ex_key[0]
    mesh2 = ex_key[1]
    year = ex_key[2]
    if total > 0:
      fp.write("{0}\t{1}\t{2}\t{3}\n".\
          format(mesh1,mesh2,year,total))
  print "Done Exploded Data writing to file"

print len(mesh_count.keys())
print len(ex_mesh_count.keys())
