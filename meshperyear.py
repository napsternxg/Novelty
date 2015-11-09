import csv
import re
"""
Create dataset for count of mesh term per year.
"""

mesh_names = {} # id to name match; find name for id
mesh_ids = {} # name to id match; find id for name
with open("data/mtrees2015.bin") as fp:
  for line in fp.readlines():
    data = line.replace("\n","").split(";")
    data[1] = data[1].replace(".","_")
    mesh_names[data[1]] = data[0]
    if data[0] not in mesh_ids:
      mesh_ids[data[0]] = []
    mesh_ids[data[0]].append(data[1])

mesh_count = {}
ex_mesh_count = {}
with open("data/pmid_year_mesh.tsv") as fp:
  datareader = csv.DictReader(fp, delimiter = "\t")
  print "Reading PubMed data"
  for row in datareader:
    mesh_items = row["mesh"].split("|")
    mesh_items = filter(lambda x: x!="-", mesh_items)
    if len(mesh_items) == 0:
      continue
    year = row["year"]
    print mesh_items, year
    for item in mesh_items:
      if item not in mesh_ids:
        continue
      if item not in mesh_count:
        mesh_count[item] = {}
      if year not in mesh_count[item]:
        mesh_count[item][year] = 0
      mesh_count[item][year] += 1
  # routine for exploded mesh term count
  """
      temp_mesh_items = []
      regex = []
      for item in mesh_items:
        if item in mesh_ids:
          for mid in mesh_ids[item]: # this leads to multiple counting of same meshterm as it will be counted for each id it has in mtree
            tags = mid.split("_")
            tags = ["_".join(tags[:t]) for t in range(1,len(tags)+1)]
            regex.extend(tags)
      regex = "|".join(regex)
      regex = re.compile(r"^("+regex+")$")
      for key in mesh_names:
        if regex.match(key) is not None:
          temp_mesh_items.append(mesh_names[key])
      mesh_items = temp_mesh_items
      for item in mesh_items:
        if item not in ex_mesh_count:
          ex_mesh_count[item] = {}
        if year not in ex_mesh_count[item]:
          ex_mesh_count[item][year] = 0
        ex_mesh_count[item][year] += 1
  """
  print "Done reading all data"
# Better routine for Exploded data. Adds all counts in specific year in batch to exploded mesh terms
for k in mesh_count:
  regex = []
  temp_mesh_items = []
  if k not in mesh_ids:
    continue
  for mid in mesh_ids[k]:
      tags = mid.split("_")
      tags = ["_".join(tags[:t]) for t in range(1,len(tags)+1)]
      regex.extend(tags)
  regex = "|".join(regex)
  regex = re.compile(r"^("+regex+")$")
  for key in mesh_names:
    if regex.match(key) is not None:
      temp_mesh_items.append(mesh_names[key]) # Should have used Set instead of list. Leads to double counting in exploded MeSH.
  temp_mesh_items = set(temp_mesh_items) # Fixed it here.
  for item in temp_mesh_items:
    if item not in mesh_ids:
      continue
    if item not in ex_mesh_count:
      ex_mesh_count[item] = {}
    for y in mesh_count[k]:
      if y not in ex_mesh_count[item]:
        ex_mesh_count[item][y] = 0
      ex_mesh_count[item][y] += mesh_count[k][y]

    


with open("out/mesh_per_year_correct.tsv", "w+") as fp:
  #Don't print Exploded MeSH Counts
  """
  for k,v in mesh_count.items():
    for y,tot in v.items():
      fp.write("{0}\t{1}\tN\t{2}\n".format(k,y,tot))
  """
  for k,v in ex_mesh_count.items():
    for y,tot in v.items():
      #fp.write("{0}\t{1}\tY\t{2}\n".format(k,y,tot))
      fp.write("{0}\t{1}\t{2}\n".format(k,y,tot))
  print "Done writing file"
