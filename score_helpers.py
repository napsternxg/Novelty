import connectDB as cdb
import mappingClasses as mc
from sqlalchemy import or_,and_, func
from sqlalchemy.orm.exc import NoResultFound
import pandas as pd

def get_ex_mesh_list(x):
  if x == []:
    return []
  return list(set(reduce(lambda a, b: a+b, map(lambda k: mesh_exploded.value.get(k, (k,)), x))))


def get_mesh_counts(x):
  x = x.split("\t")
  mesh_list = x[2].split("|")
  mesh_counts = len(mesh_list)
  ex_mesh_list = get_ex_mesh_list(x[2])
  ex_mesh_counts = len(ex_mesh_list)
  return (int(x[0]), (mesh_counts, ex_mesh_counts))

def gen_scores(row):
  row = row.split('\t')
  """
  row = [PMID,YEAR,MESH]
  """
  mesh_terms = row[2].split("|") if row[2] != '-' else []
  # EXPLODED are not used to reduce computation because
  # a paper's novelty in its own MeSH is always going
  # to be lower than any of the parent MeSH as parents include the count of childs.
  #mesh_terms = get_ex_mesh_list(mesh_terms) # Get exploded list of MeSH terms
  if len(mesh_terms) < 1:
    print >> sys.stderr, "No mesh data for row data:", row
    return
  recs = None
  mdb = mc.MeshScoreDB
  try:
    recs = cdb.session.query(mdb.TFirstP,mdb.VolFirstP,mdb.AbsVal,mdb.Acceleration,mdb.Velocity).\
        filter(and_(\
        mdb.Year == row[1],\
        mdb.MeshTerm.in_(mesh_terms))).\
        all()
    if len(recs) < 1:
      raise NoResultFound
  except NoResultFound, e:
    print >> sys.stderr, e, "No result for data:", row
    return None
  df = pd.DataFrame(recs)
  df.columns = ["TFirstP","VolFirstP","AbsVal","Acceleration","Velocity"]
  r = {}
  r["min_TFirstP"] = df['TFirstP'].min()
  r["min_VolFirstP"] = df['VolFirstP'].min()
  r["min_AbsVal"] = df['AbsVal'].min()
  r["acc_pos_vel_min"] = df[df['Acceleration'] >= 0.0]['Velocity'].min()
  r["acc_pos_vel_max"] = df[df['Acceleration'] >= 0.0]['Velocity'].max()
  r["acc_neg_vel_max"] = df[df['Acceleration'] < 0.0]['Velocity'].max()
  r["acc_neg_vel_min"] = df[df['Acceleration'] < 0.0]['Velocity'].min()
  # Replace all dataframe nan values with mysql NULL \N value
  r = {k: ("\\N" if pd.isnull(v) else v) for k, v in r.iteritems()}
  return (row[0],row[1], # PMID, Year
      r['min_AbsVal'],r['min_TFirstP'],r['min_VolFirstP'],
      r['acc_pos_vel_min'],r['acc_pos_vel_max'],r['acc_neg_vel_max'],r['acc_neg_vel_min'])

  
def gen_pair_scores(row):
  row = row.split('\t')
  """
  row = [PMID,YEAR,MESH]
  """
  mesh_terms = row[2].split("|") if row[2] != '-' else []
  # EXPLODED are not used to reduce computation because
  # a paper's novelty in its own MeSH is always going
  # to be lower than any of the parent MeSH as parents include the count of childs.
  #mesh_terms = get_ex_mesh_list(mesh_terms) # Get exploded list of MeSH terms
  if len(mesh_terms) < 1:
    print >> sys.stderr, "No mesh data for row data:", row
    return
  recs = None
  mdb = mc.MeshPairScoreDB
  try:
    recs = cdb.session.query(func.min(mdb.TFirstP),func.min(mdb.VolFirstP),func.min(mdb.AbsVal)).\
        filter(and_(\
        mdb.Year == row[1],\
        mdb.Mesh1.in_(mesh_terms),\
        mdb.Mesh2.in_(mesh_terms))).\
        all()
    if len(recs) < 1:
      raise NoResultFound
  except NoResultFound, e:
    print >> sys.stderr, e, "No result for data:", row
    return None
  df = pd.DataFrame(recs)
  df.columns = ["TFirstP","VolFirstP","AbsVal"]
  r = {}
  r["min_TFirstP"] = df['TFirstP'].min()
  r["min_VolFirstP"] = df['VolFirstP'].min()
  r["min_AbsVal"] = df['AbsVal'].min()
  # Replace all dataframe nan values with mysql NULL \N value
  r = {k: ("\\N" if pd.isnull(v) else v) for k, v in r.iteritems()}
  return (row[0],row[1], # PMID, Year
      r['min_AbsVal'],r['min_TFirstP'],r['min_VolFirstP'])
