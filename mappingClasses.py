from sqlalchemy import Column, Integer, String,Float,Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.dialects import mysql


Base = declarative_base()
class MeshScoreDB(Base):
  __tablename__ = 'mesh_scores'
  MeshTerm = Column(String(200), primary_key=True)
  Year = Column(mysql.YEAR(4), primary_key=True)
  AbsVal = Column(Integer)
  TFirstP = Column(Integer)
  VolFirstP = Column(Integer)
  PredVal = Column(Float)
  Velocity = Column(Float)
  Acceleration = Column(Float)

  def __init__(self):
    print "Mesh_Score Object created."
  None
  #
  #  def __init__(self, data_dict):
  #    for k in data_dict:
  #      self.__dict__[k] = data_dict[k]

  def __repr__(self):
    return "<MeshScore('%s', '%s', '%s')>" % (self.MeshTerm, self.Year)

  def showData(self):
    for k in self.__dict__:
      print "%s : %s" % (k, self.__dict__[k])

class MeshPairScoreDB(Base):
  __tablename__ = 'meshpair_scores'
  Mesh1 = Column(String(200), primary_key=True)
  Mesh2 = Column(String(200), primary_key=True)
  Year = Column(mysql.YEAR(4), primary_key=True)
  AbsVal = Column(Integer)
  TFirstP = Column(Integer)
  VolFirstP = Column(Integer)

  def __init__(self):
    print "Mesh_Pair_Score Object created."
  None
  #
  #  def __init__(self, data_dict):
  #    for k in data_dict:
  #      self.__dict__[k] = data_dict[k]

  def __repr__(self):
    return "<MeshPairScore('%s', '%s', '%s', '%s')>" % (self.Mesh1, self.Mesh2,self.Year)

  def showData(self):
    for k in self.__dict__:
      print "%s : %s" % (k, self.__dict__[k])

class ArticlesDB(Base):
  __tablename__ = 'Articles'
  PMID = Column(Integer, primary_key=True)
  title = Column(Text) 
  title_tokenized = Column(Text) 
  languages = Column(String(19))
  pub_types = Column(Text) 
  affiliation = Column(Text) 
  email = Column(Text) 
  mesh = Column(Text) 
  authors = Column(Text) 
  journal = Column(String(110))
  country = Column(String(40))
  issn = Column(String(12))
  volume = Column(String(60))
  issue = Column(String(40))
  pages = Column(String(100))
  year= Column(Integer)
  month = Column(String(15))
  grants = Column(Text) 
  cited = Column(Text)
  citedby = Column(Text)
  auids = Column(Text)
  def __init__(self):
    print "Articles Object created."
  None
  #
  #  def __init__(self, data_dict):
  #    for k in data_dict:
  #      self.__dict__[k] = data_dict[k]

  def __repr__(self):
    return "<Article('%s', '%s', '%s')>" % (self.PMID, self.mesh)

  def showData(self):
    for k in self.__dict__:
      print "%s : %s" % (k, self.__dict__[k])

