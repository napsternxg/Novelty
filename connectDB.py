from settings import *
from sqlalchemy import create_engine 
from sqlalchemy.orm import sessionmaker 

def getDBConnectorString(id=None):
  if id is None:
    mysql_settings = SETTINGS["MYSQL_SETTINGS"]
  elif id == 1:
    mysql_settings = SETTINGS["MYSQL_SETTINGS2"]
  elif id == 2:
    mysql_settings = SETTINGS["MYSQL_SETTINGS3"]
  return "mysql+pymysql://%s:%s@%s/%s" % (mysql_settings["USER"], mysql_settings["PASSWORD"],\
                              mysql_settings["HOST"], mysql_settings["DB_NAME"])

#print getDBConnectorString()
#print getDBConnectorString(1)
engine = create_engine(getDBConnectorString())
Session = sessionmaker(bind=engine)
session = Session()
#connection to 2nd DB
engine1 = create_engine(getDBConnectorString(1))
Session1 = sessionmaker(bind=engine1)
session1 = Session1()
