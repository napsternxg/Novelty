export SPARK_HOME="/content/smishra8/SOFTWARE/spark"
export MASTER=local[20]
#IPYTHON=1 "$SPARK_HOME/bin/pyspark"

#PYSPARK_DRIVER_PYTHON="ipython" "$SPARK_HOME/bin/pyspark" --driver-memory "50g"\
PYSPARK_DRIVER_PYTHON="jupyter" PYSPARK_DRIVER_PYTHON_OPTS="notebook --no-browser" "$SPARK_HOME/bin/pyspark" --driver-memory "50g"\
  --name "Novelty PUBMED2015 Analysis" --conf spark.eventLog.enabled=false --conf spark.local.dir="./tmp" --conf spark.executor.memory="50g" --conf spark.driver.maxResultSize="5g"
