export SPARK_HOME="/content/smishra8/SOFTWARE/spark"
export MASTER=local[20]
#IPYTHON=1 "$SPARK_HOME/bin/pyspark"
MEMORY=10g
#PYSPARK_DRIVER_PYTHON="ipython" "$SPARK_HOME/bin/pyspark" --driver-memory "50g"\
PYSPARK_DRIVER_PYTHON="jupyter" PYSPARK_DRIVER_PYTHON_OPTS="notebook --no-browser" "$SPARK_HOME/bin/pyspark" --driver-memory "$MEMORY"\
  --name "Novelty PUBMED2015 Analysis" --conf spark.eventLog.enabled=false --conf spark.local.dir="./tmp" --conf spark.executor.memory="$MEMORY" --conf spark.driver.maxResultSize="5g"
