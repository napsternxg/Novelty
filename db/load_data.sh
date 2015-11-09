cd $1
LOAD_DIR=$(pwd)
DB_NAME="novelty"
USER_NAME="smishra8"
HOST_NAME="sofus"
TBL_NAME="$2"
LOAD_TMPL="LOAD DATA LOCAL INFILE '%s' INTO TABLE $TBL_NAME FIELDS TERMINATED BY '\\\\t';\n"
TMP_FILE="$TBL_NAME.sql"
echo "Loading data"
echo "From Dir: $LOAD_DIR"
echo "To Table: $DB_NAME.$TBL_NAME"
echo "Using: $USER_NAME@$HOST_NAME"
echo "Creating $TMP_FILE"
echo "-- Load data from all files in $LOAD_DIR to $DB_NAME.$TBL_NAME" > "$TMP_FILE"
for item in $(ls "$LOAD_DIR/part-"*)
do
  printf "$LOAD_TMPL" $item >> "$TMP_FILE"
done
mysql --local-infile=1 -u "$USER_NAME" -h "$HOST_NAME" -p "$DB_NAME" < "$TMP_FILE"
# rm "$TMP_FILE"
