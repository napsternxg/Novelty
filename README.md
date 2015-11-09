# Total PMIDs per year

```sql
mysql -u sofia -p PUBMED2015 -h sofus -e "SELECT year, COUNT(PMID) as TotalPMID FROM Articles GROUP BY year ORDER BY year;" > out/PMID_PER_YEAR.tsv
```

To run the code using spark do the following:
 * Delete all folders in the `out/` directory
 * Open spark enabled ipython shell
 * Run the code using `%run -i mesh_per_year_ex.py`


# Load data into DB
```
mysql --local-infile=1 -u smishra8 -p novelty -h sofus -e "" > out/PMID_PER_YEAR.tsv
```
