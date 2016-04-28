# Total PMIDs per year

```sql
mysql -u sofia -p PUBMED2015 -h sofus -e "SELECT year, COUNT(PMID) as TotalPMID FROM Articles WHERE mesh IS NOT NULL AND mesh != '-' GROUP BY year ORDER BY year;" > out/PMID_PER_YEAR.tsv
```

To run the code using spark do the following:
 * Delete all folders in the `out/` directory
 * Open spark enabled ipython shell
 * Run the code using `%run -i mesh_per_year_ex.py`


# Load data into DB
```
db/load_data.sh <FOLDER_NAME> <tbl_name>
# Example
db/load_data.sh out/pmid_novelty_all_scores_mesh_c novelty_scores
```

## Author Data

```sql
mysql -u sofia -p PUBMED2010 -h sofus -e "SELECT au_id, au_ids FROM au_clst_all" > data/Authors.txt
```

## Citation Data

```sql
mysql -u sofia -p citation -h sofus -e "SELECT * FROM cite_list" > data/Citelist.txt
```

```sql
mysql -u sofia -p citation -h sofus -e "SELECT b.PMID, b.year, b.journal, a.Ncitedby FROM cite_list as a JOIN PUBMED2015.Articles as b ON a.PMID = b.PMID" > data/pmid_yr_journal_ncitedby.txt
```

## Data for Bruce

```sql
mysql -u sofia -p novelty -h sofus -e "SELECT PMID, Year, TFirstP as TimeNovelty, VolFirstP as VolumeNovelty, Pair_TFirstP as PairTimeNovelty, Pair_VolFirstP as PairVolumeNovelty FROM novelty_scores" > out/PubMed2015_NoveltyData.txt
```
