import os
import argparse
import pandas as pd
from google.cloud import bigquery

print("Imports finished\n")

# get command line args
parser = argparse.ArgumentParser("Download MIMIC-III data from BigQuery")
parser.add_argument('--num_rows', type=int, help='number of rows to download; specify num_rows as -1 to download the entire dataset')
args = parser.parse_args()

num_rows = args.num_rows

if num_rows >= 2083180:
    print(f"NOTEEVENTS only has 2,083,180 rows. However, you specified {num_rows} rows, which is too many.")
    exit()

elif num_rows < -1:
    print(f"You specified a negative number of rows.")
    exit()

elif num_rows == -1:
    print(f"Downloading all rows:")
    num_rows = 2083180 
    
else:
    print(f"Downloading the first {num_rows} rows:")

# getting bigquery data
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'mimic-318214-95c6dd5fc624.json'
bigquery_client = bigquery.Client()
QUERY = f"""
SELECT * from `mimic-318214.mimiciii_notes.noteevents` LIMIT {num_rows}
"""
query_job = bigquery_client.query(QUERY)
df = query_job.to_dataframe()

print("Data received from Google Cloud\n")

df.to_csv("notes.csv")
