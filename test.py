import sqlite3
import pandas as pd

conn = sqlite3.connect("/home/parasraina/ai-lab/manifests/manifests.db")

df_runs = pd.read_sql_query("SELECT * FROM runs LIMIT 5;", conn)
print(df_runs.columns)
print(df_runs.head())

df_experiments = pd.read_sql_query("SELECT * FROM experiments LIMIT 5;", conn)
print(df_experiments.columns)
print(df_experiments.head())
