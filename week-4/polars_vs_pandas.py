import pandas as pd
import polars as pl
import time

# Download a sample dataset (NYC Taxi Trips small sample)

#csv_file = "yellow_tripdata_2019-01.csv"

# ------------------ PANDAS ------------------
start = time.time()


df_pd = pd.read_csv("E:/CLASS ACTIVITY AND ASSIGNMENTS/tlhird sem/Software tools/software-tools-lab-work/week-4/yellow_tripdata_2019-01.csv")
# Example operation: filter and compute average fare
result_pd = df_pd[df_pd['passenger_count'] > 2]['total_amount'].mean()
end = time.time()
print("Pandas result:", result_pd)
print("Pandas execution time:", end - start, "seconds")

# ------------------ POLARS ------------------
start = time.time()
df_pl = pl.read_csv("E:/CLASS ACTIVITY AND ASSIGNMENTS/tlhird sem/Software tools/software-tools-lab-work/week-4/yellow_tripdata_2019-01.csv", n_rows=100000)
# Example operation: filter and compute average fare
result_pl = df_pl.filter(pl.col('passenger_count') > 2).select(pl.col('total_amount').mean())
end = time.time()
print("Polars result:", result_pl)
print("Polars execution time:", end - start, "seconds")
