import pandas as pd
import pyarrow.parquet as pq
from pathlib import Path

# List all Parquet files in a directory
parquet_files = Path("./data/").rglob("*.parquet")

# Read all Parquet files and concatenate them into a single DataFrame
dfs = [pd.read_parquet(file) for file in parquet_files]
print(dfs)
merged_df = pd.concat(dfs)

# Write the merged DataFrame to a new Parquet file
merged_df.to_parquet("merged_file.parquet")