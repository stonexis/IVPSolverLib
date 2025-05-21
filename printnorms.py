import pandas as pd

def read_norms_table(path: str) -> pd.DataFrame:
    
    df = pd.read_csv(path, dtype=str)
    
    return df



print(read_norms_table("norms_table.csv"))
