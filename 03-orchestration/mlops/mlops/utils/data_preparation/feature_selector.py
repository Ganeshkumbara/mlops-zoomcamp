from typing import List, Optional

import pandas as pd

NUMERICAL_FEATURES = ['trip_distance']
CATEGORICAL_FEATURES = ['PU_DO']

CATEGORICAL2_FEATURES = ['PULocationID','DOLocationID']
#NUMERICAL2_FEATURES = ['trip_distance']

def select_features(df: pd.DataFrame, features: Optional[List[str]] = None,  method:int = 1 ) -> pd.DataFrame:
    if method == 1: 
        columns = CATEGORICAL_FEATURES + NUMERICAL_FEATURES
    else : 
        columns = CATEGORICAL2_FEATURES
    if features:
        columns += features

    return df[columns]
