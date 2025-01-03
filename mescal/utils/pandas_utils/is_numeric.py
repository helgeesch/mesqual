import numpy as np
import pandas as pd


def pd_is_numeric(df: pd.DataFrame | pd.Series) -> bool:
    if isinstance(df, pd.Series):
        return df.map(np.isreal).all()
    return df.map(np.isreal).all().all()
