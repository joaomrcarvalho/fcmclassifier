import pandas as pd
from utils.utils import compute_and_print_metrics


def classify(nrc_csv):

    df_nrc = pd.read_csv(nrc_csv)
    print(df_nrc)
    print()

    y_true = df_nrc['Target']
    y_pred = df_nrc['Guess']
    
    compute_and_print_metrics(y_true, y_pred)
