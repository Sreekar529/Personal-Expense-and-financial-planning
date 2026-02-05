import warnings
warnings.filterwarnings ("ignore")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import Isolationforest
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet
#LSTM
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
# FILES (update paths if needed)
path = ".\\"
TXN_CSV = "datasets\expense_transactions_data.csv"
MONTHLY_CSV = "datasets\monthly_category_expenses_data.csv"
# SETTINGS
FORECAST_MONTHS = 12     # forecast horizon (months)




LSTM_WINDOW = 6  
#how many past months to predict next month
LSTM_EPOCHS = 60
LSTM_BATCH = 16
TOP_CATEGORIES= 8 #focus on top categories for forecasting plots (reduce noise)
TEST_MONTHS = 6 #last N months kept for testing model performance
# SMALL HELPERS (few functions only)
def pick_col (df, candidates):
    """Pick the first existing column from candidates (case-insensitive)."""
    cols = {c.lower(): c for c in df.columns}
    for cand in candidates:
         if cand.lower() in cols:
            return cols[cand.lower()]
    return None
   

    



def clean_amount(s):
    """Convert amounts like '1,234', '500', '$12.3' to float safely, """


    s = s.astype(str)
    s = s.str.replace(",", "", regex=False)
    s = s.str.replace("₹", "", regex=False).str.replace("$", "", regex=False)
    s = s.str.replace("INR", "", regex=False).str.replace("Rs.", "", regex=False)
    s = s.str.strip()
    return pd.to_numeric(s, errors="coerce")


def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def mape(y_true, y_pred):
    y_true = np.array(y_true, dtype = float)
    y_pred = np.array(y_pred, dtype = float)
    return np.mean(np.abs((y_true - y_pred) / (y_true + 1e-9))) * 100


def create_lstm_sequences(arr_1d, window):
    X, y = [], []
    for i in range(len(arr_1d) - window):
        X.append(arr_1d[i:i + window])
        y.append(arr_1d[i + window])
    return np.array(X), np.array(y)



#-------------------------------------------------------------------------------------------------------------------------
#LOAD DATASETS
#-------------------------------------------------------------------------------------------------------------------------

txn =pd.read_csv(path + TXN_CSV)
monthly = pd.read_csv(path + MONTHLY_CSV)


txn.columns = [c.strip() for c in txn.columns]
monthly.columns = [c.strip() for c in monthly.columns]


print("Trancsaction dataset shape : ", txn.shape)
print("Monthly dataset shape : ", monthly.shape)


#-------------------------------------------------------------------------------------------------------------------------
# STANDARDIZE MONTHLY DATA (Date/Month, Category, Amount)
#-------------------------------------------------------------------------------------------------------------------------

# Try to detect monthly columns
month_col = pick_col(monthly, ["Month", "Date", "YearMonth", "MonthYear", "Period"])
cat_col = pick_col(monthly, ["Category", "ExpenseCategory", "Type"])
amt_col = pick_col(monthly, ["Amount", "Expense", "Total", "Total_Expense", "Spend", "Value"])



if month_col is None or cat_col is None or amt_col is None:
    raise ValueError(
        "Could not auto-detect required columns in mmonthly dataset."
        "Expected something like Month/Date , Category, Amount."

    )

