import dash
from dash import dcc, html, Input, Output, State
import pandas as pd
from datetime import datetime, date
import numpy as np
from sklearn.linear_model import LinearRegression
import plotly.express as px

# --- Core Logic Classes ---

class ExpenseData:
    """Represents a single expense record."""
    def __init__(self, date, category, amount):
        self.date = date
        self.category = category
        self.amount = amount

    def to_dict(self):
        """Converts the expense data to a dictionary for DataFrame compatibility."""
        return {"Date": str(self.date), "Category": self.category, "Amount": self.amount}

class DataManager:
    """Manages saving and accessing expense data.
    Implements the Singleton pattern to ensure a single instance.
    """
    _instance = None 

    def __new__(cls, filename="expenses.csv"):
        if cls._instance is None:
            cls._instance = super(DataManager, cls).__new__(cls)
            cls._instance.filename = filename
            cls._instance.expenses = []
        return cls._instance


    def add_expense(self, expense_obj):
        """Adds a new expense and saves to file."""
        self.expenses.append(expense_obj)
        self.save_expenses() # Save after adding

    def save_expenses(self):
        """Saves current expense data to the CSV file."""
        if not self.expenses:
            df = pd.DataFrame(columns=["Date", "Category", "Amount"])
        else:
            df = pd.DataFrame([exp.to_dict() for exp in self.expenses])
        df.to_csv(self.filename, index=False)

    def get_data_frame(self):
        """Returns the current expense data as a Pandas DataFrame."""
        if not self.expenses:
            return pd.DataFrame(columns=["Date", "Category", "Amount"])
        df = pd.DataFrame([exp.to_dict() for exp in self.expenses])
        # Convert Date column to datetime objects for plotting/filtering
        df['Date'] = pd.to_datetime(df['Date'])
        return df

class PredictionManager:
    """Handles different expense prediction methods."""
    @staticmethod
    def predict_moving_average(dataFrame, period=7):
        """Calculates the moving average of the 'Amount' column."""
        if 'Amount' not in dataFrame.columns:
            raise ValueError("DataFrame must contain an 'Amount' column for moving average calculation.")
        return dataFrame['Amount'].rolling(window=period, min_periods=1).mean()

    @staticmethod
    def predict_linear_regression(dataFrame):
        """Performs linear regression to predict 'Amount' values based on 'Date'."""
        if 'Date' not in dataFrame.columns or 'Amount' not in dataFrame.columns:
            raise ValueError("DataFrame must contain 'Date' and 'Amount' columns for linear regression.")

        dataFrame['NumericalDate'] = pd.to_datetime(dataFrame['Date']).apply(lambda x: x.toordinal())

        # Need at least 2 data points for linear regression
        if len(dataFrame) < 2:
            return pd.Series([np.nan] * len(dataFrame), index=dataFrame.index)

        X = dataFrame[['NumericalDate']]
        y = dataFrame['Amount']

        model = LinearRegression()
        model.fit(X, y)

        predicted_values = model.predict(X)
        return pd.Series(predicted_values, index=dataFrame.index)

# --- Initialize DataManager ---
data_manager = DataManager()

# --- Dash App Setup ---
app = dash.Dash(__name__)
app.title = "Expense Tracker & Predictor"

# --- Basic Inline Styles  ---
card_style = {
    'border': '1px solid #ddd',
    'borderRadius': '8px',
    'padding': '20px',
    'marginBottom': '20px',
    'boxShadow': '2px 2px 8px rgba(0,0,0,0.1)'
}

header_style = {
    'backgroundColor': '#f8f8f8',
    'padding': '10px 20px',
    'borderBottom': '1px solid #eee',
    'marginBottom': '15px',
    'borderTopLeftRadius': '8px',
    'borderTopRightRadius': '8px'
}

input_row_style = {
    'display': 'flex',
    'alignItems': 'center',
    'marginBottom': '10px'
}

label_col_style = {
    'flex': '1',
    'paddingRight': '10px',
    'fontWeight': 'bold'
}

input_col_style = {
    'flex': '2'
}

button_style = {
    'backgroundColor': '#007bff',
    'color': 'white',
    'padding': '10px 15px',
    'border': 'none',
    'borderRadius': '5px',
    'cursor': 'pointer',
    'marginTop': '15px'
}

alert_base_style = {
    'padding': '10px',
    'borderRadius': '5px',
    'marginTop': '10px',
    'textAlign': 'center'
}

alert_success_style = {**alert_base_style, 'backgroundColor': '#d4edda', 'color': '#155724', 'border': '1px solid #c3e6cb'}
alert_danger_style = {**alert_base_style, 'backgroundColor': '#f8d7da', 'color': '#721c24', 'border': '1px solid #f5c6cb'}
alert_info_style = {**alert_base_style, 'backgroundColor': '#d1ecf1', 'color': '#0c5460', 'border': '1px solid #bee5eb'}


