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
