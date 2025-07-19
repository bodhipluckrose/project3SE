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
