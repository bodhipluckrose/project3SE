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
