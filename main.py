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


# --- Layout of the App ---
app.layout = html.Div(style={'maxWidth': '900px', 'margin': 'auto', 'padding': '20px'}, children=[
    html.H1("Expense Tracker & Predictor", style={'textAlign': 'center', 'marginBottom': '30px'}),

    # Add New Expense Section
    html.Div(style=card_style, children=[
        html.Div(html.H3("Add New Expense"), style=header_style),
        html.Div(style={'padding': '15px'}, children=[
            html.Div(style=input_row_style, children=[
                html.Div(html.Label("Date (YYYY-MM-DD):"), style=label_col_style),
                html.Div(style=input_col_style, children=[
                    dcc.DatePickerSingle(
                        id='expense-date-picker',
                        min_date_allowed=date(2020, 1, 1),
                        max_date_allowed=date(2030, 12, 31),
                        initial_visible_month=date.today(),
                        date=str(date.today()),
                        display_format='YYYY-MM-DD',
                        style={'width': '100%'}
                    )
                ]),
            ]),
            html.Div(style=input_row_style, children=[
                html.Div(html.Label("Category:"), style=label_col_style),
                html.Div(style=input_col_style, children=[
                    dcc.Dropdown(
                        id='expense-category-dropdown',
                        options=[
                            {'label': 'Food', 'value': 'Food'},
                            {'label': 'Transport', 'value': 'Transport'},
                            {'label': 'Rent', 'value': 'Rent'},
                            {'label': 'Utilities', 'value': 'Utilities'},
                            {'label': 'Entertainment', 'value': 'Entertainment'},
                            {'label': 'Education', 'value': 'Education'},
                            {'label': 'Other', 'value': 'Other'}
                        ],
                        value='Food',
                        clearable=False,
                        style={'width': '100%'}
                    )
                ]),
            ]),
            html.Div(style=input_row_style, children=[
                html.Div(html.Label("Amount:"), style=label_col_style),
                html.Div(style=input_col_style, children=[
                    dcc.Input(id='expense-amount-input', type='number', value=0.0, min=0.0, step=0.01,
                              style={'width': '100%', 'padding': '8px'})
                ]),
            ]),
            html.Button("Add Expense", id='add-expense-button', style=button_style),
            html.Div(id='add-expense-output', style={'marginTop': '10px'})
        ])
    ]),

    # Expenses Table Section
    html.Div(style=card_style, children=[
        html.Div(html.H3("Your Expenses"), style=header_style),
        html.Div(style={'padding': '15px'}, children=[
            dash.dash_table.DataTable(
                id='expenses-table',
                columns=[{"name": i, "id": i} for i in ["Date", "Category", "Amount"]],
                # Data initialized as empty, will be populated on adding expenses
                data=data_manager.get_data_frame().to_dict('records'),
                sort_action="native",
                filter_action="native",
                page_size=10,
                style_table={'overflowX': 'auto', 'border': '1px solid #eee'},
                style_cell={'padding': '8px', 'textAlign': 'left'},
                style_header={'backgroundColor': '#f2f2f2', 'fontWeight': 'bold'}
            )
        ])
    ]),

    # Expense Predictions Section
    html.Div(style=card_style, children=[
        html.Div(html.H3("Expense Predictions"), style=header_style),
        html.Div(style={'padding': '15px'}, children=[
            html.Div(style=input_row_style, children=[
                html.Div(html.Label("Prediction Method:"), style=label_col_style),
                html.Div(style=input_col_style, children=[
                    dcc.Dropdown(
                        id='prediction-method-dropdown',
                        options=[
                            {'label': 'Moving Average', 'value': 'moving_average'},
                            {'label': 'Linear Regression', 'value': 'linear_regression'}
                        ],
                        value='moving_average',
                        clearable=False,
                        style={'width': '100%'}
                    )
                ]),
            ]),
            html.Div(style=input_row_style, children=[
                html.Div(html.Label("Start Date:"), style=label_col_style),
                html.Div(style=input_col_style, children=[
                    dcc.DatePickerSingle(
                        id='prediction-start-date-picker',
                        min_date_allowed=date(2020, 1, 1),
                        max_date_allowed=date(2030, 12, 31),
                        initial_visible_month=date.today(),
                        date=str(date.today().replace(day=1)),
                        display_format='YYYY-MM-DD',
                        style={'width': '100%'}
                    )
                ]),
            ]),
            html.Div(style=input_row_style, children=[
                html.Div(html.Label("End Date:"), style=label_col_style),
                html.Div(style=input_col_style, children=[
                    dcc.DatePickerSingle(
                        id='prediction-end-date-picker',
                        min_date_allowed=date(2020, 1, 1),
                        max_date_allowed=date(2030, 12, 31),
                        initial_visible_month=date.today(),
                        date=str(date.today()),
                        display_format='YYYY-MM-DD',
                        style={'width': '100%'}
                    )
                ]),
            ]),
            html.Button("Generate Prediction", id='generate-prediction-button', style=button_style),
            html.Div(id='prediction-output', style={'marginTop': '10px'}),
            dcc.Graph(id='prediction-graph', style={'marginTop': '20px'})
        ])
    ])
])

# --- Callbacks (Interactive Logic) ---

@app.callback(
    Output('add-expense-output', 'children'),
    Output('expenses-table', 'data'),
    Input('add-expense-button', 'n_clicks'),
    State('expense-date-picker', 'date'),
    State('expense-category-dropdown', 'value'),
    State('expense-amount-input', 'value'),
    prevent_initial_call=True
)
def add_expense_callback(n_clicks, date_str, category, amount):
    """Callback to add a new expense."""
    if n_clicks is None:
        raise dash.exceptions.PreventUpdate

    # Input validation
    try:
        if date_str is None:
            raise ValueError("Date cannot be empty.")
        datetime.strptime(date_str, '%Y-%m-%d').date() 
    except (ValueError, TypeError) as e:
        return html.Div(f"Invalid input: Date must be in YYYY-MM-DD format. Error: {e}", style=alert_danger_style), data_manager.get_data_frame().to_dict('records')

    valid_categories = ["Food", "Transport", "Rent", "Utilities", "Entertainment", "Education", "Other"]
    if category not in valid_categories:
        return html.Div(f"Invalid input: Category must be one of {valid_categories}.", style=alert_danger_style), data_manager.get_data_frame().to_dict('records')

    if not isinstance(amount, (int, float)) or amount <= 0:
        return html.Div("Invalid input: Amount must be a positive number.", style=alert_danger_style), data_manager.get_data_frame().to_dict('records')

    # Add the expense
    expense_data_obj = ExpenseData(date_str, category, amount)
    data_manager.add_expense(expense_data_obj)

    # Update the table data displayed in the UI
    updated_table_data = data_manager.get_data_frame().to_dict('records')
    return html.Div("Expense added successfully!", style=alert_success_style), updated_table_data

@app.callback(
    Output('prediction-output', 'children'),
    Output('prediction-graph', 'figure'),
    Input('generate-prediction-button', 'n_clicks'),
    State('prediction-method-dropdown', 'value'),
    State('prediction-start-date-picker', 'date'),
    State('prediction-end-date-picker', 'date'),
    prevent_initial_call=True
)

def generate_prediction_callback(n_clicks, prediction_method, start_date_str, end_date_str):
    """Callback to generate and display expense predictions."""
    if n_clicks is None:
        raise dash.exceptions.PreventUpdate

    dataframe = data_manager.get_data_frame()

    # Validate and parse date range
    try:
        if start_date_str is None or end_date_str is None:
            raise ValueError("Start and end dates cannot be empty.")
        start_date = datetime.strptime(start_date_str, '%Y-%m-%d').date()
        end_date = datetime.strptime(end_date_str, '%Y-%m-%d').date()
    except (ValueError, TypeError) as e:
        return html.Div(f"Invalid date format for start or end date. Use YYYY-MM-DD. Error: {e}", style=alert_danger_style), {}

    # Filter data by date range
    filtered_dataframe = pd.DataFrame()
    if 'Date' in dataframe.columns and not dataframe.empty:
        filtered_dataframe = dataframe[(dataframe['Date'].dt.date >= start_date) & (dataframe['Date'].dt.date <= end_date)].copy()
        filtered_dataframe = filtered_dataframe.sort_values(by='Date').reset_index(drop=True)

    if filtered_dataframe.empty:
        return html.Div("No data for selected date range to generate predictions.", style=alert_info_style), {}

    predictions_series = pd.Series()
    alert_message_text = ""

    # Generate predictions based on selected method
    try:
        if prediction_method == "moving_average":
            predictions_series = PredictionManager.predict_moving_average(filtered_dataframe)
            alert_message_text = "Moving average predictions generated."
        elif prediction_method == "linear_regression":
            predictions_series = PredictionManager.predict_linear_regression(filtered_dataframe)
            alert_message_text = "Linear regression predictions generated."
        else:
            return html.Div("Invalid prediction method. Choose 'moving_average' or 'linear_regression'.", style=alert_danger_style), {}
    except ValueError as e:
        return html.Div(f"Prediction Error: {e}", style=alert_danger_style), {}

    # Create Plotly figure for visualization
    fig = px.line(filtered_dataframe, x='Date', y='Amount', title='Expenses and Predictions')
    fig.update_traces(mode='lines+markers', name='Actual Expenses')

    # Add predictions as a separate trace if available
    predictions_to_plot = predictions_series.dropna()
    dates_for_predictions = filtered_dataframe.loc[predictions_to_plot.index, 'Date']

    if not predictions_to_plot.empty and len(predictions_to_plot) > 0:
        fig.add_scatter(x=dates_for_predictions, y=predictions_to_plot, mode='lines', name='Predicted Expenses',
                        line=dict(dash='dash', color='red'))
    else:
        alert_message_text += " (Note: No valid predictions generated for visualization, possibly due to insufficient data for the chosen method.)"

    fig.update_layout(hovermode="x unified")

    return html.Div(alert_message_text, style=alert_success_style), fig

# --- Run the App ---
if __name__ == '__main__':
    app.run(debug=True)
