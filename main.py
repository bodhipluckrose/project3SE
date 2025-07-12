import pandas as pd 
from datetime import datetime 

class expense_data:
    def __init__(self,date,category,amount):
        self.date = date
        self.category = category
        self.amount = amount
    def dict(self):
        return {"date":self.date,"category":self.category,"amount":self.amount}
class DataManager:
    def __init__(self,filename):
        self.filename = filename
        self.expenes = self.load_expenses()
    def load_expenses(filename):
        df= pd.read_csv(filename)
        return [expense_data(row['date'],row['category'],row['amount'])]
    def add_expenses(expense)
        expense.append(expense)
    def self_expenses(self)
        if not self.expense:
            df = pd.DataFrame(columns:['date','category','amount']) 
        else:
            df = pd.DataFrame(dict for exp in expenses)
        df.to_csv(filename)
