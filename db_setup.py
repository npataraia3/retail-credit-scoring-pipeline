import pandas as pd
import sqlite3

url = "https://raw.githubusercontent.com/selva86/datasets/master/GermanCredit.csv"
df = pd.read_csv(url)

connection = sqlite3.connect("bank_data.db")

df.to_sql("applicants", connection, if_exists="replace", index=False)

connection.close()