import os
os.environ["DISABLE_PANDERA_IMPORT_WARNING"] = "True"
import pandas as pd
import pandera.pandas as pa
from pandera import Column, DataFrameSchema, Check
from pathlib import Path

schema = DataFrameSchema({
    "age": Column(int, Check.in_range(18, 110), nullable=False),
    "sexe": Column(str, Check.isin(["F", "M"]), nullable=False),
    "hta": Column(int, Check.isin([0, 1]), nullable=False),
    "diabete": Column(int, Check.isin([0, 1]), nullable=False),
    "cardiopathie": Column(int, Check.isin([0, 1]), nullable=False),
    "temperature": Column(float, Check.in_range(34, 42), nullable=False),
    "spo2": Column(float, Check.in_range(80, 100), nullable=False),
    "freq_resp": Column(float, Check.in_range(10, 40), nullable=False),
    "infection_respi": Column(int, Check.isin([0, 1]), nullable=False),
    "vaccinated": Column(int, Check.isin([0, 1]), nullable=False),
    "severity_score": Column(int, Check.in_range(0, 10), nullable=False),
    "hospitalisation": Column(int, Check.isin([0, 1]), nullable=False),
})

if __name__ == "__main__":
    path = Path("data/processed/hospit_risk_dataset.csv")
    df = pd.read_csv(path)
    validated_df = schema.validate(df, lazy=True)
    print("Schéma validé avec Pandera (qualité des données).")
