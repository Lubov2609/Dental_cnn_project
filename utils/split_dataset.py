import pandas as pd
from sklearn.model_selection import train_test_split
import os

DATA_PATH = "data/label.csv"

df = pd.read_csv(DATA_PATH)

train, temp = train_test_split(
    df,
    test_size=0.3,
    random_state=42
)

val, test = train_test_split(
    temp,
    test_size=0.5,
    random_state=42
)

os.makedirs("data/splits", exist_ok=True)

train.to_csv("data/splits/train.csv", index=False)
val.to_csv("data/splits/val.csv", index=False)
test.to_csv("data/splits/test.csv", index=False)

print("Dataset split completed")