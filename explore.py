"""Step 1: look at what we actually got. Prints schema, ranges, and mask rates."""
import pandas as pd

train = pd.read_csv("data/Train.csv")
test = pd.read_csv("data/Test.csv")
sub = pd.read_csv("data/SampleSubmission.csv")

for name, df in [("Train", train), ("Test", test), ("SampleSubmission", sub)]:
    print(f"\n===== {name}: {df.shape} =====")
    print(df.dtypes)
    print(df.head(3).to_string())
    print("null fraction:")
    print((df.isna().mean().sort_values(ascending=False)).head(15))
