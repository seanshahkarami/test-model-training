#!/usr/bin/env python3
import argparse
import re
from pathlib import Path
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("filename")
args = parser.parse_args()

data = Path(args.filename).read_text()
df = pd.DataFrame(
    {"model": r[0], "item": int(r[1]), "duration": float(r[2])}
    for r in re.findall("result (\S+) cuda (\d+) (\S+)", data)
)

# drop first item to ignore any warm up behavior
df = df[df.item > 0]

print(df.groupby("model").duration.describe().sort_values("mean"))
