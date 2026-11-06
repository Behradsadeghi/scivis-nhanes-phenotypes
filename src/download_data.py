"""
Step 0 - Download the 20 public NHANES 2017-2018 files used in this project from the CDC.
Files are saved, unchanged, to data/raw/.  (~27 MB)
"""
from pathlib import Path
from urllib.request import urlretrieve

FILES = ["DEMO", "BMX", "BPX", "GHB", "GLU", "INS", "TRIGLY", "HDL", "HSCRP", "BIOPRO",
         "ALB_CR", "CBC", "COT", "DIQ", "BPQ", "MCQ", "PAQ", "HIQ", "HUQ", "FSQ"]
BASE = "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/{}_J.xpt"
RAW = Path(__file__).resolve().parents[1] / "data" / "raw"
RAW.mkdir(parents=True, exist_ok=True)

for f in FILES:
    out = RAW / f"{f}_J.xpt"
    if out.exists():
        print("have", out.name)
        continue
    print("get ", out.name)
    urlretrieve(BASE.format(f), out)
