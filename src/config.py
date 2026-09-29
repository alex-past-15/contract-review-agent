from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
REPORTS = ROOT / "reports"
CUAD_JSON = DATA / "CUADv1.json"
WINDOWS_PATH = DATA / "windows.parquet"

# берём 12 категорий, которые важны бизнесу и по которым в CUAD хватает примеров
CATEGORIES = [
    "Termination For Convenience",
    "Cap On Liability",
    "Uncapped Liability",
    "Liquidated Damages",
    "Renewal Term",
    "Notice Period To Terminate Renewal",
    "Non-Compete",
    "Exclusivity",
    "Change Of Control",
    "Anti-Assignment",
    "Ip Ownership Assignment",
    "Audit Rights",
]

WINDOW_CHARS = 1500
STRIDE_CHARS = 750
SEED = 42
