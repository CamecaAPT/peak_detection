import json
from pathlib import Path

DATA_FILE = Path(__file__).parent.parent / "data" / "periodic_table.json"

with open(DATA_FILE, encoding="utf8") as f:
    PERIODIC_TABLE = json.load(f)

''' 
Example json format:

"Si": {
    "atomic_number": 14,
    "average_mass": 28.08538362730701,
    "isotopes": {
      "28": {
        "abundance": 0.9222968,
        "mass": 27.97692649
      },
      "29": {
        "abundance": 0.0468316,
        "mass": 28.97649468
      },
      "30": {
        "abundance": 0.0308716,
        "mass": 29.97377018
      }
    },
    "monoisotopic_mass": 27.97692649,
    "most_abundant_isotope": 28,
    "name": "Silicon",
    "symbol": "Si"
  },

parse_formula()
generate_isotope_fingerprint()
simulate_isotope_pattern()
'''

def is_valid_element(symbol: str) -> bool:
    if not symbol:
        return False
    return symbol.strip().capitalize() in PERIODIC_TABLE

def get_most_abundant_isotope(symbol: str) -> int:
    symbol = symbol.strip().capitalize()
    if not is_valid_element(symbol):
        raise ValueError(f"Unknown element: {symbol}")
    return PERIODIC_TABLE[symbol]["most_abundant_isotope"]

def get_isotope_fingerprint(symbol: str, mass_number: int) -> dict:
    symbol = symbol.strip().capitalize()
    if not is_valid_element(symbol):
        raise ValueError(f"Unknown element: {symbol}")
    return PERIODIC_TABLE[symbol]["isotopes"][str(mass_number)]

def get_average_mass(symbol: str) -> float:
    symbol = symbol.strip().capitalize()
    if not is_valid_element(symbol):
        raise ValueError(f"Unknown element: {symbol}")
    return PERIODIC_TABLE[symbol]["average_mass"]

def get_monoisotopic_mass(symbol: str) -> float:
    symbol = symbol.strip().capitalize()
    if not is_valid_element(symbol):
        raise ValueError(f"Unknown element: {symbol}")
    return PERIODIC_TABLE[symbol]["monoisotopic_mass"]

def get_name(symbol: str) -> str:
    symbol = symbol.strip().capitalize()
    if not is_valid_element(symbol):
        raise ValueError(f"Unknown element: {symbol}")
    return PERIODIC_TABLE[symbol]["name"]

def get_atomic_number(symbol: str) -> int:
    symbol = symbol.strip().capitalize()
    if not is_valid_element(symbol):
        raise ValueError(f"Unknown element: {symbol}")
    return PERIODIC_TABLE[symbol]["atomic_number"]
    