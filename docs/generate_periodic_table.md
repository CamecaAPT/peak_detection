# scripts/generate_periodic_table.py

One-time/occasional generator: parses `data/mass.txt` (element/isotope mass + natural-abundance table) into `data/periodic_table.json`, the element/isotope database consumed by `peak_detection/utils/periodic_table.py` and `scripts/generate_synthetic_data.py`. Takes no CLI arguments.

## Usage

```powershell
.venv\Scripts\python.exe scripts\generate_periodic_table.py
```

## Arguments

None. Paths are hardcoded relative to the repo root: reads `data/mass.txt`, writes `data/periodic_table.json`.

## Notes

- Re-run only when `data/mass.txt` changes (e.g. an updated isotope-abundance source table).
- Normalizes the historical "Lw" (old Lawrencium symbol) to "Lr" in both element and isotope symbols.
- For each element, computes `average_mass` (abundance-weighted), `most_abundant_isotope`, and `monoisotopic_mass` from the parsed isotopes.
