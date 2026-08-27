# Separate script to create periodic_table.json from mass.txt
# Run once to generate the element/isotope database.

from pathlib import Path
import json
import re

ROOT = Path(__file__).parent.parent

MASS_FILE = ROOT / "data" / "mass.txt"
OUTPUT_FILE = ROOT / "data" / "periodic_table.json"

# Matches:
# H(1) 1.0078250319 99.984426
# Fe(56) 55.9349418 91.754
ELEMENT_PATTERN = re.compile(
    r"([A-Z][a-z]?)\((\d+)\)\s+([\d.]+)\s+([\d.]+)"
)

# Periodic table atomic numbers
ATOMIC_NUMBERS = {
    "H": 1, "He": 2,
    "Li": 3, "Be": 4, "B": 5, "C": 6, "N": 7, "O": 8, "F": 9, "Ne": 10,
    "Na": 11, "Mg": 12, "Al": 13, "Si": 14, "P": 15, "S": 16, "Cl": 17, "Ar": 18,
    "K": 19, "Ca": 20, "Sc": 21, "Ti": 22, "V": 23, "Cr": 24, "Mn": 25,
    "Fe": 26, "Co": 27, "Ni": 28, "Cu": 29, "Zn": 30,
    "Ga": 31, "Ge": 32, "As": 33, "Se": 34, "Br": 35, "Kr": 36,
    "Rb": 37, "Sr": 38, "Y": 39, "Zr": 40, "Nb": 41, "Mo": 42,
    "Tc": 43, "Ru": 44, "Rh": 45, "Pd": 46, "Ag": 47, "Cd": 48,
    "In": 49, "Sn": 50, "Sb": 51, "Te": 52, "I": 53, "Xe": 54,
    "Cs": 55, "Ba": 56, "La": 57, "Ce": 58, "Pr": 59, "Nd": 60,
    "Pm": 61, "Sm": 62, "Eu": 63, "Gd": 64, "Tb": 65, "Dy": 66,
    "Ho": 67, "Er": 68, "Tm": 69, "Yb": 70, "Lu": 71,
    "Hf": 72, "Ta": 73, "W": 74, "Re": 75, "Os": 76, "Ir": 77,
    "Pt": 78, "Au": 79, "Hg": 80, "Tl": 81, "Pb": 82, "Bi": 83,
    "Po": 84, "At": 85, "Rn": 86, "Fr": 87, "Ra": 88, "Ac": 89,
    "Th": 90, "Pa": 91, "U": 92, "Np": 93, "Pu": 94, "Am": 95,
    "Cm": 96, "Bk": 97, "Cf": 98, "Es": 99, "Fm": 100, "Md": 101,
    "No": 102, "Lr": 103
}


def main():
    elements = {}
    current_name = None

    with open(MASS_FILE, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip()

            matches = ELEMENT_PATTERN.findall(line)

            if not matches:
                continue

            # Detect element name on the first line of an element block
            name_match = re.match(
                r"^\s*([A-Za-z]+(?:\s[A-Za-z]+)*)",
                line
            )

            if name_match:
                possible_name = name_match.group(1)

                if possible_name and "(" not in possible_name:
                    current_name = possible_name

            # Symbol from first isotope match
            symbol = matches[0][0]

            # Normalize old Lawrencium symbol used in mass.txt
            if symbol == "Lw":
                symbol = "Lr"

            if symbol not in elements:
                elements[symbol] = {
                    "name": current_name,
                    "symbol": symbol,
                    "atomic_number": ATOMIC_NUMBERS.get(symbol),
                    "average_mass": 0.0,
                    "monoisotopic_mass": None,
                    "most_abundant_isotope": None,
                    "isotopes": {}
                }

            # Add isotopes
            for iso_symbol, mass_number, mass, abundance in matches:

                if iso_symbol == "Lw":
                    iso_symbol = "Lr"

                elements[symbol]["isotopes"][mass_number] = {
                    "mass": float(mass),
                    "abundance": float(abundance) / 100.0
                }

    # Compute derived properties
    for element in elements.values():

        average_mass = 0.0
        most_abundant_isotope = None
        highest_abundance = -1.0

        for mass_number, isotope in element["isotopes"].items():

            abundance = isotope["abundance"]
            mass = isotope["mass"]

            average_mass += mass * abundance

            if abundance > highest_abundance:
                highest_abundance = abundance
                most_abundant_isotope = int(mass_number)

        element["average_mass"] = average_mass
        element["most_abundant_isotope"] = most_abundant_isotope

        if most_abundant_isotope is not None:
            element["monoisotopic_mass"] = (
                element["isotopes"][str(most_abundant_isotope)]["mass"]
            )

    # Write JSON
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(
            elements,
            f,
            indent=2,
            sort_keys=True
        )

    print(f"Generated {OUTPUT_FILE}")
    print(f"Elements: {len(elements)}")


if __name__ == "__main__":
    main()