#!/usr/bin/env python3

"""
Si -> Al Exchange Maker

Reads a standard XYZ zeolite cluster, randomly selects Si atoms for
Si -> Al substitution, and searches for the configuration with the
lowest Coulomb-like repulsion between the Al atoms.

The Coulomb potential between identical charges is proportional to:

    sum(1 / r_ij)

The common charge-related factor is omitted because it is identical
for all configurations and therefore does not affect the ranking.

Input:
    *.xyz

The script asks for:
    1. Number of Si -> Al substitutions
    2. Number of random configurations to search

Output (optional):
    <input_name>_AlN_optimal.xyz
"""

import os
import sys
import math
import random
import itertools
import numpy as np
from pathlib import Path
from datetime import datetime
from time import perf_counter


# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------

# Finding xyz-file in current working directory
for file in os.listdir(os.getcwd()):
    if file.endswith(".xyz"):
        INPUT_XYZ_FILE = file
        break

# ANSI Escape Sequences
GREEN = "\033[32m"
RESET = "\033[0m"


# ----------------------------------------------------------------------
# Elapsed time
# ----------------------------------------------------------------------

def elapsed_time(duration):
    d = divmod(duration, 86400)
    h = divmod(d[1], 3600)
    m = divmod(h[1], 60)
    s = int(m[1])
    ms = int(1000*(m[1] - s))
    
    if d[0] == 0:
        if h[0] == 0:
            if m[0] == 0:
                if s == 0: return f'{ms} ms'
                else: return f'{s} s {ms} ms'
            else: return f'{int(m[0])} min {s} s {ms} ms'
        else: return f'{int(h[0])} hours {int(m[0])} min {s} s {ms} ms'
    else: return f'{int(d[0])} days {int(h[0])} hours {int(m[0])} min {s} s {ms} ms'


# ----------------------------------------------------------------------
# XYZ input/output
# ----------------------------------------------------------------------

def read_xyz(filename):
    """Read a standard XYZ file."""
    path = Path(filename)

    with path.open("r", encoding="utf-8") as f:
        lines = f.readlines()

    if len(lines) < 2:
        raise ValueError("The XYZ file is too short.")

    try:
        atom_count = int(lines[0].strip())
    except ValueError:
        raise ValueError("The first line of the XYZ file must be an integer atom count.")

    if len(lines) < atom_count + 2:
        raise ValueError(
            f"The XYZ file contains fewer than {atom_count} atom lines."
        )

    comment = lines[1].rstrip("\n")

    atoms = []

    for line_number, line in enumerate(lines[2:atom_count + 2], start=3):
        parts = line.split()

        if len(parts) < 4:
            raise ValueError(
                f"Invalid atom line {line_number}: expected element and x y z."
            )

        element = parts[0]

        try:
            x, y, z = map(float, parts[1:4])
        except ValueError:
            raise ValueError(
                f"Invalid coordinates on line {line_number}."
            )

        atoms.append((element, x, y, z))

    return comment, atoms


def write_xyz(filename, comment, atoms, al_indices):
    """Write the structure with selected Si atoms changed to Al."""
    with open(filename, "w", encoding="utf-8") as f:
        f.write(f"{len(atoms)}\n")
        f.write(f"{comment}\n")

        al_indices = set(al_indices)

        for i, (element, x, y, z) in enumerate(atoms):
            if i in al_indices:
                element = "Al"

            f.write(f"{element:2s} {x: .12f} {y: .12f} {z: .12f}\n")


# ----------------------------------------------------------------------
# Geometry and scoring
# ----------------------------------------------------------------------

def get_si_indices(atoms):
    """Return zero-based indices of Si atoms."""
    return [i for i, atom in enumerate(atoms) if atom[0].lower() == "si"]


def get_al_indices(atoms):
    """Return zero-based indices of existing Al atoms."""
    return [i for i, atom in enumerate(atoms) if atom[0].lower() == "al"]


def calculate_distance_matrix(atoms):
    """Calculate the complete pairwise distance matrix."""
    coordinates = np.array(
        [[x, y, z] for _, x, y, z in atoms],
        dtype=float,
    )

    differences = coordinates[:, None, :] - coordinates[None, :, :]
    return np.linalg.norm(differences, axis=2)


def calculate_periodic_distance_matrix(atoms, cell_lengths):
    """
    Calculate pairwise distances using the minimum-image convention
    for an orthorhombic periodic cell.
    """
    coordinates = np.array(
        [[x, y, z] for _, x, y, z in atoms],
        dtype=float,
    )

    cell = np.asarray(cell_lengths, dtype=float)

    differences = (
        coordinates[:, None, :]
        - coordinates[None, :, :]
    )

    differences -= cell * np.round(differences / cell)

    return np.linalg.norm(differences, axis=2)


def minimum_al_distance(al_indices, distance_matrix):
    """Return the minimum distance between all Al atoms."""
    if len(al_indices) < 2:
        return math.inf

    distances = distance_matrix[np.ix_(al_indices, al_indices)]
    pair_distances = distances[np.triu_indices(len(al_indices), k=1)]

    return float(np.min(pair_distances))


def calculate_cell_lengths(atoms):
    """
    Calculate periodic cell lengths from the minimum and maximum
    coordinates of all non-hydrogen atoms.

    Returns:
        (Lx, Ly, Lz) in Å.
    """
    coordinates = np.array(
        [
            [x, y, z]
            for element, x, y, z in atoms
            if element.lower() != "h"
        ],
        dtype=float,
    )

    if len(coordinates) == 0:
        raise ValueError("No non-hydrogen atoms found.")

    minimum = np.min(coordinates, axis=0)
    maximum = np.max(coordinates, axis=0)

    cell_lengths = maximum - minimum

    if np.any(cell_lengths <= 0):
        raise ValueError(
            "One or more calculated cell lengths are zero or negative."
        )

    return tuple(cell_lengths)


def score_configuration(al_indices, distance_matrix):
    """
    Calculate the Coulomb-like score.

    Lower is better:

        score = sum(1 / r_ij)

    for all pairs of substituted Al atoms.

    Distributes Al atoms on the outer surface.
    Periodic distances remove this surface bias.
    """
    if len(al_indices) < 2:
        return 0.0

    distances = distance_matrix[np.ix_(al_indices, al_indices)]

    # Upper triangle only: each Al-Al pair is counted once.
    pair_distances = distances[np.triu_indices(len(al_indices), k=1)]

    if np.any(pair_distances <= 0):
        return math.inf

    return float(np.sum(1.0 / pair_distances))


def score_configuration_Al_variance(al_indices, distance_matrix):
    """
    Calculate the variance of the Al-Al distances.

    Lower is better: configurations with more similar Al-Al
    distances have lower variance.

    Creates a ball of Al atoms in the center.
    """
    if len(al_indices) < 2:
        return 0.0

    distances = distance_matrix[np.ix_(al_indices, al_indices)]

    # Upper triangle only: each Al-Al pair is counted once.
    pair_distances = distances[np.triu_indices(len(al_indices), k=1)]

    return float(np.var(pair_distances))


def score_configuration_mixed(al_indices, distance_matrix, si_indices):
    """
    Calculate the variance of the Al-Si distances.

    Lower is better: configurations in which the distances from
    substituted Al atoms to the remaining Si atoms are more similar
    have lower variance.
    
    Coulomb score is added.
    """
    if len(al_indices) == 0 or len(si_indices) == len(al_indices):
        return 0.0

    # Si atoms that remain Si after the substitutions.
    remaining_si = [i for i in si_indices if i not in al_indices]

    # Distances between every substituted Al and every remaining Si.
    al_si_distances = distance_matrix[np.ix_(al_indices, remaining_si)]
    al_distances = distance_matrix[np.ix_(al_indices, al_indices)]

    # Upper triangle only: each Al-Al pair is counted once.
    al_pair_distances = al_distances[np.triu_indices(len(al_indices), k=1)]

    if np.any(al_pair_distances <= 0):
        return math.inf

    # a = 1: too big central concentration;
    # a = 0.5 and 0.25: still too big central concentration and clustering;
    a = 0.1
    return float(np.sum(1.0 / al_pair_distances)) + a * float(np.var(al_si_distances))


# ----------------------------------------------------------------------
# Configuration search
# ----------------------------------------------------------------------

def generate_random_configuration(si_indices, number_of_al):
    """Choose one random set of Si atoms for substitution."""
    return tuple(sorted(random.sample(si_indices, number_of_al)))


def find_optimal_configuration(
    si_indices,
    number_of_al,
    initial_al_indices,
    number_of_configurations,
    total_possible,
    distance_matrix,
):
    """
    Search random configurations and return the one with the lowest score.
    """
    if number_of_configurations > total_possible:
        print(
            f"\nRequested {number_of_configurations:,} configurations, "
            f"but only {total_possible:,} unique configurations exist."
        )
        print("All possible configurations will be searched instead.")
        number_of_configurations = total_possible

    # For small configuration spaces, searching all combinations avoids
    # unnecessary random duplicates.
    if number_of_configurations == total_possible:
        configurations = itertools.combinations(si_indices, number_of_al)
    else:
        configurations = None

    best_configuration = None
    best_score = math.inf
    best_min_distance = 0.0

    if configurations is not None:
        for configuration_number, configuration in enumerate(
            configurations, start=1
        ):
            all_al_indices = tuple(
                sorted(initial_al_indices + list(configuration))
            )
            score = score_configuration(all_al_indices, distance_matrix)

            if score < best_score:
                best_score = score
                best_configuration = tuple(configuration)
                best_min_distance = minimum_al_distance(
                    all_al_indices, distance_matrix
                )

    else:
        # Random sampling without replacement at the configuration level.
        # A set prevents the same configuration from being evaluated twice.
        seen = set()

        while len(seen) < number_of_configurations:
            configuration = generate_random_configuration(
                si_indices, number_of_al
            )

            if configuration in seen:
                continue

            seen.add(configuration)

            all_al_indices = tuple(
                sorted(initial_al_indices + list(configuration))
            )
            score = score_configuration(all_al_indices, distance_matrix)

            if score < best_score:
                best_score = score
                best_configuration = configuration
                best_min_distance = minimum_al_distance(
                    all_al_indices, distance_matrix
                )

    return best_configuration, best_score, best_min_distance


# ----------------------------------------------------------------------
# Main program
# ----------------------------------------------------------------------

def main():
    # Forces Windows legacy console (conhost.exe) to enable ANSI escape codes processing
    # Necessary to print text with colored background
    os.system('')

    # The center of the cluster may not coincide with the origin (0, 0, 0).
    # This requirement isn't necessary for the minimum-image calculation itself.
    # Translation of the entire structure doesn't change any pairwise distances.

    print("Si -> Al Exchange Maker\n")
    print("The input cluster must be a rectangular parallelepiped")
    print("with its faces perpendicular to the x, y, and z axes.\n")
    
    # This requirement is necessary, because the code assumes an orthorhombic cell,
    # meaning perpendicular coordinates axes with cell vectors:
    # (Lx, 0, 0)
    # (0, Ly, 0)
    # (0, 0, Lz)
    # A general parallelepiped, such as a real triclinic zeolite unit cell with non-90° angles,
    # would require a somewhat more sophisticated treatment using the cell vectors
    # rather than three independent lengths.

    try:
        comment, atoms = read_xyz(INPUT_XYZ_FILE)
    except (OSError, ValueError) as e:
        print(f"Error reading {INPUT_XYZ_FILE}: {e}")
        sys.exit(1)

    si_indices = get_si_indices(atoms)
    initial_al_indices = get_al_indices(atoms)

    if not si_indices:
        print("Error: no Si atoms were found in the XYZ file.")
        sys.exit(1)

    print(f"Input file: {INPUT_XYZ_FILE}")
    print(f"Number of atoms: {len(atoms)}")
    print(f"Number of Si atoms: {len(si_indices)}\n")

    try:
        number_of_al = int(input("Number of Si -> Al exchanges: "))
    except ValueError:
        print("Error: please enter an integer.")
        sys.exit(1)

    if not 1 <= number_of_al <= len(si_indices):
        print(
            f"Error: the number of exchanges must be between "
            f"1 and {len(si_indices)}."
        )
        sys.exit(1)

    total_possible = math.comb(len(si_indices), number_of_al)
    print(f"Total possible unique configurations: {total_possible}")

    try:
        number_of_configurations = int(
            input("Number of configurations to search: ")
        )
    except ValueError:
        print("Error: please enter an integer.")
        sys.exit(1)

    if number_of_configurations < 1:
        print("Error: the number of configurations must be at least 1.")
        sys.exit(1)

    start_time = perf_counter()
    print(f"Started at: {datetime.now().strftime('%Y.%m.%d %H:%M:%S')}")

    cell_lengths = calculate_cell_lengths(atoms)

    print(
        f"\nPeriodic cell lengths: "
        f"Lx = {cell_lengths[0]:.3f} Å, "
        f"Ly = {cell_lengths[1]:.3f} Å, "
        f"Lz = {cell_lengths[2]:.3f} Å"
    )

    print("Calculating distance matrix...")

    distance_matrix = calculate_periodic_distance_matrix(atoms, cell_lengths)

    # distance_matrix = calculate_distance_matrix(atoms)

    print("Searching configurations...")

    best_configuration, best_score, best_min_distance = (
        find_optimal_configuration(
            si_indices,
            number_of_al,
            initial_al_indices,
            number_of_configurations,
            total_possible,
            distance_matrix,
        )
    )

    if best_configuration is None:
        print("Error: no configuration was found.")
        sys.exit(1)

    print("\nOptimal configuration found.")
    print(f"{GREEN}Score: {best_score:.6f}{RESET}")

    if number_of_al == 1:
        print("Minimum Al-Al distance: N/A (only one Al atom)")
    else:
        print(f"{GREEN}Minimum Al-Al distance: {best_min_distance:.3f} Å{RESET}")

    print("\nAl substitution sites (1-based atom numbers):")
    print(" ".join(str(i + 1) for i in best_configuration))

    print(f"\nTotal runtime: {elapsed_time(perf_counter() - start_time)}")
    print(f"Finished at: {datetime.now().strftime('%Y.%m.%d %H:%M:%S')}")
    input_path = Path(INPUT_XYZ_FILE)
    output_file = (
        input_path.parent
        / f"{input_path.stem}_Al{number_of_al}_optimal.xyz"
    )

    if bool(int(input("\nDump output? [1/0]: "))):
        write_xyz(
            output_file,
            comment,
            atoms,
            best_configuration,
        )
        print(f"Output written to: {output_file}")

    print("\nScript execution finished successfully.")


if __name__ == "__main__":
    main()
