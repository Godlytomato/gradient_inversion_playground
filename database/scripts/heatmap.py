import json
import sqlite3

import matplotlib.pyplot as plt
import numpy as np

from database.connection import DB_PATH
from validation.attack_presets import ATTACK_PRESETS
from validation.defense_presets import DEFENSE_PRESETS


def load_results():
    connection = sqlite3.connect(DB_PATH)

    rows = connection.execute(
        """
        SELECT
            attack_name,
            defense_name,
            outcome
        FROM runs
        """
    ).fetchall()

    connection.close()

    return rows

def build_matrix(rows):

    attacks = list(ATTACK_PRESETS.keys())
    defenses = list(DEFENSE_PRESETS.keys())

    matrix = np.full(
        (len(attacks), len(defenses)),
        np.nan
    )

    for attack, defense, outcome in rows:

        if attack not in attacks:
            continue

        if defense not in defenses:
            continue

        attack_index = attacks.index(attack)
        defense_index = defenses.index(defense)

        if outcome == "success":
            matrix[attack_index, defense_index] = 1

        elif outcome == "failure":
            matrix[attack_index, defense_index] = 0

    return attacks, defenses, matrix

def plot_heatmap(
    attacks,
    defenses,
    matrix
):

    fig, ax = plt.subplots(
        figsize=(16, 8)
    )

    image = ax.imshow(
        matrix,
        vmin=0,
        vmax=1,
        aspect="auto"
    )

    ax.set_xticks(
        range(len(defenses))
    )

    ax.set_xticklabels(
        defenses,
        rotation=60,
        ha="right"
    )

    ax.set_yticks(
        range(len(attacks))
    )

    ax.set_yticklabels(
        attacks
    )

    ax.set_xlabel(
        "Defense"
    )

    ax.set_ylabel(
        "Attack"
    )

    ax.set_title(
        "Attack × Defense Reconstruction Success"
    )

    # show result in each cell
    for row in range(len(attacks)):
        for column in range(len(defenses)):

            value = matrix[row, column]

            if np.isnan(value):
                label = "-"
            elif value == 1:
                label = "S"
            else:
                label = "F"

            ax.text(
                column,
                row,
                label,
                ha="center",
                va="center"
            )

    colorbar = plt.colorbar(
        image,
        ax=ax
    )

    colorbar.set_ticks(
        [0, 1]
    )

    colorbar.set_ticklabels(
        [
            "Attack Failed",
            "Attack Succeeded"
        ]
    )

    plt.tight_layout()

    plt.savefig(
        "attack_defense_heatmap.png",
        dpi=200
    )

    plt.show()


def main():

    rows = load_results()

    attacks, defenses, matrix = build_matrix(
        rows
    )

    plot_heatmap(
        attacks,
        defenses,
        matrix
    )


if __name__ == "__main__":
    main()