import sqlite3
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

from database.connection import DB_PATH


def main():
    with sqlite3.connect(DB_PATH) as connection:
        rows = connection.execute("""
            SELECT model, attack_name, defense_name,
                   outcome, name, attack_hash, defense_hash
            FROM runs
        """).fetchall()

    results = defaultdict(lambda: defaultdict(list))
    attacks = set()
    defenses = set()

    for model, attack, defense, outcome, name, ah, dh in rows:
        parts = name.split("__")

        attack = attack or (parts[0] if len(parts) >= 4 else ah[:12])
        defense = defense or (parts[1] if len(parts) >= 4 else dh[:12])

        attacks.add(attack)
        defenses.add(defense)

        cell = results[model][(attack, defense)]

        if outcome == "success":
            cell.append(1)
        elif outcome == "failure":
            cell.append(0)

    attacks = sorted(attacks)
    defenses = sorted(defenses)

    for model, cells in sorted(results.items()):
        matrix = np.full(
            (len(attacks), len(defenses)),
            np.nan
        )

        fig, ax = plt.subplots(figsize=(18, 8))

        for i, attack in enumerate(attacks):
            for j, defense in enumerate(defenses):
                values = cells.get((attack, defense), [])

                if values:
                    matrix[i, j] = np.mean(values)
                    label = f"{matrix[i, j]:.0%}"
                else:
                    label = "-"

                ax.text(j, i, label, ha="center", va="center")

        image = ax.imshow(
            matrix,
            cmap="RdYlGn_r",
            vmin=0,
            vmax=1,
            aspect="auto"
        )

        ax.set_xticks(range(len(defenses)))
        ax.set_xticklabels(defenses, rotation=60, ha="right")
        ax.set_yticks(range(len(attacks)))
        ax.set_yticklabels(attacks)

        ax.set_xlabel("Defense")
        ax.set_ylabel("Attack")
        ax.set_title(f"{model} - Attack Success Rate")

        fig.colorbar(image, ax=ax, label="Attack Success Rate")
        fig.tight_layout()

        path = f"attack_defense_heatmap_{model}.png"
        fig.savefig(path, dpi=200)
        plt.close(fig)

        print(f"Saved: {path}")


if __name__ == "__main__":
    main()