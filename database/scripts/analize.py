from pathlib import Path
import json
import sqlite3

import matplotlib.pyplot as plt

from database.connection import DB_PATH


OUTPUT_DIR = Path(
    "temp_logs"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


def load_runs(
    connection
):

    return connection.execute(
        """
        SELECT
            id,
            run_hash,
            name,
            model,
            dataset,
            batch_size,
            attack_name,
            attack_hash,
            defense_name,
            defense_hash,
            outcome
        FROM runs

        ORDER BY created_at
        """
    ).fetchall()


def load_logs(
    connection,
    run_id
):

    return connection.execute(
        """
        SELECT
            step,
            metrics

        FROM logs

        WHERE run_id = ?

        ORDER BY step
        """,
        (
            run_id,
        )
    ).fetchall()


def analyze_mse(
    log_rows
):

    steps = []
    mse_values = []

    for (
        step,
        metrics_json,
    ) in log_rows:

        metrics = json.loads(
            metrics_json
        )

        mse = metrics.get(
            "mse"
        )

        if mse is None:
            continue

        steps.append(
            step
        )

        mse_values.append(
            float(mse)
        )

    if not mse_values:
        return None

    start_mse = mse_values[0]

    final_mse = mse_values[-1]

    best_mse = min(
        mse_values
    )

    best_index = mse_values.index(
        best_mse
    )

    best_step = steps[
        best_index
    ]

    improvement = (
        start_mse
        - final_mse
    )

    percent_improvement = (
        improvement
        / start_mse
        * 100
        if start_mse != 0
        else 0.0
    )

    return {
        "steps": steps,
        "mse": mse_values,

        "start_mse": start_mse,
        "final_mse": final_mse,

        "best_mse": best_mse,
        "best_step": best_step,

        "improvement": improvement,
        "percent_improvement":
            percent_improvement,
    }


def save_plot(
    run_name,
    run_hash,
    analysis,
):

    plt.figure()

    plt.plot(
        analysis["steps"],
        analysis["mse"],
    )

    plt.xlabel(
        "Iteration"
    )

    plt.ylabel(
        "MSE"
    )

    plt.title(
        f"{run_name}\n"
        f"{run_hash[:12]}"
    )

    plt.tight_layout()

    output_path = OUTPUT_DIR / (
        f"{run_name}"
        f"__{run_hash[:12]}"
        f"__mse.png"
    )

    plt.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()

    return output_path


def main():

    connection = sqlite3.connect(
        DB_PATH
    )

    runs = load_runs(
        connection
    )


    print()
    print("=" * 70)
    print("RUN LOG ANALYSIS")
    print("=" * 70)


    for (
        run_id,
        run_hash,
        run_name,
        model,
        dataset,
        batch_size,
        attack_name,
        attack_hash,
        defense_name,
        defense_hash,
        outcome,
    ) in runs:

        logs = load_logs(
            connection,
            run_id,
        )

        analysis = analyze_mse(
            logs
        )


        print()
        print("-" * 70)

        print(
            run_name
        )

        print(
            f"hash:      {run_hash}"
        )

        print(
            f"model:     {model}"
        )

        print(
            f"dataset:   {dataset}"
        )

        print(
            f"batch:     {batch_size}"
        )

        print(
            f"attack:    "
            f"{attack_name or attack_hash[:12]}"
        )

        print(
            f"defense:   "
            f"{defense_name or defense_hash[:12]}"
        )

        print(
            f"outcome:   {outcome}"
        )

        print(
            f"logs:      {len(logs)}"
        )


        if analysis is None:

            print(
                "No MSE values found."
            )

            continue


        print(
            f"start MSE: "
            f"{analysis['start_mse']:.6f}"
        )

        print(
            f"final MSE: "
            f"{analysis['final_mse']:.6f}"
        )

        print(
            f"best MSE:  "
            f"{analysis['best_mse']:.6f}"
            f" at step "
            f"{analysis['best_step']}"
        )

        print(
            f"change:    "
            f"{analysis['improvement']:.6f}"
        )

        print(
            f"improved:  "
            f"{analysis['percent_improvement']:.2f}%"
        )


        if (
            analysis["final_mse"]
            < analysis["start_mse"]
        ):

            print(
                "trend:     improved overall"
            )

        elif (
            analysis["final_mse"]
            > analysis["start_mse"]
        ):

            print(
                "trend:     got worse overall"
            )

        else:

            print(
                "trend:     unchanged overall"
            )


        if (
            analysis["best_step"]
            < analysis["steps"][-1]
        ):

            print(
                "note:      best reconstruction "
                "occurred before the final iteration"
            )


        plot_path = save_plot(
            run_name,
            run_hash,
            analysis,
        )

        print(
            f"plot:      {plot_path}"
        )


    connection.close()

    print()
    print("=" * 70)


if __name__ == "__main__":
    main()