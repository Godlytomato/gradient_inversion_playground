import io
import json
import sqlite3

import pandas as pd
import streamlit as st
import torch

from database.connection import DB_PATH
from database.image_display import viewable, align_for_display

st.set_page_config(page_title="Gradient Inversion Playground", layout="wide")
st.title("Gradient Inversion Playground")
st.caption(f"Database: {DB_PATH}")


def query_dataframe(query, parameters=()):
    with sqlite3.connect(DB_PATH) as connection:
        return pd.read_sql_query(query, connection, params=parameters)


def load_runs():
    with sqlite3.connect(DB_PATH) as connection:
        columns = {row[1] for row in connection.execute("PRAGMA table_info(runs)")}
    extra = ", summary" if "summary" in columns else ", '{}' AS summary"
    return query_dataframe("""
        SELECT id, run_hash, name, model, dataset, batch_size,
               attack_name, attack_hash, defense_name, defense_hash,
               outcome, run_metrics, raw_description, created_at
    """ + extra + """
        FROM runs ORDER BY id DESC
    """)


def load_components(run_ids=None):
    query = "SELECT id, run_id, category, name, position, parameters FROM components"
    params = ()
    if run_ids is not None:
        if not run_ids:
            return pd.DataFrame(columns=["id", "run_id", "category", "name", "position", "parameters"])
        query += f" WHERE run_id IN ({','.join('?' for _ in run_ids)})"
        params = tuple(run_ids)
    return query_dataframe(query + " ORDER BY run_id, category, position", params)


def load_logs(run_ids=None):
    query = "SELECT id, run_id, step, metrics, created_at FROM logs"
    params = ()
    if run_ids is not None:
        if not run_ids:
            return pd.DataFrame(columns=["id", "run_id", "step", "metrics", "created_at"])
        query += f" WHERE run_id IN ({','.join('?' for _ in run_ids)})"
        params = tuple(run_ids)
    return query_dataframe(query + " ORDER BY run_id, step", params)


def load_image_rows(run_ids=None):
    query = "SELECT image_hash, run_id, image_type, length(image_data) AS bytes, created_at FROM images"
    params = ()
    if run_ids is not None:
        if not run_ids:
            return pd.DataFrame(columns=["image_hash", "run_id", "image_type", "bytes", "created_at"])
        query += f" WHERE run_id IN ({','.join('?' for _ in run_ids)})"
        params = tuple(run_ids)
    return query_dataframe(query + " ORDER BY run_id, image_type", params)


def load_images(run_id):
    with sqlite3.connect(DB_PATH) as connection:
        columns = {row[1] for row in connection.execute("PRAGMA table_info(runs)")}
        summary = {}
        if "summary" in columns:
            raw = connection.execute("SELECT summary FROM runs WHERE id = ?", (int(run_id),)).fetchone()
            summary = json.loads(raw[0]) if raw else {}
        rows = connection.execute(
            "SELECT image_hash, image_type, image_data FROM images WHERE run_id = ? ORDER BY image_type",
            (int(run_id),),
        ).fetchall()
    return [
        {"hash": hash_value, "type": kind,
         "tensor": align_for_display(torch.load(io.BytesIO(data), map_location="cpu", weights_only=True), kind, summary)}
        for hash_value, kind, data in rows
    ]


def display_name(name, hash_value):
    if pd.notna(name) and str(name).strip():
        return str(name)
    return str(hash_value)[:12] if pd.notna(hash_value) else "unknown"


def make_viewable(image):
    return viewable(image)


def show_images(run_id):
    records = load_images(run_id)
    if not records:
        st.info("No images stored for this run.")
    for record in records:
        st.markdown(f"**{record['type']}**")
        st.caption(record["hash"])
        tensor = record["tensor"]
        if tensor.ndim == 4:
            columns = st.columns(min(len(tensor), 4))
            for index, sample in enumerate(tensor):
                with columns[index % len(columns)]:
                    st.image(make_viewable(sample), caption=f"Sample {index}", use_container_width=True)
        else:
            st.image(make_viewable(tensor), use_container_width=False)


def parse_json_dict(raw):
    if isinstance(raw, dict):
        return raw
    if raw is None or pd.isna(raw):
        return {}
    try:
        value = json.loads(raw)
        return value if isinstance(value, dict) else {}
    except (ValueError, TypeError):
        return {}


TRANSFORMATION_METRICS = (
    "delta_l2", "relative_delta_l2", "changed_fraction", "missing_fraction", "selected_fraction"
)


def transformation_dict(raw):
    values = numeric_dict(raw)
    return {name: values[name] for name in TRANSFORMATION_METRICS if name in values}


def numeric_dict(raw):
    values = {}
    for name, value in parse_json_dict(raw).items():
        try:
            number = float(value)
            if pd.notna(number) and abs(number) != float("inf"):
                values[name] = number
        except (ValueError, TypeError, OverflowError):
            pass
    return values


def parse_logs(logs):
    rows = []
    for row in logs.itertuples(index=False):
        rows.append({"run_id": int(row.run_id), "step": int(row.step), **numeric_dict(row.metrics)})
    return pd.DataFrame(rows)


def available_values(dataframe, column):
    return sorted(dataframe[column].dropna().astype(str).unique().tolist())


def filter_runs(dataframe, filters):
    filtered = dataframe
    for column, selection in filters.items():
        if selection:
            filtered = filtered[filtered[column].isin(selection)]
    return filtered


try:
    runs = load_runs()
except Exception as error:
    st.error(f"Could not load database: {error}")
    st.stop()
if runs.empty:
    st.warning("The database contains no runs.")
    st.stop()

runs["attack"] = runs.apply(lambda r: display_name(r["attack_name"], r["attack_hash"]), axis=1)
runs["defense"] = runs.apply(lambda r: display_name(r["defense_name"], r["defense_hash"]), axis=1)
runs["label"] = runs.apply(lambda r: f"{r['id']} | {r['name']} | {r['attack']} | {r['defense']}", axis=1)
labels = dict(zip(runs["id"].astype(int), runs["label"]))

st.sidebar.header("Run Filters")
filters = {
    column: st.sidebar.multiselect(title, available_values(runs, column))
    for column, title in [
        ("model", "Model"), ("dataset", "Dataset"), ("batch_size", "Batch Size"),
        ("attack", "Attack"), ("defense", "Defense"), ("outcome", "Outcome"),
    ]
}
# Sidebar multiselect returns matching native types for batch_size, not strings.
if filters["batch_size"]:
    filters["batch_size"] = [int(value) for value in filters["batch_size"]]
filtered_runs = filter_runs(runs, filters)

run_tab, compare_tab, components_tab, logs_tab, images_tab = st.tabs(
    ["Runs", "Compare Runs", "Components", "Logs", "Images"]
)

with run_tab:
    st.subheader("Runs")
    st.dataframe(
        filtered_runs[["id", "name", "model", "dataset", "batch_size", "attack", "defense", "outcome", "created_at"]],
        use_container_width=True, hide_index=True,
    )
    if filtered_runs.empty:
        st.warning("No runs match the sidebar filters.")
    else:
        selected_id = st.selectbox(
            "Select run", filtered_runs["id"].astype(int).tolist(),
            format_func=lambda run_id: labels[run_id],
        )
        selected = runs.loc[runs["id"] == selected_id].iloc[0]
        st.subheader("Run Details")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Model", selected["model"])
            st.metric("Dataset", selected["dataset"])
        with col2:
            st.metric("Attack", selected["attack"])
            st.metric("Defense", selected["defense"])
        with col3:
            st.metric("Batch Size", int(selected["batch_size"]))
            st.metric("Outcome", selected["outcome"] if pd.notna(selected["outcome"]) else "running")
        st.markdown("**Defense transformation measurements**")
        run_values = transformation_dict(selected["run_metrics"])
        if run_values:
            st.dataframe(pd.DataFrame([run_values]), hide_index=True, use_container_width=True)
        else:
            st.info("No defense magnitude or mask measurements saved for this run.")
        with st.expander("Run description"):
            st.json(parse_json_dict(selected["raw_description"]))
        with st.expander("Final summary (practical and oracle results)"):
            st.json(parse_json_dict(selected["summary"]))
        with st.expander("Hashes"):
            st.code(f"Run: {selected['run_hash']}\nAttack: {selected['attack_hash']}\nDefense: {selected['defense_hash']}")
        st.subheader("Components")
        st.dataframe(load_components([selected_id]).drop(columns=["id", "run_id"]), use_container_width=True, hide_index=True)
        st.subheader("Images")
        show_images(selected_id)
        st.subheader("Iteration Metrics")
        single_logs = parse_logs(load_logs([selected_id]))
        if single_logs.empty:
            st.info("No iteration metrics logged for this run.")
        else:
            metric_names = [c for c in single_logs if c not in ("run_id", "step")]
            chosen = st.multiselect("Metrics to graph", metric_names, default=metric_names[:1], key="single_metrics")
            if chosen:
                st.line_chart(single_logs.set_index("step")[chosen], use_container_width=True)
            with st.expander("Metric data"):
                st.dataframe(single_logs, use_container_width=True, hide_index=True)

with compare_tab:
    st.subheader("Compare Runs")
    mode = st.radio("Select runs by", ["Filters", "Run IDs"], horizontal=True)
    if mode == "Filters":
        st.caption("The sidebar filters define the candidate runs. All matching runs are initially selected; remove any you don't want.")
        candidates = filtered_runs["id"].astype(int).tolist()
        chosen_ids = st.multiselect(
            "Runs to compare", candidates, default=candidates,
            format_func=lambda run_id: labels[run_id], key="compare_filtered_ids",
        )
    else:
        id_text = st.text_input("Run IDs (comma-separated)", placeholder="12, 15, 20")
        chosen_ids = []
        if id_text.strip():
            try:
                chosen_ids = list(dict.fromkeys(int(part.strip()) for part in id_text.split(",") if part.strip()))
            except ValueError:
                st.error("Enter numeric run IDs separated by commas.")
        unknown = sorted(set(chosen_ids) - set(labels))
        if unknown:
            st.warning(f"Unknown run IDs: {unknown}")
            chosen_ids = [run_id for run_id in chosen_ids if run_id in labels]
    if not chosen_ids:
        st.info("Select one or more runs to compare.")
    else:
        selected_runs = runs[runs["id"].isin(chosen_ids)].copy()
        st.caption(f"{len(selected_runs)} runs selected")
        st.dataframe(
            selected_runs[["id", "name", "model", "dataset", "attack", "defense", "outcome"]],
            hide_index=True, use_container_width=True,
        )
        st.markdown("#### Defense transformation measurements")
        run_rows = [{"run_id": int(r.id), "run": labels[int(r.id)], **transformation_dict(r.run_metrics)}
                    for r in selected_runs.itertuples(index=False)]
        run_frame = pd.DataFrame(run_rows)
        run_metric_names = [c for c in run_frame.columns if c not in ("run_id", "run")]
        if run_metric_names:
            run_metric = st.selectbox("Run-level metric", run_metric_names)
            plot_data = run_frame[["run", run_metric]].dropna().set_index("run")
            if not plot_data.empty:
                st.bar_chart(plot_data, use_container_width=True)
            st.dataframe(run_frame, hide_index=True, use_container_width=True)
        else:
            st.info("Selected runs have no numeric run-level metrics yet.")
        st.markdown("#### Iteration metric comparison")
        history = parse_logs(load_logs(chosen_ids))
        if history.empty:
            st.info("No iteration logs for the selected runs.")
        else:
            history_metrics = [c for c in history.columns if c not in ("run_id", "step")]
            if not history_metrics:
                st.info("The selected logs contain no numeric metrics.")
            else:
                metric = st.selectbox("Iteration metric", history_metrics)
                chart = history.pivot_table(index="step", columns="run_id", values=metric, aggfunc="last")
                chart.columns = [labels[int(run_id)] for run_id in chart.columns]
                st.line_chart(chart.sort_index(), use_container_width=True)
                st.caption("Each line is one run. The X axis is the optimization step; missing steps remain blank.")

with components_tab:
    st.subheader("All Components")
    components = load_components()
    categories = st.multiselect("Component categories", available_values(components, "category"))
    if categories:
        components = components[components["category"].isin(categories)]
    st.dataframe(components, use_container_width=True, hide_index=True)

with logs_tab:
    st.subheader("All Logs")
    st.dataframe(load_logs(), use_container_width=True, hide_index=True)

with images_tab:
    st.subheader("Stored Images")
    image_rows = load_image_rows()
    st.dataframe(image_rows, use_container_width=True, hide_index=True)
    if not image_rows.empty:
        image_id = st.selectbox("View images for run", sorted(image_rows["run_id"].unique().astype(int).tolist()))
        show_images(image_id)
