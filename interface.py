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


def extract_defense_config(raw):
    """Flatten searchable defense metadata without changing the database."""
    description = parse_json_dict(raw)
    result = []
    for item in description.get("gradient_transformations", []) or []:
        if not isinstance(item, dict):
            continue
        ranking = item.get("ranking_policy") or {}
        draft = item.get("draft_policy") or {}
        transform = item.get("transform") or {}
        params = transform.get("parameters") or {} if isinstance(transform, dict) else {}
        construction = params.get("construction") or {}
        scaling = params.get("scaling") or {}
        result.append({
            "preset": str(description.get("defense_name") or ""),
            "transformation": str(item.get("name") or ""),
            "ranking": str(ranking.get("name") or "") if isinstance(ranking, dict) else str(ranking),
            "draft": str(draft.get("name") or "") if isinstance(draft, dict) else str(draft),
            "transform": str(transform.get("name") or "") if isinstance(transform, dict) else str(transform),
            "construction": str(construction.get("name") or "") if isinstance(construction, dict) else "",
            "scaling": str(scaling.get("name") or "") if isinstance(scaling, dict) else "",
            "parameters": {
                "ranking": ranking.get("parameters", {}) if isinstance(ranking, dict) else {},
                "draft": draft.get("parameters", {}) if isinstance(draft, dict) else {},
                "transform": params,
                "construction": construction.get("parameters", {}) if isinstance(construction, dict) else {},
                "scaling": scaling.get("parameters", {}) if isinstance(scaling, dict) else {},
            },
        })
    return result


def flatten_parameter_paths(value, prefix=""):
    if isinstance(value, dict):
        for key, child in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            yield from flatten_parameter_paths(child, path)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from flatten_parameter_paths(child, f"{prefix}[{index}]")
    else:
        yield prefix, value


def matches_parameter(transform, key, text):
    values = list(flatten_parameter_paths(transform["parameters"]))
    return any(
        (not key or key.casefold() in path.casefold())
        and (not text or text.casefold() in str(value).casefold())
        for path, value in values
    )


def transformation_matches(transform, choices, keyword, parameter_key, parameter_value):
    for field, selected in choices.items():
        if selected and transform[field] not in selected:
            return False
    if keyword:
        searchable = " ".join(str(transform[field]) for field in (
            "preset", "transformation", "ranking", "draft", "transform", "construction", "scaling"
        ))
        if keyword.casefold() not in searchable.casefold():
            return False
    if parameter_key or parameter_value:
        if not matches_parameter(transform, parameter_key, parameter_value):
            return False
    return True


def defense_matches(run, choices, keyword, parameter_key, parameter_value):
    transforms = run["defense_config"]
    # A family search can also match a preset name even if the nested config is absent.
    if not transforms:
        return not any(choices.values()) and not (parameter_key or parameter_value) and (
            not keyword or keyword.casefold() in str(run["defense"]).casefold()
        )
    return any(
        transformation_matches(t, choices, keyword, parameter_key, parameter_value)
        for t in transforms
    )


def all_config_values(runs, field):
    return sorted({t[field] for transforms in runs["defense_config"] for t in transforms if t[field]})


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
runs["defense_config"] = runs["raw_description"].apply(extract_defense_config)
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

st.sidebar.divider()
st.sidebar.subheader("Defense internals")
st.sidebar.caption("Combine filters within the same transformation. Leave fields blank for all values.")
family_query = st.sidebar.text_input(
    "Defense family / name contains", placeholder="max_svd", key="defense_family_query"
).strip()
internal_choices = {
    field: st.sidebar.multiselect(title, all_config_values(runs, field), key=f"defense_{field}")
    for field, title in [
        ("ranking", "Ranking policy"),
        ("draft", "Draft policy"),
        ("transform", "Transform type"),
        ("construction", "Noise construction"),
        ("scaling", "Noise scaling / threshold"),
    ]
}
with st.sidebar.expander("Parameter search"):
    st.caption("Search nested configuration parameters (e.g. scaling.target_error or scaling.mu_j).")
    parameter_key = st.text_input("Parameter path contains", placeholder="scaling.target_error").strip()
    parameter_value = st.text_input("Parameter value contains", placeholder="0.02").strip()

if family_query or any(internal_choices.values()) or parameter_key or parameter_value:
    mask = filtered_runs.apply(
        lambda run: defense_matches(run, internal_choices, family_query, parameter_key, parameter_value),
        axis=1,
    )
    filtered_runs = filtered_runs.loc[mask]

st.sidebar.caption(f"{len(filtered_runs)} / {len(runs)} runs match")


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
    only_matching_components = st.checkbox("Only components from matching runs", value=True)
    components = load_components(filtered_runs["id"].astype(int).tolist()) if only_matching_components else load_components()
    categories = st.multiselect("Component categories", available_values(components, "category"))
    if categories:
        components = components[components["category"].isin(categories)]
    st.dataframe(components, use_container_width=True, hide_index=True)

with logs_tab:
    st.subheader("All Logs")
    only_matching_logs = st.checkbox("Only logs from matching runs", value=True)
    st.dataframe(load_logs(filtered_runs["id"].astype(int).tolist() if only_matching_logs else None), use_container_width=True, hide_index=True)

with images_tab:
    st.subheader("Stored Images")
    only_matching_images = st.checkbox("Only images from matching runs", value=True)
    image_rows = load_image_rows(filtered_runs["id"].astype(int).tolist() if only_matching_images else None)
    st.dataframe(image_rows, use_container_width=True, hide_index=True)
    if not image_rows.empty:
        image_id = st.selectbox("View images for run", sorted(image_rows["run_id"].unique().astype(int).tolist()))
        show_images(image_id)