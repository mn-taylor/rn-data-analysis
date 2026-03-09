import os, json
import pandas as pd

def _clean(x):
    return str(x).replace("/", "-").replace("\\", "-").replace(" ", "_")

def split_cycles_sorted_by_patient_type(
    df: pd.DataFrame,
    metadata_json_path: str,
    patients_json_path: str,
    out_dir: str = "data",
    unknown_folder: str = "UNKNOWN",
    analyte_value: str = "analyte",
):
    os.makedirs(out_dir, exist_ok=True)

    # --- load metadata: run_id -> patient_id ---
    with open(metadata_json_path, "r") as f:
        meta_list = json.load(f)
    run_to_patient = {
        str(m["run_id"]): str(m["patient_id"])
        for m in meta_list
        if "run_id" in m and "patient_id" in m
    }

    # --- load patients: patient_id -> type ---
    with open(patients_json_path, "r") as f:
        patients_list = json.load(f)
    patient_to_type = {
        str(p["patient_id"]): str(p.get("type", unknown_folder)).strip()
        for p in patients_list
        if "patient_id" in p
    }

    df = df.copy()
    df["run_id"] = df["run_id"].astype(str)

    for (run_id, cycle), g in df.groupby(["run_id", "cycle"], sort=False):
        # keep only analyte rows
        if "section" in g.columns:
            g = g[g["section"] == analyte_value]
        else:
            # if there's no section column, skip (or you could just save as-is)
            continue

        # if the cycle has no analyte rows, skip writing
        if g.empty:
            continue

        pid = run_to_patient.get(run_id)

        # fallback: if metadata missing, try the df column if it exists
        if pid is None and "patient_id" in g.columns and not g["patient_id"].isna().all():
            pid = str(g["patient_id"].iloc[0])

        patient_type = patient_to_type.get(str(pid), unknown_folder)
        patient_type = _clean(patient_type) if patient_type else unknown_folder

        type_dir = os.path.join(out_dir, patient_type)
        os.makedirs(type_dir, exist_ok=True)

        fname = f"{_clean(run_id)}_{_clean(cycle)}.csv"
        g.to_csv(os.path.join(type_dir, fname), index=False)

if __name__ == "__main__":

    df = pd.read_csv("/Users/keane/Desktop/research/rn-data-analysis/data/mdd_data_v3/unparsed/MDD-48h.csv")

    # usage:
    split_cycles_sorted_by_patient_type(
        df, "/Users/keane/Desktop/research/rn-data-analysis/data/mdd_data_v3/unparsed/MDD-48h_metadata.json", "/Users/keane/Desktop/research/rn-data-analysis/data/mdd_data_v3/unparsed/MDD-48h_patients.json", out_dir="/Users/keane/Desktop/research/rn-data-analysis/data/mdd_data_v3/mdd_48h/"
    )
