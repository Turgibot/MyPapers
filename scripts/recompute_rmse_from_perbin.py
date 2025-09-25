import csv
import glob
import os
import math

ROOT = "/home/guy/Projects/Traffic/MyPapers/results"

def load_perbin(perbin_path):
    by_epoch = {}
    with open(perbin_path, newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            if row.get("split") != "val":
                continue
            e = int(row["epoch"])
            try:
                rmse = float(row["rmse_s"])
                n = int(row["n"]) if row.get("n") else 0
            except:
                continue
            by_epoch.setdefault(e, {"sq_sum":0.0, "n":0})
            by_epoch[e]["sq_sum"] += (rmse * rmse) * n
            by_epoch[e]["n"]      += n
    # convert to rmse
    for e, agg in by_epoch.items():
        if agg["n"] > 0:
            agg["rmse"] = math.sqrt(agg["sq_sum"] / agg["n"])
        else:
            agg["rmse"] = None
    return by_epoch

def rewrite_epoch(epoch_path, rmse_by_epoch):
    with open(epoch_path, newline="") as f:
        rows = list(csv.DictReader(f))
        fields = rows[0].keys() if rows else [
            "epoch","split","loss","lb","val_MSE","val_secMAE","val_secRMSE","val_MAPE","router_entropy","router_max_share","target_key"
        ]

    changed = False
    for row in rows:
        if row.get("split") != "val":
            continue
        try:
            e = int(row["epoch"])
        except:
            continue
        rmse = rmse_by_epoch.get(e, {}).get("rmse")
        if rmse is not None:
            # if existing is missing or suspiciously tiny (< 5 sec), replace
            try:
                curr = float(row.get("val_secRMSE",""))
            except:
                curr = None
            if curr is None or curr < 5.0:
                row["val_secRMSE"] = f"{rmse:.15f}"
                changed = True

    if changed:
        with open(epoch_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for r in rows:
                w.writerow(r)

def main():
    for epoch_path in glob.glob(os.path.join(ROOT, "epoch_metrics_*.csv")):
        base = os.path.basename(epoch_path).replace("epoch_metrics_", "")
        perbin_path = os.path.join(ROOT, f"perbin_metrics_{base}")
        if not os.path.exists(perbin_path):
            continue
        rmse_by_epoch = load_perbin(perbin_path)
        rewrite_epoch(epoch_path, rmse_by_epoch)

if __name__ == "__main__":
    main()


