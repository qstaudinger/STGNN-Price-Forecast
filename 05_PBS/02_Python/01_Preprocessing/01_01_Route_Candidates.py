import pandas as pd
import numpy as np
import requests
from tqdm import tqdm

BASE_URL = "http://localhost:5000"
INFILE = "stations_candidates_r14_max30.csv.gz"
OUTFILE = "stations_travel_times.csv.gz"

CHUNK_SIZE = 50
TIMEOUT = 30

def osrm_table_one_to_many(src_lonlat, dst_lonlat_list):
    coords = [src_lonlat] + dst_lonlat_list
    coord_str = ";".join(f"{lon},{lat}" for lon, lat in coords)
    dest_idx = ";".join(str(i) for i in range(1, len(coords)))
    url = (
        f"{BASE_URL}/table/v1/driving/{coord_str}"
        f"?sources=0&destinations={dest_idx}&annotations=duration"
    )
    r = requests.get(url, timeout=TIMEOUT)
    r.raise_for_status()
    durs = r.json()["durations"][0]
    return np.array([np.nan if d is None else d for d in durs], dtype=float)

def main():
    edges = pd.read_csv(INFILE)

    # Minimally check whether the columns are there
    needed = {"src_uuid","dst_uuid","src_lat","src_lon","dst_lat","dst_lon"}
    missing = needed - set(edges.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    out_parts = []

    for src_uuid, g in tqdm(edges.groupby("src_uuid"), total=edges["src_uuid"].nunique()):
        src_lat = float(g.iloc[0]["src_lat"])
        src_lon = float(g.iloc[0]["src_lon"])
        src = (src_lon, src_lat)

        dst_lonlat = list(zip(g["dst_lon"].astype(float), g["dst_lat"].astype(float)))

        # chunking if necessary (if you have more than 50 candidates per source)
        mins_all = []
        for start in range(0, len(dst_lonlat), CHUNK_SIZE):
            chunk = dst_lonlat[start:start+CHUNK_SIZE]
            durs_s = osrm_table_one_to_many(src, chunk)
            mins_all.append(durs_s / 60.0)

        travel_min = np.concatenate(mins_all) if mins_all else np.array([], dtype=float)

        tmp = g[["src_uuid","dst_uuid"]].copy()
        tmp["travel_time_min"] = travel_min.astype("float32")
        out_parts.append(tmp)

    out = pd.concat(out_parts, ignore_index=True)

    # Optional: Throw out unreachable
    out = out[np.isfinite(out["travel_time_min"])].copy()

    out.to_csv(OUTFILE, index=False, compression="gzip")
    print("Wrote:", OUTFILE, "rows:", len(out))

if __name__ == "__main__":
    main()