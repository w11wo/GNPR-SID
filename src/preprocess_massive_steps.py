from collections import Counter, defaultdict
from argparse import ArgumentParser
from pathlib import Path
import csv

from openlocationcode import openlocationcode as olc
import pandas as pd


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--massive_steps_path", type=str, default="../Massive-STEPS/")
    parser.add_argument("--city", type=str)
    args = parser.parse_args()
    return args


def get_pluscode(latitude, longitude):
    # Plus Code
    plus_code = olc.encode(latitude, longitude)
    return plus_code[:6]


def get_forward_neighbors(df, column, min_freq=1):
    neighbor_counts = defaultdict(Counter)
    all_pois = set()

    for sequence in df[column]:
        all_pois.update(sequence)
        for i in range(len(sequence) - 1):
            current_poi = sequence[i]
            next_poi = sequence[i + 1]
            neighbor_counts[current_poi][next_poi] += 1

    df_data = []
    for poi in all_pois:
        counter = neighbor_counts.get(poi, {})
        filtered_neighbors = {neighbor: freq for neighbor, freq in counter.items() if freq >= min_freq}
        if filtered_neighbors:
            sorted_neighbors = [
                neighbor for neighbor, _ in sorted(filtered_neighbors.items(), key=lambda x: x[1], reverse=True)
            ]
        else:
            sorted_neighbors = []
        df_data.append((poi, sorted_neighbors))

    neighbors_df = pd.DataFrame(df_data, columns=[column, "neighbors"])
    return neighbors_df


def get_neighbors(df, column, min_freq=1):
    neighbor_counts = defaultdict(Counter)
    all_pois = set()

    for sequence in df[column]:
        all_pois.update(sequence)
        for i, poi in enumerate(sequence):
            if i > 0:
                neighbor_counts[poi][sequence[i - 1]] += 1
            if i < len(sequence) - 1:
                neighbor_counts[poi][sequence[i + 1]] += 1

    df_data = []
    for poi in all_pois:
        counter = neighbor_counts.get(poi, {})

        filtered_neighbors = {neighbor: freq for neighbor, freq in counter.items() if freq >= min_freq}

        sorted_neighbors = [
            neighbor for neighbor, _ in sorted(filtered_neighbors.items(), key=lambda x: x[1], reverse=True)
        ]
        df_data.append((poi, sorted_neighbors))

    neighbors_df = pd.DataFrame(df_data, columns=[column, "neighbors"])
    return neighbors_df


def main(args):
    massive_steps_path = Path(args.massive_steps_path)
    city = args.city
    data_path = massive_steps_path / "data" / city

    all_categories = set()
    category2schema = {}
    with open(massive_steps_path / "semantic-trails" / "mapping.csv", "r") as f:
        lines = csv.reader(f, delimiter=",")
        for line in lines:
            category_id, _, schema = line
            schema = schema.replace("schema:", "")
            category2schema[category_id] = schema
            all_categories.add(schema)

    all_categories = sorted(all_categories)

    df = pd.read_csv(data_path / f"{city}_checkins.csv")

    # fill missing lat/lon with city lat/lon
    df["latitude"] = df["latitude"].fillna(df["venue_city_latitude"])
    df["longitude"] = df["longitude"].fillna(df["venue_city_longitude"])

    # remap venue_category_id to schema.org's 162 categories
    df["Catname"] = df["venue_category_id"].apply(lambda x: category2schema[x])

    df["Region"] = df.apply(lambda row: get_pluscode(row["latitude"], row["longitude"]), axis=1)
    df["Time"] = pd.to_datetime(df["timestamp"], format="%Y-%m-%d %H:%M:%S").dt.hour

    df = df.rename(columns={"user_id": "Uid", "venue_id": "Pid"})
    df = df[["Uid", "Pid", "Catname", "Region", "Time"]]

    # encode Uid, Pid, Catname, Region to integers starting from 1
    uids = sorted(df["Uid"].unique())
    pids = sorted(df["Pid"].unique())
    cats = sorted(df["Catname"].unique())
    regs = sorted(df["Region"].unique())

    uids = [int(uid) for uid in uids]
    pids = [int(pid) for pid in pids]

    uid_map = {uid: i for i, uid in enumerate(uids, start=1)}
    pid_map = {pid: i for i, pid in enumerate(pids, start=1)}
    cat_map = {cat: i for i, cat in enumerate(cats, start=1)}
    reg_map = {reg: i for i, reg in enumerate(regs, start=1)}

    df["Uid"] = df["Uid"].apply(lambda x: uid_map[x])
    df["Pid"] = df["Pid"].apply(lambda x: pid_map[x])
    df["Catname"] = df["Catname"].apply(lambda x: cat_map[x])
    df["Region"] = df["Region"].apply(lambda x: reg_map[x])

    pd.DataFrame(list(uid_map.items()), columns=["Original_Uid", "Mapped_Uid"]).to_csv(
        data_path / "uid_mapping.csv", index=False
    )
    pd.DataFrame(list(pid_map.items()), columns=["Original_Pid", "Mapped_Pid"]).to_csv(
        data_path / "pid_mapping.csv", index=False
    )
    pd.DataFrame(list(cat_map.items()), columns=["Original_Catname", "Mapped_Catname"]).to_csv(
        data_path / "catname_mapping.csv", index=False
    )
    pd.DataFrame(list(reg_map.items()), columns=["Original_Region", "Mapped_Region"]).to_csv(
        data_path / "region_mapping.csv", index=False
    )

    poi_sequence = df.groupby("Uid").agg({"Pid": list, "Catname": list}).reset_index()

    poi_info = (
        df.groupby("Pid")
        .agg({"Uid": list, "Catname": lambda x: x.iloc[0], "Region": lambda x: x.iloc[0], "Time": list})
        .reset_index()
    )

    poi_info["Uid"] = poi_info["Uid"].apply(lambda uids: [uid for uid, count in Counter(uids).items() if count >= 1])
    poi_info["Time"] = poi_info["Time"].apply(
        lambda times: [time for time, count in Counter(times).items() if count >= 1]
    )

    poi_neighbors = get_neighbors(poi_sequence, "Pid", 1)
    poi_info["neighbors"] = poi_info["Pid"].map(poi_neighbors.set_index("Pid")["neighbors"])

    forward_neighbors = get_forward_neighbors(poi_sequence, "Pid", 1)
    poi_info["forward_neighbors"] = poi_info["Pid"].map(forward_neighbors.set_index("Pid")["neighbors"])
    poi_info.to_csv(data_path / "poi_info.csv", index=False)


if __name__ == "__main__":
    args = parse_args()
    main(args)
