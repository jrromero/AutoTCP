# J.R. Romero, A. Ramírez, A. Fuentes-Almoguera, C. García.
# "Automated machine learning for test case prioritisation".
# 2024.

# Script to prepare and process the datasets

from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from pathlib import Path
from Feature import Feature
import logging
from tqdm import tqdm

class Dataset:

    PRED_COLS = [
        "qid",
        "Q",
        "target",
        "verdict",
        "duration",
        "test",
        "build",
        "no.",
        "score",
        "indri",
    ]

    def __init__(self, path):
        self.test_count = 0
        self.feature_id_map_path = Path(path + "/id_map.csv")
        if self.feature_id_map_path.exists():
            feature_id_map_df = pd.read_csv(self.feature_id_map_path)
            keys = feature_id_map_df["key"].values.tolist()
            values = feature_id_map_df["value"].values.tolist()
            self.feature_id_map = dict(zip(keys, values))
            self.next_fid = max(values) + 1
        else:
            self.feature_id_map = {}
            self.next_fid = 1
        builds_df = pd.read_csv(
            Path(path + "/builds.csv"), parse_dates=["started_at"]
        )
        self.build_time_d = dict(
            zip(
                builds_df["id"].values.tolist(), builds_df["started_at"].values.tolist()
            )
        )

    def get_feature_id(self, feature_name):
        if feature_name not in self.feature_id_map:
            self.feature_id_map[feature_name] = self.next_fid
            self.next_fid += 1
        return self.feature_id_map[feature_name]

    def save_feature_id_map(self):
        keys = list(self.feature_id_map.keys())
        values = list(self.feature_id_map.values())
        feature_id_map_df = pd.DataFrame({"key": keys, "value": values})
        feature_id_map_df.to_csv(self.feature_id_map_path, index=False)

    def normalize_dataset(self, dataset, scaler):
        non_feature_cols = [
            Feature.BUILD,
            Feature.TEST,
            Feature.VERDICT,
            Feature.DURATION,
        ]
        feature_dataset = dataset.drop(non_feature_cols, axis=1)
        if scaler == None:
            scaler = MinMaxScaler()
            scaler.fit(feature_dataset)
        normalized_dataset = pd.DataFrame(
            scaler.transform(feature_dataset),
            columns=feature_dataset.columns,
        )
        for col in non_feature_cols:
            normalized_dataset[col] = dataset[col]

        return normalized_dataset, feature_dataset, scaler

    def convert_to_ranklib_dataset(self, dataset, scaler=None):
        if dataset.empty:
            return None
        dataset = dataset.copy()
        dataset[Feature.VERDICT] = dataset[Feature.VERDICT].apply(lambda v: int(v > 0))
        normalized_dataset, feature_dataset, _ = self.normalize_dataset(dataset, scaler)
        builds = normalized_dataset[Feature.BUILD].unique()
        ranklib_ds_rows = []
        for i, build in list(enumerate(builds)):
            build_ds = normalized_dataset[
                normalized_dataset[Feature.BUILD] == build
            ].copy()
            build_ds["B_Verdict"] = (build_ds[Feature.VERDICT] > 0).astype(int)
            build_ds.sort_values(
                ["B_Verdict", Feature.DURATION],
                ascending=[False, True],
                inplace=True,
                ignore_index=True,
            )
            build_ds.drop("B_Verdict", axis=1, inplace=True)
            build_ds["Target"] = -build_ds.index + len(build_ds)
            for _, record in build_ds.iterrows():
                row_items = [int(record["Target"]), f"qid:{i+1}"]
                row_feature_items = []
                for _, f in enumerate(feature_dataset.columns):
                    fid = self.get_feature_id(f)
                    row_feature_items.append(f"{fid}:{record[f]}")
                row_feature_items.sort(key=lambda v: int(v.split(":")[0]))
                row_items.extend(row_feature_items)
                row_items.extend(
                    [
                        "#",
                        int(record["Target"]),
                        int(record[Feature.VERDICT]),
                        int(record[Feature.DURATION]),
                        int(record[Feature.TEST]),
                        int(record[Feature.BUILD]),
                    ]
                )
                ranklib_ds_rows.append(row_items)
        headers = (
            ["target", "qid"]
            + [f"f{i+1}" for i in range(len(feature_dataset.columns))]
            + ["hashtag", "i_target", "i_verdict", "i_duration", "i_test", "i_build"]
        )
        self.save_feature_id_map()
        return pd.DataFrame(ranklib_ds_rows, columns=headers)

    def convert_to_lightGBM_dataset(self, dataset, scaler=None):
        if dataset.empty:
            return None
        dataset = dataset.copy()
        dataset[Feature.VERDICT] = dataset[Feature.VERDICT].apply(lambda v: int(v > 0))
        normalized_dataset, feature_dataset, _ = self.normalize_dataset(dataset, scaler)
        builds = normalized_dataset[Feature.BUILD].unique()
        ranklib_ds_rows = []
        for i, build in list(enumerate(builds)):
            build_ds = normalized_dataset[
                normalized_dataset[Feature.BUILD] == build
            ].copy()
            build_ds["B_Verdict"] = (build_ds[Feature.VERDICT] > 0).astype(int)
            build_ds.sort_values(
                ["B_Verdict", Feature.DURATION],
                ascending=[False, True],
                inplace=True,
                ignore_index=True,
            )
            build_ds.drop("B_Verdict", axis=1, inplace=True)
            build_ds["Target"] = -build_ds.index + len(build_ds)
            for _, record in build_ds.iterrows():
                row_items = [int(record["Target"]), f"{i+1}"]
                row_feature_items = []
                for _, f in enumerate(feature_dataset.columns):
                    fid = self.get_feature_id(f)
                    row_feature_items.append(f"{record[f]}")
                #row_feature_items.sort(key=lambda v: int(v.split(":")[0]))
                row_items.extend(row_feature_items)
                row_items.extend(
                    [
                        "#",
                        int(record["Target"]),
                        int(record[Feature.VERDICT]),
                        int(record[Feature.DURATION]),
                        int(record[Feature.TEST]),
                        int(record[Feature.BUILD]),
                    ]
                )
                ranklib_ds_rows.append(row_items)
        headers = (
            ["target", "qid"]
            + [f"f{i+1}" for i in range(len(feature_dataset.columns))]
            + ["hashtag", "i_target", "i_verdict", "i_duration", "i_test", "i_build"]
        )
        self.save_feature_id_map()
        return pd.DataFrame(ranklib_ds_rows, columns=headers)

    def create_ranklib_training_sets(
            self, ranklib_ds, output_path, custom_test_builds=None
        ):
            builds = ranklib_ds["i_build"].unique().tolist()
            builds.sort(key=lambda b: self.build_time_d[b])
            if custom_test_builds is None:
                test_builds = set(builds[-self.test_count :])
            else:
                test_builds = [b for b in custom_test_builds if b in builds]
            logging.info("Creating training sets")
            for i, build in tqdm(list(enumerate(builds)), desc="Creating training sets"):
                if build not in test_builds:
                    continue
                train_ds = ranklib_ds[ranklib_ds["i_build"].isin(builds[:i])]
                if len(train_ds) == 0:
                    continue
                test_ds = ranklib_ds[ranklib_ds["i_build"] == build]
                build_out_path = output_path / str(build)
                build_out_path.mkdir(parents=True, exist_ok=True)
                if (
                    not (output_path / str(build) / "train.csv").exists()
                    and not (output_path / str(build) / "model.csv").exists()
                ):
                    train_ds.to_csv(
                        output_path / str(build) / "train.csv",
                        sep=",",
                        header=ranklib_ds.columns,
                        index=False,
                    )
                if not (output_path / str(build) / "test.csv").exists():
                    test_ds.to_csv(
                        output_path / str(build) / "test.csv",
                        sep=",",
                        header=ranklib_ds.columns,
                        index=False,
                    )