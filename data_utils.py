from beta_stability.data.io import load_pickle
from beta_stability.algorithm_preparation import prepare_common

import beta_stability.config as config_module
from argparse import Namespace
import numpy as np


def extract_raw_from_prepared(dataset_name):
    args = Namespace(
        config=f"configs/final/{dataset_name}/beta_stability_phc100.yaml", base_config=None, save_dir=None, interactive=False, seed=None, prob_human_correct=None, overrides=None
    )
    config = config_module.apply_run_tag(config_module.load(args.config, args.base_config, args.overrides), seed=args.seed, prob_human_correct=args.prob_human_correct)

    common = prepare_common(config)
    embeddings = common["embeddings_dict"]["miewid"].embeddings
    uuids = np.array(list(common["embeddings_dict"]["miewid"].uuids.values()))
    filtered_df = common["filtered_df"].set_index("uuid").loc[uuids]

    if "name_viewpoint" in filtered_df:
        identities = filtered_df["name_viewpoint"].to_numpy()
    else:
        id_column = "individual_uuid" if "individual_uuid" in filtered_df else "individual_id"
        identities = (filtered_df[id_column].astype(str) + "_" + filtered_df["viewpoint"].astype(str)).to_numpy()

    assert len(embeddings) == len(uuids) == len(identities)
    assert len(set(uuids)) == len(uuids)
    assert np.array_equal(uuids, common["filtered_df"]["uuid"].to_numpy())

    return embeddings, uuids, identities


import json


def load_raw(dataset_name):
    if dataset_name == "GZCD":
        pickle_path = "inputs/GZCD/miewid_census_region.pickle"
        ann_path = "inputs/GZCD/reid_census_region_final.json"
    else:
        pickle_path = f"inputs/{dataset_name}/embeddings_{dataset_name}.pickle"
        ann_path = f"inputs/{dataset_name}/annotations_{dataset_name}.json"

    embs, uuids = load_pickle(pickle_path)

    with open(ann_path, "r") as f:
        loaded_ann = json.load(f)

    one_ann = loaded_ann["annotations"][0]
    ann_uuids = np.array([ann["uuid"] for ann in loaded_ann["annotations"]])
    if "name_viewpoint" in one_ann:
        identities = np.array([ann["name_viewpoint"] for ann in loaded_ann["annotations"]])
    else:
        id_entry = "individual_uuid" if "individual_uuid" in one_ann else "individual_id"
        identities = np.array([f"{ann[id_entry]}_{ann['viewpoint']}" for ann in loaded_ann["annotations"]])

    emb_dict = dict(zip(uuids, embs))
    identity_dict = dict(zip(ann_uuids, identities))
    uuid_intersection = list(set(uuids).intersection(set(identity_dict.keys())))

    filtered_identities = np.array([identity_dict[uuid] for uuid in uuid_intersection])
    filtered_embs = np.array([emb_dict[uuid] for uuid in uuid_intersection])


    return filtered_embs, uuid_intersection, filtered_identities


import torch
from graph_creation import create_graph_2hop_recip

class TopKDataset(torch.utils.data.Dataset):
    def __init__(self, embeddings, identities, k, dataset_name, is_in_train):
        self.k = k
        self.embeddings = torch.tensor(embeddings)
        self.identities = np.array(identities)
        self.dataset_name = dataset_name

        self.stacked_embs = torch.stack(list(self.embeddings))
        self.vec_norm = torch.nn.functional.normalize(self.stacked_embs, dim=1)
        self.simm = torch.mm(self.vec_norm, self.vec_norm.t())
        self.simm.fill_diagonal_(-2)
        self.ranks = self.simm.argsort(dim=1, descending=True)
        self.ranks = self.ranks[:, :-1]  # remove nodes themselves
        self.is_in_train = is_in_train

    def __len__(self):
        return len(self.identities)

    def __getitem__(self, index):
        graph = create_graph_2hop_recip(self.ranks, self.simm, index, self.k, self.identities, self.vec_norm)
        graph["dataset_name"] = self.dataset_name
        is_in_train = self.is_in_train[index]
        graph["in_train"] = is_in_train
        return graph


    def get_rank_1_stats(self):
        return self._get_rank_stats(1)

    def get_rank_5_stats(self):
        return self._get_rank_stats(5)

    def _get_rank_stats(self, rank):
        total_in_train = sum(self.is_in_train)
        correct_in_train = 0
        total_out_train = len(self) - sum(self.is_in_train)
        correct_out_train = 0


        q_id = self.identities
        g_id = q_id[self.ranks[:, :rank]]
        correct_answers = (g_id == q_id[:, None]).any(axis=1)

        for i in range(len(self)):
            if self.is_in_train[i]:
                correct_in_train += correct_answers[i].item()
            else:
                correct_out_train += correct_answers[i].item()

        return correct_in_train, correct_out_train, total_in_train, total_out_train


from pathlib import Path
import lightning
import pandas as pd
from torch_geometric.loader import DataLoader

class TopKDataModule(lightning.LightningDataModule):
    def __init__(self, dataset_names, k, num_workers, batch_size):
        super().__init__()
        self.num_workers = num_workers
        self.batch_size = batch_size

        datasets_train = []
        datasets_val = []
        datasets_test = []

        for dataset_name in dataset_names:
            embeddings, uuids, identities = load_raw(dataset_name)

            folder = Path("data/splits") / dataset_name

            train_df = pd.read_csv(folder / "train.csv")
            val_df = pd.read_csv(folder / "val.csv")
            test_df = pd.read_csv(folder / "test.csv")

            train_mask = np.isin(np.asarray(uuids), np.asarray(train_df.uuid))
            val_mask = np.isin(np.asarray(uuids), np.asarray(val_df.uuid))
            test_mask = np.isin(np.asarray(uuids), np.asarray(test_df.uuid))

            train_identities = identities[train_mask]
            is_train_identity_val = np.isin(identities[val_mask], train_identities)
            is_train_identity_test = np.isin(identities[test_mask], train_identities)

            topkdataset_train = TopKDataset(embeddings[train_mask], identities[train_mask], k, dataset_name, np.ones_like(train_mask, dtype=bool))
            topkdataset_val = TopKDataset(embeddings[val_mask], identities[val_mask], k, dataset_name, is_train_identity_val)
            topkdataset_test = TopKDataset(embeddings[test_mask], identities[test_mask], k, dataset_name, is_train_identity_test)

            datasets_train.append(topkdataset_train)
            datasets_val.append(topkdataset_val)
            datasets_test.append(topkdataset_test)

        self.concat_train = torch.utils.data.ConcatDataset(datasets_train)
        self.concat_val = torch.utils.data.ConcatDataset(datasets_val)
        self.concat_test = torch.utils.data.ConcatDataset(datasets_test)

    def train_dataloader(self):
        return DataLoader(self.concat_train, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.concat_val, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.concat_test, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)

    def get_rank_1_acc_test(self):
        total_in_cum = 0
        correct_in_cum = 0
        total_out_cum = 0
        correct_out_cum = 0
        correct_in_cum_5 = 0
        correct_out_cum_5 = 0

        test_results = {}

        for dataset in self.concat_test.datasets:
            dataset: TopKDataset
            im_corr_in_train, im_corr_out_train, im_in_train_count, im_out_train_count = dataset.get_rank_1_stats()
            im_corr_in_train_5, im_corr_out_train_5, _, _ = dataset.get_rank_5_stats()
            total_in_cum += im_in_train_count
            correct_in_cum += im_corr_in_train
            total_out_cum += im_out_train_count
            correct_out_cum += im_corr_out_train
            correct_in_cum_5 += im_corr_in_train_5
            correct_out_cum_5 += im_corr_out_train_5

            test_results[dataset.dataset_name] = {
                "Top1-acc-IT": im_corr_in_train / im_in_train_count if im_in_train_count > 0 else 0.0,
                "Top1-acc-OT": im_corr_out_train / im_out_train_count if im_out_train_count > 0 else 0.0,
                "Top1-acc": (im_corr_in_train + im_corr_out_train) / (im_in_train_count + im_out_train_count),
                "Top5-acc-IT": im_corr_in_train_5 / im_in_train_count if im_in_train_count > 0 else 0.0,
                "Top5-acc-OT": im_corr_out_train_5 / im_out_train_count if im_out_train_count > 0 else 0.0,
                "Top5-acc": (im_corr_in_train_5 + im_corr_out_train_5) / (im_in_train_count + im_out_train_count)
            }

        test_results["Cumulative"] = {
            "Top1-acc-IT": correct_in_cum / total_in_cum if total_in_cum > 0 else 0.0,
            "Top1-acc-OT": correct_out_cum / total_out_cum if total_out_cum > 0 else 0.0,
            "Top1-acc": (correct_in_cum + correct_out_cum) / (total_in_cum + total_out_cum),
            "Top5-acc-IT": correct_in_cum_5 / total_in_cum if total_in_cum > 0 else 0.0,
            "Top5-acc-OT": correct_out_cum_5 / total_out_cum if total_out_cum > 0 else 0.0,
            "Top5-acc": (correct_in_cum_5 + correct_out_cum_5) / (total_in_cum + total_out_cum)
        }


        return pd.DataFrame(test_results).T


if __name__ == "__main__":
    embs, uuids, identities = load_raw("lion")
    print(embs.shape, len(uuids), len(identities))
