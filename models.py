from collections import defaultdict
import torch
from torch_geometric.utils import scatter
from torch_geometric.nn.models import GAT
import lightning
from torch_geometric.data import Batch
import pandas as pd


class ListwiseLoss(torch.nn.Module):
    def __init__(self, temperature=1.0, mode="sum"):
        super().__init__()
        self.temperature = temperature
        self.mode = mode

    def forward(self, scores, batch):
        s = scores.view(-1) / self.temperature
        ptr = batch.ptr
        is_query = torch.zeros_like(batch.batch, dtype=torch.bool)
        is_query[ptr[:-1] + batch.query] = True  # query is NOT a candidate
        cand = ~is_query
        pos = cand & (batch.y.view(-1) > 0.5)
        g = batch.batch
        B = batch.num_graphs
        NEG = -1e9

        def graph_lse(x):  # per-graph logsumexp
            m = scatter(x, g, dim=0, dim_size=B, reduce="max")
            e = torch.exp(x - m[g])
            return torch.log(scatter(e, g, dim=0, dim_size=B, reduce="sum")) + m

        lse_all = graph_lse(s.masked_fill(~cand, NEG))
        has_pos = scatter(pos.float(), g, dim=0, dim_size=B, reduce="sum") > 0

        if self.mode == "sum":  # -log sum_p p_p
            lse_pos = graph_lse(s.masked_fill(~pos, NEG))
            loss = lse_all - lse_pos
        else:  # mean over positives of -log p_p
            per_node = (lse_all[g] - s) * pos
            n_pos = scatter(pos.float(), g, dim=0, dim_size=B, reduce="sum").clamp(min=1)
            loss = scatter(per_node, g, dim=0, dim_size=B, reduce="sum") / n_pos

        return loss[has_pos].mean()  # skip graphs with no positive


class IntermediateTestResults:
    def __init__(self):
        self.corr_in_train = 0
        self.in_train_count = 0
        self.corr_out_train = 0
        self.out_train_count = 0
        self.corr5_in_train = 0
        self.corr5_out_train = 0


class RerankingModel(lightning.LightningModule):
    def __init__(self, input_size, hidden_size, output_size, num_layers, batch_size, lr, heads=1, note="", **kwargs):
        super().__init__()
        self.reduction = torch.nn.Linear(input_size, hidden_size)
        self.gnn = GAT(hidden_size, hidden_size, num_layers, output_size, v2=True, edge_dim=1, heads=heads)
        self.alpha = torch.nn.Parameter(torch.tensor(0.1))
        # self.criterion = torch.nn.BCELoss(pos_weight=torch.tensor(2.2464))
        self.criterion = ListwiseLoss(mode="sum")
        self.batch_size = batch_size
        self.lr = lr

        self.save_hyperparameters()

    def forward(self, x, edge_index, edge_attr):
        sim_to_q = x[:, -2].reshape(-1, 1)
        out = self.reduction(x)
        out = self.gnn(out, edge_index, edge_attr)
        # return sim_to_q + out
        return sim_to_q + self.alpha * out

    def training_step(self, batch: Batch, batch_idx):
        node_out = self.forward(batch.x, batch.edge_index, batch.edge_attr)
        loss = self.criterion(node_out, batch)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=self.batch_size)
        return loss

    def validation_step(self, batch: Batch, batch_idx):
        node_out = self.forward(batch.x, batch.edge_index, batch.edge_attr)
        loss = self.criterion(node_out, batch)

        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=False, logger=True, batch_size=self.batch_size)

        nodes_out_start = 0
        corr_in_train = 0
        in_train_count = 0
        corr_out_train = 0
        out_train_count = 0
        corr5_in_train = 0
        corr5_out_train = 0
        for i in range(batch.num_graphs):
            graph = batch.get_example(i)
            pred = node_out[nodes_out_start : nodes_out_start + graph.x.shape[0]]
            nodes_out_start += graph.x.shape[0]
            # top_pred = torch.argmax(pred[1:])
            # correct = bool(graph.y[top_pred+1].item())

            query_idx = graph.query
            candidate_mask = torch.ones(graph.x.shape[0], dtype=torch.bool, device=node_out.device)
            candidate_mask[query_idx] = False

            candidate_scores = pred[candidate_mask]
            candidate_labels = graph.y[candidate_mask]

            top_pred = candidate_scores.argmax()
            correct = bool(candidate_labels[top_pred].item())
            top5_pred = candidate_scores.ravel().topk(min(5, candidate_scores.numel())).indices
            correct_top5 = bool(candidate_labels[top5_pred].any().item())

            if graph.in_train:
                in_train_count += 1
                if correct:
                    corr_in_train += 1
                if correct_top5:
                    corr5_in_train += 1
            else:
                out_train_count += 1
                if correct:
                    corr_out_train += 1
                if correct_top5:
                    corr5_out_train += 1

        if in_train_count > 0:
            in_train_acc = corr_in_train / in_train_count
            self.log("top1_acc_IT", in_train_acc, on_step=True, on_epoch=True, prog_bar=False, logger=True, batch_size=self.batch_size)
        if out_train_count > 0:
            out_train_acc = corr_out_train / out_train_count
            self.log("top1_acc_OT", out_train_acc, on_step=True, on_epoch=True, prog_bar=False, logger=True, batch_size=self.batch_size)
        if in_train_count > 0:
            self.log("top5_acc_IT", corr5_in_train / in_train_count, on_step=True, on_epoch=True, prog_bar=False, logger=True, batch_size=self.batch_size)
        if out_train_count > 0:
            self.log("top5_acc_OT", corr5_out_train / out_train_count, on_step=True, on_epoch=True, prog_bar=False, logger=True, batch_size=self.batch_size)

        top1acc = (corr_in_train + corr_out_train) / (in_train_count + out_train_count)
        self.log("top1_acc", top1acc, on_step=True, on_epoch=True, prog_bar=False, logger=True, batch_size=self.batch_size)
        top5acc = (corr5_in_train + corr5_out_train) / (in_train_count + out_train_count)
        self.log("top5_acc", top5acc, on_step=True, on_epoch=True, prog_bar=False, logger=True, batch_size=self.batch_size)

    def on_test_epoch_start(self):
        self.im_test_results = defaultdict(IntermediateTestResults)

    def on_test_epoch_end(self):
        # self.top1acc_in_train = self.corr_in_train / self.in_train_count if self.in_train_count > 0 else 0.0
        # self.top1acc_out_train = self.corr_out_train / self.out_train_count if self.out_train_count > 0 else 0.0
        # self.top1acc = (self.corr_in_train + self.corr_out_train) / (self.in_train_count + self.out_train_count)
        self.test_results = {}
        for dataset_name in self.im_test_results:
            im_results = self.im_test_results[dataset_name]
            self.test_results[dataset_name] = {
                "Top1-acc-IT": im_results.corr_in_train / im_results.in_train_count if im_results.in_train_count > 0 else 0.0,
                "Top1-acc-OT": im_results.corr_out_train / im_results.out_train_count if im_results.out_train_count > 0 else 0.0,
                "Top1-acc": (im_results.corr_in_train + im_results.corr_out_train) / (im_results.in_train_count + im_results.out_train_count),
                "Top5-acc-IT": im_results.corr5_in_train / im_results.in_train_count if im_results.in_train_count > 0 else 0.0,
                "Top5-acc-OT": im_results.corr5_out_train / im_results.out_train_count if im_results.out_train_count > 0 else 0.0,
                "Top5-acc": (im_results.corr5_in_train + im_results.corr5_out_train) / (im_results.in_train_count + im_results.out_train_count),
            }

        total_samples_in = sum(im_results.in_train_count for im_results in self.im_test_results.values())
        total_samples_out = sum(im_results.out_train_count for im_results in self.im_test_results.values())
        self.test_results["Cumulative"] = {
            "Top1-acc-IT": sum(im_results.corr_in_train for im_results in self.im_test_results.values()) / total_samples_in if total_samples_in > 0 else 0.0,
            "Top1-acc-OT": sum(im_results.corr_out_train for im_results in self.im_test_results.values()) / total_samples_out if total_samples_out > 0 else 0.0,
            "Top1-acc": (
                sum(im_results.corr_in_train for im_results in self.im_test_results.values())
                + sum(im_results.corr_out_train for im_results in self.im_test_results.values())
            )
            / (total_samples_in + total_samples_out),
            "Top5-acc-IT": sum(im_results.corr5_in_train for im_results in self.im_test_results.values()) / total_samples_in if total_samples_in > 0 else 0.0,
            "Top5-acc-OT": sum(im_results.corr5_out_train for im_results in self.im_test_results.values()) / total_samples_out if total_samples_out > 0 else 0.0,
            "Top5-acc": (
                sum(im_results.corr5_in_train for im_results in self.im_test_results.values())
                + sum(im_results.corr5_out_train for im_results in self.im_test_results.values())
            )
            / (total_samples_in + total_samples_out),
        }
        self.test_results = pd.DataFrame(self.test_results).T

    def get_test_results(self):
        return self.test_results

    def test_step(self, batch: Batch, batch_idx):
        node_out = self.forward(batch.x, batch.edge_index, batch.edge_attr)

        nodes_out_start = 0
        for i in range(batch.num_graphs):
            graph = batch.get_example(i)
            pred = node_out[nodes_out_start : nodes_out_start + graph.x.shape[0]]
            nodes_out_start += graph.x.shape[0]

            query_idx = graph.query
            candidate_mask = torch.ones(graph.x.shape[0], dtype=torch.bool, device=node_out.device)
            candidate_mask[query_idx] = False

            candidate_scores = pred[candidate_mask]
            candidate_labels = graph.y[candidate_mask]

            top_pred = candidate_scores.argmax()
            correct = bool(candidate_labels[top_pred].item())
            top5_pred = candidate_scores.ravel().topk(min(5, candidate_scores.numel())).indices
            correct_top5 = bool(candidate_labels[top5_pred].any().item())

            if graph.in_train:
                self.im_test_results[graph.dataset_name].in_train_count += 1
                if correct:
                    self.im_test_results[graph.dataset_name].corr_in_train += 1
                if correct_top5:
                    self.im_test_results[graph.dataset_name].corr5_in_train += 1
            else:
                self.im_test_results[graph.dataset_name].out_train_count += 1
                if correct:
                    self.im_test_results[graph.dataset_name].corr_out_train += 1
                if correct_top5:
                    self.im_test_results[graph.dataset_name].corr5_out_train += 1

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        return optimizer

