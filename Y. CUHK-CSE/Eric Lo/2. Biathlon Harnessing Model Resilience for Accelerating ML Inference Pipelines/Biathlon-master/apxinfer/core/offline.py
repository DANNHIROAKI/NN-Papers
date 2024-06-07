import os
import numpy as np
import pandas as pd
from typing import List
from tqdm import tqdm
import logging
import pickle
import joblib
import json

from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from apxinfer.core.utils import XIPExecutionProfile, XIPQType
from apxinfer.core.fengine import XIPFEngine as XIPFeatureExtractor
from apxinfer.core.qcost import QueryCostModel, XIPQCostModel
from apxinfer.core.model import XIPModel


class OfflineExecutor:
    def __init__(
        self,
        working_dir: str,
        fextractor: XIPFeatureExtractor,
        model: XIPModel,
        nparts: int,
        ncfgs: int,
        verbose: bool = False,
    ) -> None:
        self.fextractor: XIPFeatureExtractor = fextractor
        self.model = model
        self.nparts = nparts
        self.ncfgs = ncfgs
        self.verbose = verbose
        self.working_dir = working_dir
        self.model_dir = os.path.join(self.working_dir, "model")
        os.makedirs(self.model_dir, exist_ok=True)

        self.logger = logging.getLogger("OfflineExecutor")
        if self.verbose:
            self.logger.setLevel(logging.DEBUG)

        self.agg_qids = []
        self.agg_fids = []
        self.agg_fnames = []
        self.agg_fid2qid = {}
        cnt = 0
        for qid, qry in enumerate(self.fextractor.queries):
            if qry.qtype == XIPQType.AGG:
                self.agg_qids.append(qid)
                for i in range(len(qry.fnames)):
                    self.agg_fids.append(cnt + i)
                    self.agg_fnames.append(qry.fnames[i])
                    self.agg_fid2qid[cnt + i] = qid
            cnt += len(qry.fnames)

    def preprocess(self, dataset: pd.DataFrame) -> dict:
        cols = dataset.columns.tolist()
        req_cols = [col for col in cols if col.startswith("req_")]
        fcols = [col for col in cols if col.startswith("f_")]
        label_col = "label"

        requests = dataset[req_cols].to_dict(orient="records")
        labels = dataset[label_col].to_numpy()
        ext_features = dataset[fcols].to_numpy()

        return {
            "requests": requests,
            "labels": labels,
            "ext_features": ext_features,
        }

    def collect(self, requests: pd.DataFrame) -> List[List[XIPExecutionProfile]]:
        self.fextractor.set_exec_mode("sequential")
        records = []
        for i, request in tqdm(
            enumerate(requests),
            desc="Serving requests",
            total=len(requests),
            disable=self.verbose,
        ):
            self.logger.debug(f"request[{i}] = {request}")
            profiles: List[XIPExecutionProfile] = []
            for cfg_id in tqdm(
                range(1, self.ncfgs + 1, 1),
                desc="Serving configurations",
                leave=False,
                disable=self.verbose,
            ):
                qcfgs = []
                for qid, qry in enumerate(self.fextractor.queries):
                    if qry.qtype == XIPQType.AGG:
                        sample = cfg_id * 1.0 / self.ncfgs
                    else:
                        sample = 1.0
                    qcfgs.append(qry.get_qcfg(cfg_id=cfg_id, sample=sample))
                try:
                    fvec, qcosts = self.fextractor.extract(
                        request, qcfgs
                    )
                except TypeError:
                    fvec, qcosts = self.fextractor.extract(request, qcfgs)
                if len(profiles) > 0:
                    for i in range(len(qcosts)):
                        qcosts[i]["time"] += profiles[-1]["qcosts"][i]["time"]
                profiles.append(
                    XIPExecutionProfile(
                        request=request, qcfgs=qcfgs, fvec=fvec, qcosts=qcosts
                    )
                )
            records.append(profiles)
        return records

    def plot_fvars(self, qsamples: np.ndarray, fvars: np.ndarray) -> None:
        fig_height = min(1<<16, 6 * len(self.agg_fids))
        fig, axes = plt.subplots(
            len(self.agg_fids), 1, figsize=(8, fig_height)
        )
        if len(self.agg_fids) == 1:
            axes = [axes]
        for k in range(len(self.agg_fids)):
            ax: plt.Axes = axes[k]
            fname = self.agg_fnames[k]
            for i in range(len(fvars)):
                ax.plot(qsamples[i, :, k], np.log(fvars[i, :, k]))
            ax.set_title(fname)
        plt.savefig(f"{self.working_dir}/fvars.pdf", bbox_inches="tight")

    def plot_qcost(self, features: pd.DataFrame):
        fig_height = min(1<<16, 6 * len(self.agg_fids))
        fig, axes = plt.subplots(
            len(self.agg_qids), 1, figsize=(8, fig_height)
        )
        if len(self.agg_qids) == 1:
            axes = [axes]
        for k, qid in enumerate(self.agg_qids):
            ax: plt.Axes = axes[k]
            qname = self.fextractor.qnames[qid]
            gf = features[features["qid_idx"] == qid]
            ax.scatter(gf["qsample"], gf["tcost"], alpha=0.7)
            ax.set_title(qname)
            ax.set_xlabel("qsample")
            ax.set_ylabel("tcost")
        plt.savefig(f"{self.working_dir}/qtcosts.pdf", bbox_inches="tight")

    def build_cost_model(self, records: List[List[XIPExecutionProfile]]):
        models = []
        for _, qid in enumerate(self.agg_qids):
            qname = self.fextractor.queries[qid].qname
            qcfgs = []
            qcosts = []
            for i, profiles in enumerate(records):
                for j, profile in enumerate(profiles):
                    qcfgs.append(profile["qcfgs"][qid])
                    qcosts.append(profile["qcosts"][qid])
            np.random.seed(0)
            model = QueryCostModel(qname)
            X_train, X_test, y_train, y_test = train_test_split(qcfgs, qcosts)
            model.fit(None, X_train, y_train)
            evals = model.evaluate(None, X_test, y_test)
            slope = model.get_weight()
            evals["slope"] = slope
            print(f"q-{qid} {qname}, slope={slope}")
            print(f"mae={evals['mae']}, mape={evals['mape']}")
            model_tag = f"q-{qname}"
            joblib.dump(model, os.path.join(self.model_dir, f"{model_tag}.pkl"))
            with open(
                os.path.join(self.model_dir, f"{model_tag}_evals.json"), "w"
            ) as f:
                json.dump(evals, f, indent=4)
            models.append(model)
        xip_qcm = XIPQCostModel(models)
        joblib.dump(xip_qcm, os.path.join(self.model_dir, "xip_qcm.pkl"))

    def get_initial_plan(self, results: dict, records: List[List[XIPExecutionProfile]]):
        labels = results["labels"]
        exact_fvals = results["ext_features"]

        # each query have #ncfgs versions, we need to enumerate all plans
        # a plan indicate the version of each query
        # for each plan, we need to evaluate the cost and the prediction error
        # we need to find the best plan
        if len(self.agg_qids) > 5:
            nversions = min(2, self.ncfgs)
        elif len(self.agg_qids) > 3:
            nversions = min(5, self.nparts)
        else:
            nversions = self.ncfgs
        if np.power(nversions, len(self.agg_qids)) > 1000:
            if nversions > 10:
                nversions = 10
            if np.power(nversions, len(self.agg_qids)) > 1000:
                print(
                    f"Warning: too many plans, {np.power(nversions, len(self.agg_qids))}, skip"
                )
                return

        all_plans = np.array(
            np.meshgrid(*[np.arange(nversions) for _ in range(len(self.agg_qids))])
        ).T.reshape(-1, len(self.agg_qids))
        all_plans = all_plans * (self.ncfgs // nversions) + (self.ncfgs % nversions)

        all_preds = []
        all_costs = []
        all_pvars = []
        for plan in all_plans:
            # get the features and costs for this plan
            plan_fvals = np.zeros((len(records), len(self.fextractor.fnames)))
            plan_fests = np.zeros((len(records), len(self.fextractor.fnames)))
            plan_tcosts = np.zeros((len(records), len(self.agg_qids)))
            for i, profiles in enumerate(records):
                for fid in range(len(self.fextractor.fnames)):
                    if fid in self.agg_fids:
                        qid = self.agg_fid2qid[fid]
                        # print(f'fid={fid}, qid={qid}, {self.agg_fids}, {self.agg_fid2qid}')
                        j = self.agg_qids.index(qid)
                        plan_fvals[i][fid] = profiles[plan[j]]["fvec"]["fvals"][fid]
                        plan_fests[i][fid] = profiles[plan[j]]["fvec"]["fests"][fid]
                    else:
                        plan_fvals[i][fid] = exact_fvals[i][fid]
                        plan_fests[i][fid] = 0.0
                for j in range(len(self.agg_qids)):
                    plan_tcosts[i][j] = profiles[plan[j]]["qcosts"][qid]["time"]
            # get costs for this plan
            costs = np.sum(plan_tcosts, axis=1)
            all_costs.append(np.mean(costs))
            # get predictions for this plan
            preds = self.model.predict(plan_fvals)
            all_preds.append(preds)
            # get prediction variance for this plan
            from apxinfer.core.festimator import get_feature_samples

            n_samples = 100
            seed = 0
            pvars = np.zeros(len(records))
            for i in range(len(records)):
                fvals = plan_fvals[i]
                fests = plan_fests[i]
                p = len(fvals)
                samples = np.zeros((n_samples, p))
                for j in range(p):
                    samples[:, j] = get_feature_samples(
                        fvals[j], "normal", fests[j], seed + j, n_samples
                    )
                pvars[i] = np.var(self.model.predict(samples))
            all_pvars.append(pvars)

        all_preds = np.array(all_preds)
        all_costs = np.array(all_costs)

        all_mses = np.array([np.mean((labels - preds) ** 2) for preds in all_preds])

        exact_preds = self.model.predict(exact_fvals)
        all_mses_sim = np.array(
            [np.mean((exact_preds - preds) ** 2) for preds in all_preds]
        )
        # print(f'all_plans: {all_plans}')
        print(f"all_costs: {all_costs}")
        print(f"all_mses: {all_mses}")
        print(f"all_mses_sim: {all_mses_sim}")

        # plot scatter (cost, mse)
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(all_costs, all_mses, alpha=0.7)
        ax.set_xlabel("cost")
        ax.set_ylabel("mse")
        plt.savefig(f"{self.working_dir}/cost_mse.pdf", bbox_inches="tight")

        # plot scatter (cost, mse_sim)
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(all_costs, all_mses_sim, alpha=0.7)
        ax.set_xlabel("cost")
        ax.set_ylabel("mse_sim")
        if len(all_plans) < 100:
            # plot the plans
            for i, plan in enumerate(all_plans):
                ax.annotate(
                    str(plan),
                    (all_costs[i], all_mses_sim[i]),
                    textcoords="offset points",
                    xytext=(0, 10),
                    ha="center",
                )
        plt.savefig(f"{self.working_dir}/cost_mse_sim.pdf", bbox_inches="tight")

        # find the plans whos mses is less or equal than mses[-1]
        plan_ids = np.where(all_mses <= all_mses[-1])[0]
        print(f"plan_ids: {plan_ids}")
        # find the plan with minimal cost in plan_ids
        best_plan_id = plan_ids[np.argmin(all_costs[plan_ids])]
        best_plan = all_plans[best_plan_id]
        print(f"best_plan_id: {best_plan_id}, best_plan={best_plan}")
        print(f"best_cost: {all_costs[best_plan_id]}")
        print(f"best_mse: {all_mses[best_plan_id]}")
        print(f"best_mse_sim: {all_mses_sim[best_plan_id]}")

        # plot the cdf of pvars, and mark the best_plan
        fig, ax = plt.subplots(figsize=(8, 6))
        for i in range(len(all_pvars)):
            pvars = all_pvars[i]
            if i == best_plan_id:
                ax.plot(
                    np.sort(pvars), np.linspace(0, 1, len(pvars)), label="best_plan"
                )
            else:
                ax.plot(np.sort(pvars), np.linspace(0, 1, len(pvars)), alpha=0.1)
        ax.set_xlabel("pvar")
        ax.set_ylabel("cdf")
        ax.legend()
        plt.savefig(f"{self.working_dir}/pvar_cdf_all.pdf", bbox_inches="tight")

        best_pvars = all_pvars[best_plan_id]
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.hist(
            best_pvars,
            bins=100,
            density=True,
            cumulative=True,
            histtype="step",
            label="cdf",
        )
        ax.set_xlabel("pvar")
        ax.set_ylabel("cdf")
        plt.savefig(f"{self.working_dir}/pvar_cdf.pdf", bbox_inches="tight")
        # save best_pvars, best_preds, exact_preds, labels to csv
        df = pd.DataFrame(
            np.vstack((best_pvars, all_preds[best_plan_id], exact_preds, labels)).T,
            columns=["pvar", "pred", "exact", "label"],
        )
        df["pdiff"] = df["pred"] - df["exact"]
        df["is_label"] = df["pred"] == df["label"]
        df["is_exact"] = df["pred"] == df["exact"]
        df.to_csv(os.path.join(self.working_dir, "pvar_preds.csv"), index=False)
        # plot pdiff v.s. pvar and pdiff^2 v.s. pvar
        fig, axes = plt.subplots(2, 1, figsize=(8, 12))
        axes[0].scatter(df["pvar"], df["pdiff"], alpha=0.7)
        axes[0].set_xlabel("pvar")
        axes[0].set_ylabel("pdiff")
        axes[1].scatter(df["pvar"], np.power(df["pdiff"], 2), alpha=0.7)
        axes[1].set_xlabel("pvar")
        axes[1].set_ylabel("abs(pdiff)")
        plt.savefig(f"{self.working_dir}/pvar_pdiff.pdf", bbox_inches="tight")

    def run(self, dataset: pd.DataFrame, clear_cache: bool = False) -> dict:
        self.logger.info("Running offline executor")
        results = self.preprocess(dataset)
        records_path = f"{self.working_dir}/records.pkl"
        if os.path.exists(records_path) and not clear_cache:
            with open(records_path, "rb") as f:
                records = pickle.load(f)
        else:
            records = self.collect(requests=results["requests"])
            with open(records_path, "wb") as f:
                pickle.dump(records, f)

        qsamples = np.zeros((len(records), self.ncfgs, len(self.agg_fids)))
        fvars = np.zeros((len(records), self.ncfgs, len(self.agg_fids)))
        tcosts = np.zeros((len(records), self.ncfgs, len(self.agg_qids)))
        for i, profiles in enumerate(records):
            for j, profile in enumerate(profiles):
                for k, fid in enumerate(self.agg_fids):
                    qsample = profile["qcfgs"][self.agg_fid2qid[fid]]["qsample"]
                    fvar = profile["fvec"]["fests"][fid]
                    qsamples[i][j][k] = qsample
                    fvars[i][j][k] = fvar
                for k, qid in enumerate(self.agg_qids):
                    tcosts[i][j][k] = profile["qcosts"][qid]["time"]
        self.logger.debug(f"qsamples={qsamples[0]}")
        self.logger.debug(f"fvars={fvars[0]}")
        # plot the fvars
        self.plot_fvars(qsamples, fvars)

        self.build_cost_model(records)
        self.logger.info(f"model saved to {self.model_dir}")

        # Convert the array to a pandas DataFrame
        df = pd.DataFrame(
            tcosts.reshape(-1, 1),
            columns=["tcost"],
        )
        # Add two new columns with the cfg and qid indices
        df["req_idx"] = np.repeat(
            np.arange(tcosts.shape[0]), tcosts.shape[1] * tcosts.shape[2]
        )
        df["cfg_idx"] = np.tile(
            np.repeat(np.arange(tcosts.shape[1]), tcosts.shape[2]), tcosts.shape[0]
        )
        df["qid_idx"] = np.tile(
            np.array(self.agg_qids), tcosts.shape[0] * tcosts.shape[1]
        )
        df["qsample"] = df["cfg_idx"] * 1.0 / self.ncfgs
        df.to_csv(os.path.join(self.model_dir, "features.csv"), index=False)

        self.plot_qcost(df)

        # self.get_initial_plan(results, records)

        return None
