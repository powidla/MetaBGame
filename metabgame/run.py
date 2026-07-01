import argparse
import os
import json
import time
import numpy as np
from scipy.stats import sem, t as t_dist
from typing import List, Tuple, Any
from tqdm import trange

from env import cmae, vagc
from train import nash, bnash
from config import get_config
from utils import load_env, load_agent
from plots import reward_curves, scaling_plots, behavior_plot
from logs import log_scaling_start, log_scaling_experiment, log_scaling_complete, log_scaling_error
from prime import gauss_prng_keys

AGENT_COUNTS = [10, 15, 18, 20]
SCALING_CONFIG = {"NUM_ENVS": 16, 
                  "NUM_STEPS": 100, "NUM_UPDATES": 1e4, "USE_IMPLICIT_GRADIENTS": False, "ACTOR_LR": 1e-4, "CRITIC_LR": 5e-4, "CLIP_EPS": 0.2, "GAE_LAMBDA": 0.95}


def make_agents(nagt: int, ecpds: list, adb: str = "agora") -> list:
    acfgs = []
    cpa = max(1, len(ecpds) // nagt)
    for i in range(nagt):
        try:
            acfgs.append(load_agent(f"agent_{i+1}", adb))
        except Exception:
            sidx = i * cpa
            eidx = len(ecpds) if i == nagt - 1 else min((i + 1) * cpa, len(ecpds))
            acfgs.append({"id": f"agent_{i}", "name": f"Agent {i}", "essential_compounds": ecpds[sidx:eidx]})
    return acfgs


def scale_config(nagt: int) -> dict:
    cfg = get_config()
    cfg.update(SCALING_CONFIG)
    return cfg


def mean_ci(data: np.ndarray, conf: float = 0.95) -> Tuple[np.ndarray, np.ndarray]:
    n = data.shape[0]
    mean = np.mean(data, axis=0)
    h = sem(data, axis=0) * t_dist.ppf((1 + conf) / 2., n - 1)
    return mean, h


def extract_traj(run: dict, nagt: int) -> np.ndarray:
    return sum(
        np.array(run["metrics"][f"episode_reward_agent_{i}"])
        for i in range(nagt)
        if f"episode_reward_agent_{i}" in run["metrics"]
    )


def run_agents(env, eprm, cfg, nrun: int = 1, verb: bool = True) -> Tuple[List[Any], List[Any]]:
    keys = gauss_prng_keys(nrun)
    nres, bres = [], []
    for i in trange(nrun, desc=f"Running {env.nagt}-agent", unit="run"):
        nres.append(nash(env, eprm, cfg, keys[i]))
        bres.append(bnash(env, eprm, cfg, keys[i]))
    return nres, bres


def run_experiment(nagt: int, ename: str, adb: str, nrun: int = 1) -> dict:
    log_scaling_experiment(nagt)

    ecfg = load_env(ename)
    acfgs = make_agents(nagt, list(ecfg["contents"].keys()), adb)
    vagc(acfgs, mina=nagt)

    env = cmae(ecfg, acfgs)
    eprm = env.dparm
    cfg = scale_config(nagt)

    t0 = time.time()
    nruns, bruns = run_agents(env=env, eprm=eprm, cfg=cfg, nrun=nrun, verb=False)
    ttot = time.time() - t0

    ntrjs = [extract_traj(r, nagt) for r in nruns]
    btrjs = [extract_traj(r, nagt) for r in bruns]
    nfin = [float(t[-1]) for t in ntrjs]
    bfin = [float(t[-1]) for t in btrjs]

    behavior_plot(
        nruns=nruns,
        bruns=bruns,
        nagt=nagt,
        supd=cfg["NUM_ENVS"] * cfg["NUM_STEPS"],
        cscl=1.0,
        spath=f"./scaling_results/behavior_{nagt}_agents.png"
    )

    return {
        "num_agents": nagt,
        "num_runs": nrun,
        "total_time_seconds": ttot,
        "time_per_run": ttot / nrun,
        "config": cfg,
        "nash_trajectory": np.array(ntrjs),
        "bayesian_trajectory": np.array(btrjs),
        "nash_final_total_rewards": nfin,
        "bayesian_final_total_rewards": bfin,
        "nash_mean": float(np.mean(nfin)),
        "nash_std": float(np.std(nfin)),
        "bayesian_mean": float(np.mean(bfin)),
        "bayesian_std": float(np.std(bfin)),
        "bayesian_advantage": float(np.mean(bfin) - np.mean(nfin)),
        "advantage_percent": float((np.mean(bfin) - np.mean(nfin)) / np.mean(nfin) * 100) if np.mean(nfin) != 0 else 0,
    }


def run_experiments(acnts: list, ename: str, adb: str, nrun: int = 1, odir: str = "./scaling_results/") -> list:
    os.makedirs(odir, exist_ok=True)
    ares = []
    for nagt in acnts:
        try:
            res = run_experiment(nagt, ename, adb, nrun)
            ares.append(res)
            ser = {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in res.items()}
            with open(f"{odir}/results_{nagt}_agents.json", 'w') as f:
                json.dump(ser, f, indent=2)
        except Exception as e:
            log_scaling_error(nagt, e)
            continue
    sera = [{k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in r.items()} for r in ares]
    with open(f"{odir}/all_scaling_results.json", 'w') as f:
        json.dump(sera, f, indent=2)
    return ares


def main():
    prs = argparse.ArgumentParser(description="Scalability Experiment")
    prs.add_argument("--env", type=str, default="model_2")
    prs.add_argument("--agent_db", type=str, default="agora")
    prs.add_argument("--num_runs", type=int, default=1)
    prs.add_argument("--output_dir", type=str, default="./scaling_results/")
    prs.add_argument("--agent_counts", nargs='+', type=int, default=AGENT_COUNTS)
    args = prs.parse_args()

    log_scaling_start(args.env, args.agent_db, args.agent_counts, args.output_dir)
    res = run_experiments(acnts=args.agent_counts, ename=args.env, adb=args.agent_db, nrun=args.num_runs, odir=args.output_dir)

    if res:
        rlist = [res] if isinstance(res, dict) else res
        supd = rlist[0]["config"]["NUM_ENVS"] * rlist[0]["config"]["NUM_STEPS"]
        scaling_plots(rlist, args.output_dir)
        reward_curves(res=rlist, acnts=args.agent_counts, supd=supd, spath=f"{args.output_dir}/reward_scaling.png")
        log_scaling_complete(args.output_dir, len(rlist), len(args.agent_counts))
    return 0 if res else 1


if __name__ == "__main__":
    exit(main())
