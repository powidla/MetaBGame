from huggingface_hub import hf_hub_download
from tqdm import trange
import json
import numpy as np
from typing import Any, List, Dict
from prime import gauss_prng_keys
from train import nash

REPO_ID = "powidla/MetabGame"

def load_env(name):
    path = hf_hub_download(REPO_ID, f"environments/{name}.json", repo_type="dataset")
    with open(path) as f:
        return json.load(f)


def load_agent(name, col):
    path = hf_hub_download(REPO_ID, f"agents/{col}/{name}.json", repo_type="dataset")
    with open(path) as f:
        return json.load(f)


def load_agents(anms: List[str], col: str) -> List[Dict]:
    return [load_agent(name, col) for name in anms]


def create_agents(bacfg: Dict, nagt: int, ecpds: List[str]) -> List[Dict]:
    cpa = max(1, len(ecpds) // nagt)
    agts = []
    for i in range(nagt):
        acfg = bacfg.copy()
        acfg['id'] = f"agent_{i}"
        acfg['name'] = f"Agent_{i}"
        sidx = i * cpa
        eidx = len(ecpds) if i == nagt - 1 else min((i + 1) * cpa, len(ecpds))
        acfg['essential_compounds'] = ecpds[sidx:eidx]
        agts.append(acfg)
    return agts


def run_agents(env, eprm, cfg, nrun: int = 5, verb: bool = True) -> List[Any]:
    keys = gauss_prng_keys(nrun)
    res = []
    for i in trange(nrun, desc=f"Running {env.nagt}-agent trials", unit="run"):
        if verb:
            print(f"\n=== Run {i + 1}/{nrun} | Agents: {env.nagt} ===")
        res.append(nash(env, eprm, cfg, keys[i]))
    return res


def run(env, eprm, cfg, nrun: int = 5, verb: bool = True) -> List[Any]:
    return run_agents(env, eprm, cfg, nrun, verb)


def stack_rewards(nres, nagt: int) -> Dict:
    ares = {}
    for aidx in range(nagt):
        key = f"episode_reward_agent_{aidx}"
        try:
            ares[f'nash_agent_{aidx}'] = np.stack([np.array(run["metrics"][key]) for run in nres])
        except KeyError:
            print(f"Warning: Could not find Nash rewards for agent {aidx}")
            ares[f'nash_agent_{aidx}'] = np.zeros((len(nres), 100))
    return ares


def stack_actions(nres, atype: str, nagt: int) -> Dict:
    ares = {}
    for aidx in range(nagt):
        key = f"agent_{aidx}_{atype}"
        try:
            ares[f'nash_agent_{aidx}_{atype}'] = np.stack([np.array(run["metrics"][key]) for run in nres])
        except KeyError:
            print(f"Warning: Could not find Nash {atype} actions for agent {aidx}")
            ares[f'nash_agent_{aidx}_{atype}'] = np.zeros((len(nres), 100))
    return ares


def agg_rewards(nres) -> Dict:
    if not nres:
        return {}
    smp = nres[0]
    nagt = smp.get("num_agents") or sum(
        1 for k in smp.get("metrics", {}) if k.startswith("episode_reward_agent_")
    ) or 2
    return stack_rewards(nres, nagt)


def agg_comp(nres) -> Dict:
    if not nres:
        return {}
    smp = nres[0]
    nagt = smp.get("num_agents") or sum(
        1 for k in smp.get("metrics", {}) if k.startswith("agent_") and k.endswith("_competitive")
    ) or 2
    return stack_actions(nres, "competitive", nagt)


def agg_coop(nres) -> Dict:
    if not nres:
        return {}
    smp = nres[0]
    nagt = smp.get("num_agents") or sum(
        1 for k in smp.get("metrics", {}) if k.startswith("agent_") and k.endswith("_cooperative")
    ) or 2
    return stack_actions(nres, "cooperative", nagt)


def export_results(res: Dict, fpath: str):
    import jax.numpy as jnp

    def convert_numpy(obj):
        if isinstance(obj, (np.ndarray, jnp.ndarray)):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif hasattr(obj, '__array__'):
            return np.array(obj).tolist()
        elif isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [convert_numpy(item) for item in obj]
        else:
            try:
                return float(obj) if isinstance(obj, (int, float)) else obj
            except (ValueError, TypeError):
                return str(obj)

    with open(fpath, 'w') as f:
        json.dump(convert_numpy(res), f, indent=2)


def load_results(fpath: str) -> Dict:
    with open(fpath, 'r') as f:
        res = json.load(f)

    def convert_lists(obj):
        if isinstance(obj, dict):
            if "metrics" in obj:
                for k, val in obj["metrics"].items():
                    if isinstance(val, list) and val and isinstance(val[0], (int, float)):
                        obj["metrics"][k] = np.array(val)
            return {k: convert_lists(v) for k, v in obj.items()}
        elif isinstance(obj, list) and obj and isinstance(obj[0], (int, float)):
            return np.array(obj)
        elif isinstance(obj, list):
            return [convert_lists(item) for item in obj]
        return obj

    return convert_lists(res)


def validate_results(res: Dict, eagt: int) -> bool:
    if "metrics" not in res:
        return False
    for i in range(eagt):
        if not any(k in res["metrics"] for k in [f"episode_reward_agent_{i}", f"episode_reward_leader_{i}", f"episode_reward_follower_{i}"]):
            print(f"Missing reward metric for agent {i}")
            return False
    return True


def agent_summary(rlist: List[Dict], nagt: int) -> Dict:
    summ = {}
    for aidx in range(nagt):
        arews = []
        for rres in rlist:
            if validate_results(rres, nagt):
                rews = next((rres["metrics"][k] for k in [f"episode_reward_agent_{aidx}", f"episode_reward_leader_{aidx}", f"episode_reward_follower_{aidx}"] if k in rres["metrics"]), None)
                if rews is not None:
                    arews.append(float(rews[-1] if hasattr(rews, '__len__') and len(rews) > 0 else rews))
        if arews:
            summ[f"agent_{aidx}"] = {"mean_final_reward": np.mean(arews), "std_final_reward": np.std(arews), "min_final_reward": np.min(arews), "max_final_reward": np.max(arews), "num_runs": len(arews)}
        else:
            print(f"Warning: No valid rewards found for agent {aidx}")
    return summ


def compound_dist(acfgs: List[Dict], ecpds: List[str]) -> Dict:
    cassn = {cpd: [] for cpd in ecpds}
    accnt = []
    for i, agt in enumerate(acfgs):
        ess = agt.get('essential_compounds', [])
        accnt.append(len(ess))
        for cpd in ess:
            if cpd in cassn:
                cassn[cpd].append(i)
    assn = sum(1 for agts in cassn.values() if agts)
    return {
        "total_compounds": len(ecpds),
        "assigned_compounds": assn,
        "unassigned_compounds": len(ecpds) - assn,
        "compound_overlap_count": sum(1 for agts in cassn.values() if len(agts) > 1),
        "agent_compound_counts": accnt,
        "mean_compounds_per_agent": np.mean(accnt),
        "std_compounds_per_agent": np.std(accnt),
        "compound_assignment": cassn,
        "distribution_balance": np.std(accnt) / np.mean(accnt) if np.mean(accnt) > 0 else 0,
    }
