import logging
import pandas as pd
from typing import List

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("benchmark")


def log_bench(env_name: str, agent_db: str, counts: List[int], output_dir: str, mode: str):
    logger.info(f"SCALING BENCHMARK | mode={mode}")
    logger.info(f"env={env_name} | db={agent_db} | output={output_dir}")
    logger.info(f"counts: {counts}")


def log_agents(num_agents: int):
    logger.info(f"BENCHMARKING: {num_agents} agents")


def log_compounds(num_compounds: int):
    logger.info(f"BENCHMARKING: {num_compounds} compounds")


def log_env_result(label_key: str, label_val: int, num_envs: int, entry: dict):
    if entry['steps_per_second'] > 0:
        logger.info(f"[env_scaling] {label_key}={label_val} | envs={num_envs} | {entry['steps_per_second']:.1f} steps/s | compile={entry['compilation_time']:.2f}s | exec={entry['execution_time']:.2f}s")
    else:
        logger.warning(f"[env_scaling] {label_key}={label_val} | envs={num_envs} | FAILED")


def log_step_result(label_key: str, label_val: int, num_steps: int, entry: dict):
    if entry['steps_per_second'] > 0:
        logger.info(f"[step_scaling] {label_key}={label_val} | steps={num_steps} | {entry['steps_per_second']:.1f} steps/s | exec={entry['execution_time']:.2f}s")
    else:
        logger.warning(f"[step_scaling] {label_key}={label_val} | steps={num_steps} | FAILED")


def log_agent_summary(label_val: int, env_scaling: list):
    env_data = pd.DataFrame(env_scaling)
    valid = env_data[env_data['steps_per_second'] > 0]
    if valid.empty:
        logger.warning(f"[summary] key={label_val} | no valid runs")
        return
    max_tput = valid['steps_per_second'].max()
    best_envs = int(valid.loc[valid['steps_per_second'].idxmax(), 'num_envs'])
    logger.info(f"[summary] key={label_val} | max_throughput={max_tput:.1f} steps/s @ {best_envs} envs")


def log_benchmark_complete(output_dir: str, num_configs: int):
    logger.info("BENCHMARK COMPLETE")
    logger.info(f"configs tested: {num_configs} | results saved to: {output_dir}")


def log_scaling_start(env_name: str, agent_db: str, agent_counts: list, output_dir: str):
    logger.info("SCALABILITY EXPERIMENT")
    logger.info(f"env={env_name} | db={agent_db} | output={output_dir}")
    logger.info(f"agent counts: {agent_counts}")


def log_scaling_experiment(num_agents: int):
    logger.info(f"SCALING EXPERIMENT: {num_agents} agents")


def log_scaling_error(num_agents: int, error: Exception):
    logger.error(f"FAILED | agents={num_agents} | {type(error).__name__}: {error}")


def log_scaling_complete(output_dir: str, num_completed: int, num_total: int):
    logger.info("SCALABILITY ANALYSIS COMPLETE")
    logger.info(f"completed: {num_completed}/{num_total} | results saved to: {output_dir}")

