import jax
import jax.numpy as jnp
import numpy as np
from typing import NamedTuple, Union, Tuple, List, Optional, Dict
import time
import chex
from flax import struct
from functools import partial
from gymnax.environments import environment, spaces
from brax import envs
from brax.envs.wrappers.training import EpisodeWrapper, AutoResetWrapper

GK = jax.random.PRNGKey(0)

class EnvParams(NamedTuple):
    elen: int = 10
    cmode: bool = True
    alpha: float = 1.0
    beta: float = 0.0
    gamma: float = 0.0
    delta: float = 0.001
    mconc: float = 1.0
    alo: float = -0.1
    ahi: float = 0.1


class EnvState(NamedTuple):
    cpds: jax.Array
    step: int
    key: jax.Array


class Observation(NamedTuple):
    obs: chex.Array


class MetabGame:
    def __init__(self, ecfg: Dict, acfgs: List[Dict], magts: Optional[int] = None):
        self.ecfg = ecfg
        self.acfgs = acfgs
        self.cnms = list(self.ecfg["contents"].keys())
        self.ncpd = len(self.cnms)
        self.c2i = {name: idx for idx, name in enumerate(self.cnms)}
        self.amsk = self._mkmsk()
        self.nagt = len(self.acfgs)
        self.magt = magts or self.nagt

        if self.nagt > self.magt:
            raise ValueError(f"Number of agents ({self.nagt}) exceeds maximum ({self.magt})")
        if self.nagt < 2:
            raise ValueError(f"Must have at least 2 agents, got {self.nagt}")

        print(f"Environment initialized with {self.ncpd} compounds and {self.nagt} agents")
        print(f"Compounds: {self.cnms[:5]}..." if len(self.cnms) > 5 else f"Compounds: {self.cnms}")

    def _mkmsk(self) -> jax.Array:
        msks = []
        for acfg in self.acfgs:
            msk = jnp.zeros(self.ncpd)
            ecmp = acfg.get("essential_compounds", [])
            eidx = [self.c2i[c] for c in ecmp if c in self.c2i]
            if eidx:
                msk = msk.at[jnp.array(eidx)].set(1.0)
            msks.append(msk)
            print(f"Agent {acfg['id']}: {len(eidx)} essential compounds")
        return jnp.array(msks)

    @property
    def dparm(self) -> EnvParams:
        return EnvParams()

    def reset(self, key: chex.PRNGKey, prm: EnvParams) -> Tuple[EnvState, Observation]:
        key = key if key is not None else GK
        iconc = jnp.array([
            self.ecfg["contents"].get(c, 0.0)
            for c in self.cnms
        ]) * 1000.0
        st = EnvState(cpds=iconc, step=0, key=key)
        return st, Observation(obs=st.cpds / prm.mconc)

    def step(self, st: EnvState, acts: Tuple[chex.Array, ...], prm: EnvParams) -> Tuple[EnvState, Observation, Tuple[float, ...], bool, dict]:
        if len(acts) != self.nagt:
            raise ValueError(f"Expected {self.nagt} actions, got {len(acts)}")
        cact = [jnp.clip(a, prm.alo, prm.ahi) for a in acts]
        nstep = st.step + 1
        tact = jnp.sum(jnp.stack(cact), axis=0)
        ncpds = jnp.clip(st.cpds * (1 + tact), 0.0, prm.mconc)
        rews = tuple(self._rew(cact[i], cact, i, ncpds, prm) for i in range(self.nagt))
        done = nstep >= prm.elen
        nst = EnvState(cpds=ncpds, step=nstep, key=st.key)
        return nst, Observation(obs=ncpds / prm.mconc), rews, done, {}

    def _rew(self, oact: chex.Array, aacts: List[chex.Array], aidx: int, cpds: chex.Array, prm: EnvParams) -> float:
        emsk = self.amsk[aidx]
        crew = prm.alpha * jnp.sum(jnp.maximum(-oact * emsk, 0))
        prew = sum(
            prm.beta * jnp.sum(jnp.maximum(aacts[oidx] * emsk, 0))
            for oidx in range(self.nagt) if oidx != aidx
        )
        pen = prm.delta * jnp.sum(jnp.square(oact))
        return crew + prew + prm.gamma - pen

    def obssh(self) -> Tuple[int, ...]:
        return (self.ncpd,)

    def actsh(self) -> Tuple[int, ...]:
        return (self.ncpd,)

    def isterm(self, st: EnvState, prm: EnvParams) -> bool:
        return st.step >= prm.elen

    def cinfo(self) -> Dict:
        return {
            "num_compounds": self.ncpd,
            "num_agents": self.nagt,
            "compound_names": self.cnms,
            "agent_names": [cfg["name"] for cfg in self.acfgs],
            "essential_compounds_per_agent": [len(cfg.get("essential_compounds", [])) for cfg in self.acfgs],
        }

    def ainfo(self) -> Dict:
        return {
            "num_agents": self.nagt,
            "max_agents": self.magt,
            "agent_names": [cfg["name"] for cfg in self.acfgs],
            "agent_ids": [cfg["id"] for cfg in self.acfgs],
            "essential_compounds_per_agent": [len(cfg.get("essential_compounds", [])) for cfg in self.acfgs],
            "agent_essential_masks": self.amsk,
        }


def rspe(keys: chex.Array, env: MetabGame, eprm: EnvParams, nenv: int) -> Tuple[EnvState, Observation]:
    return jax.vmap(lambda key: env.reset(key, eprm))(keys)


def sspe(sts, acts, env, eprm, nenv):
    def _ss(st, aset):
        return env.step(st, tuple(aset[i] for i in range(env.nagt)), eprm)
    return jax.vmap(_ss)(sts, jnp.stack(acts, axis=1))


def capmc(acts: Tuple[jnp.ndarray, ...], env, nenv: int) -> Dict[str, jnp.ndarray]:
    acls = {}
    for aidx in range(len(acts)):
        aact = acts[aidx]
        emsk = env.amsk[aidx]
        acls[f'agent_{aidx}_competitive'] = jnp.sum(jnp.maximum(-aact * emsk[None, :], 0.0), axis=1)
        acls[f'agent_{aidx}_cooperative'] = sum(
            jnp.sum(jnp.maximum(aact * env.amsk[oth][None, :], 0.0), axis=1)
            for oth in range(len(acts)) if oth != aidx
        )
    return acls


def cmae(ecfg: Dict, acfgs: List[Dict]) -> MetabGame:
    return MetabGame(ecfg, acfgs, magts=len(acfgs))


def vagc(acfgs: List[Dict], mina: int = 2) -> bool:
    if len(acfgs) < mina:
        raise ValueError(f"At least {mina} agents required, got {len(acfgs)}")
    for i, cfg in enumerate(acfgs):
        for fld in ['id', 'name']:
            if fld not in cfg:
                raise ValueError(f"Agent {i} missing required field: {fld}")
    ids = [cfg['id'] for cfg in acfgs]
    if len(set(ids)) != len(ids):
        raise ValueError("Agent IDs must be unique")
    for i, cfg in enumerate(acfgs):
        if 'essential_compounds' in cfg and not isinstance(cfg['essential_compounds'], list):
            raise ValueError(f"Agent {i}: essential_compounds must be a list")
    return True


def csya(bacfg: Dict, nagt: int, ecpds: List[str]) -> List[Dict]:
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


class GymnaxWrapper(object):
    def __init__(self, env):
        self._env = env

    def __getattr__(self, name):
        return getattr(self._env, name)


class FlattenObservationWrapper(GymnaxWrapper):
    def __init__(self, env: environment.Environment):
        super().__init__(env)

    def observation_space(self, prm) -> spaces.Box:
        return spaces.Box(low=self._env.observation_space(prm).low, high=self._env.observation_space(prm).high, shape=(np.prod(self._env.observation_space(prm).shape),), dtype=self._env.observation_space(prm).dtype)

    @partial(jax.jit, static_argnums=(0,))
    def reset(self, key: chex.PRNGKey, prm: Optional[environment.EnvParams] = None) -> Tuple[chex.Array, environment.EnvState]:
        obs, st = self._env.reset(key, prm)
        return jnp.reshape(obs, (-1,)), st

    @partial(jax.jit, static_argnums=(0,))
    def step(self, key: chex.PRNGKey, st: environment.EnvState, act: Union[int, float], prm: Optional[environment.EnvParams] = None) -> Tuple[chex.Array, environment.EnvState, float, bool, dict]:
        obs, st, rew, done, info = self._env.step(key, st, act, prm)
        return jnp.reshape(obs, (-1,)), st, rew, done, info


@struct.dataclass
class LogEnvState:
    est: environment.EnvState
    eret: float
    elns: int
    rret: float
    rlen: int
    tstp: int


class LogWrapper(GymnaxWrapper):
    def __init__(self, env: environment.Environment):
        super().__init__(env)

    @partial(jax.jit, static_argnums=(0,))
    def reset(self, key: chex.PRNGKey, prm: Optional[environment.EnvParams] = None) -> Tuple[chex.Array, environment.EnvState]:
        obs, est = self._env.reset(key, prm)
        return obs, LogEnvState(est, 0, 0, 0, 0, 0)

    @partial(jax.jit, static_argnums=(0,))
    def step(self, key: chex.PRNGKey, st: environment.EnvState, act: Union[int, float], prm: Optional[environment.EnvParams] = None) -> Tuple[chex.Array, environment.EnvState, float, bool, dict]:
        obs, est, rew, done, info = self._env.step(key, st.est, act, prm)
        nret = st.eret + rew
        nlen = st.elns + 1
        st = LogEnvState(
            est=est,
            eret=nret * (1 - done),
            elns=nlen * (1 - done),
            rret=st.rret * (1 - done) + nret * done,
            rlen=st.rlen * (1 - done) + nlen * done,
            tstp=st.tstp + 1,
        )
        info["returned_episode_returns"] = st.rret
        info["returned_episode_lengths"] = st.rlen
        info["timestep"] = st.tstp
        info["returned_episode"] = done
        return obs, st, rew, done, info


class BraxGymnaxWrapper:
    def __init__(self, ename, bkd="positional"):
        env = envs.get_environment(env_name=ename, backend=bkd)
        env = EpisodeWrapper(env, episode_length=1000, action_repeat=1)
        self._env = AutoResetWrapper(env)
        self.asz = env.action_size
        self.osz = (env.observation_size,)

    def reset(self, key, prm=None):
        st = self._env.reset(key)
        return st.obs, st

    def step(self, key, st, act, prm=None):
        nst = self._env.step(st, act)
        return nst.obs, nst, nst.reward, nst.done > 0.5, {}

    def observation_space(self, prm):
        return spaces.Box(low=-jnp.inf, high=jnp.inf, shape=(self._env.observation_size,))

    def action_space(self, prm):
        return spaces.Box(low=-1.0, high=1.0, shape=(self._env.action_size,))


class ClipAction(GymnaxWrapper):
    def __init__(self, env, lo=-1.0, hi=1.0):
        super().__init__(env)
        self.lo = lo
        self.hi = hi

    def step(self, key, st, act, prm=None):
        return self._env.step(key, st, jnp.clip(act, self.lo, self.hi), prm)


class TransformObservation(GymnaxWrapper):
    def __init__(self, env, tobs):
        super().__init__(env)
        self.tobs = tobs

    def reset(self, key, prm=None):
        obs, st = self._env.reset(key, prm)
        return self.tobs(obs), st

    def step(self, key, st, act, prm=None):
        obs, st, rew, done, info = self._env.step(key, st, act, prm)
        return self.tobs(obs), st, rew, done, info


class TransformReward(GymnaxWrapper):
    def __init__(self, env, trew):
        super().__init__(env)
        self.trew = trew

    def step(self, key, st, act, prm=None):
        obs, st, rew, done, info = self._env.step(key, st, act, prm)
        return obs, st, self.trew(rew), done, info


class VecEnv(GymnaxWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.reset = jax.vmap(self._env.reset, in_axes=(0, None))
        self.step = jax.vmap(self._env.step, in_axes=(0, 0, 0, None))


@struct.dataclass
class NormalizeVecObsEnvState:
    mean: jnp.ndarray
    var: jnp.ndarray
    count: float
    est: environment.EnvState


class NormalizeVecObservation(GymnaxWrapper):
    def __init__(self, env):
        super().__init__(env)

    def reset(self, key, prm=None):
        obs, st = self._env.reset(key, prm)
        st = NormalizeVecObsEnvState(mean=jnp.zeros_like(obs), var=jnp.ones_like(obs), count=1e-4, est=st)
        dlt = jnp.mean(obs, axis=0) - st.mean
        tcnt = st.count + obs.shape[0]
        nmean = st.mean + dlt * obs.shape[0] / tcnt
        m2 = st.var * st.count + jnp.var(obs, axis=0) * obs.shape[0] + jnp.square(dlt) * st.count * obs.shape[0] / tcnt
        st = NormalizeVecObsEnvState(mean=nmean, var=m2 / tcnt, count=tcnt, est=st.est)
        return (obs - st.mean) / jnp.sqrt(st.var + 1e-8), st

    def step(self, key, st, act, prm=None):
        obs, est, rew, done, info = self._env.step(key, st.est, act, prm)
        dlt = jnp.mean(obs, axis=0) - st.mean
        tcnt = st.count + obs.shape[0]
        nmean = st.mean + dlt * obs.shape[0] / tcnt
        m2 = st.var * st.count + jnp.var(obs, axis=0) * obs.shape[0] + jnp.square(dlt) * st.count * obs.shape[0] / tcnt
        st = NormalizeVecObsEnvState(mean=nmean, var=m2 / tcnt, count=tcnt, est=est)
        return (obs - st.mean) / jnp.sqrt(st.var + 1e-8), st, rew, done, info


@struct.dataclass
class NormalizeVecRewEnvState:
    mean: jnp.ndarray
    var: jnp.ndarray
    count: float
    rval: float
    est: environment.EnvState


class NormalizeVecReward(GymnaxWrapper):
    def __init__(self, env, gam):
        super().__init__(env)
        self.gam = gam

    def reset(self, key, prm=None):
        obs, st = self._env.reset(key, prm)
        return obs, NormalizeVecRewEnvState(mean=0.0, var=1.0, count=1e-4, rval=jnp.zeros((obs.shape[0],)), est=st)

    def step(self, key, st, act, prm=None):
        obs, est, rew, done, info = self._env.step(key, st.est, act, prm)
        rval = st.rval * self.gam * (1 - done) + rew
        dlt = jnp.mean(rval, axis=0) - st.mean
        tcnt = st.count + obs.shape[0]
        nmean = st.mean + dlt * obs.shape[0] / tcnt
        m2 = st.var * st.count + jnp.var(rval, axis=0) * obs.shape[0] + jnp.square(dlt) * st.count * obs.shape[0] / tcnt
        st = NormalizeVecRewEnvState(mean=nmean, var=m2 / tcnt, count=tcnt, rval=rval, est=est)
        return obs, st, rew / jnp.sqrt(st.var + 1e-8), done, info


def gaim(env: MetabGame) -> jnp.ndarray:
    imat = jnp.zeros((env.nagt, env.nagt))
    for i in range(env.nagt):
        for j in range(env.nagt):
            if i != j:
                imat = imat.at[i, j].set(jnp.sum(env.amsk[j]))
    return imat


def aebal(env: MetabGame) -> Dict:
    ecnt = [jnp.sum(m) for m in env.amsk]
    mavg = np.mean(ecnt)
    mstd = np.std(ecnt)
    imat = gaim(env)
    tint = jnp.sum(imat)
    mxp = env.nagt * (env.nagt - 1) * env.ncpd
    return {
        "num_agents": env.nagt,
        "num_compounds": env.ncpd,
        "essential_compounds_per_agent": ecnt,
        "mean_essential_compounds": mavg,
        "std_essential_compounds": mstd,
        "balance_ratio": mstd / mavg if mavg > 0 else 0,
        "interaction_matrix": imat,
        "interaction_density": float(tint / mxp) if mxp > 0 else 0,
        "total_interactions": float(tint),
        "is_balanced": (mstd / mavg if mavg > 0 else 0) < 0.2,
        "is_highly_interactive": float(tint / mxp) > 0.1 if mxp > 0 else False,
    }


def cbenv(ecfg: Dict, nagt: int, cpa: Optional[int] = None) -> Tuple[Dict, List[Dict]]:
    cpds = list(ecfg["contents"].keys())
    cpa = cpa or max(1, len(cpds) // nagt)
    acfgs = []
    for i in range(nagt):
        sidx = i * cpa
        eidx = len(cpds) if i == nagt - 1 else min((i + 1) * cpa, len(cpds))
        acfgs.append({"id": f"balanced_agent_{i}", "name": f"Balanced Agent {i}", "essential_compounds": cpds[sidx:eidx]})
    return ecfg, acfgs


def tenv(env: MetabGame, eprm: EnvParams, ntst: int = 10) -> Dict:
    key = GK
    try:
        key, rkey = jax.random.split(key)
        st, obs = env.reset(rkey, eprm)
        tres = {"reset_successful": True, "initial_obs_shape": obs.obs.shape, "initial_compounds_sum": float(jnp.sum(st.cpds)), "step_results": []}
        for s in range(ntst):
            key, *akeys = jax.random.split(key, env.nagt + 1)
            acts = tuple(jax.random.normal(akeys[i], env.actsh()) * 0.01 for i in range(env.nagt))
            nst, nobs, rews, done, info = env.step(st, acts, eprm)
            tres["step_results"].append({"step": s, "rewards": [float(r) for r in rews], "total_reward": float(sum(rews)), "compounds_sum": float(jnp.sum(nst.cpds)), "done": bool(done), "obs_shape": nobs.obs.shape})
            st = nst
            if done:
                break
        tres["test_successful"] = True
        tres["total_steps_completed"] = len(tres["step_results"])
    except Exception as e:
        tres = {"test_successful": False, "error": str(e), "error_type": type(e).__name__}
    return tres


def benv(env: MetabGame, eprm: EnvParams, neps: int = 100, elen: int = 50) -> Dict:
    key = GK
    wkey, key = jax.random.split(key)
    st, _ = env.reset(wkey, eprm)
    env.step(st, tuple(jnp.zeros(env.actsh()) for _ in range(env.nagt)), eprm)
    rtms = []
    for _ in range(neps):
        rkey, key = jax.random.split(key)
        t0 = time.time()
        env.reset(rkey, eprm)
        rtms.append(time.time() - t0)
    stms = []
    ekey, key = jax.random.split(key)
    st, _ = env.reset(ekey, eprm)
    for _ in range(elen):
        key, *akeys = jax.random.split(key, env.nagt + 1)
        acts = tuple(jax.random.normal(akeys[i], env.actsh()) * 0.01 for i in range(env.nagt))
        t0 = time.time()
        st, _, _, done, _ = env.step(st, acts, eprm)
        stms.append(time.time() - t0)
        if done:
            break
    return {
        "num_agents": env.nagt,
        "num_compounds": env.ncpd,
        "reset_time_mean": np.mean(rtms),
        "reset_time_std": np.std(rtms),
        "step_time_mean": np.mean(stms),
        "step_time_std": np.std(stms),
        "steps_per_second": 1.0 / np.mean(stms) if stms else 0,
        "episodes_benchmarked": neps,
        "steps_benchmarked": len(stms),
    }
    
