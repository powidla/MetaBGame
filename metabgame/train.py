import jax
import jax.numpy as jnp
from flax.training.train_state import TrainState
from env import sspe, rspe, capmc
from ac import Continuous_ActorCritic, Critic, MultiAgentTransition
import optax

# Bayes params
MCS = 4
MRT = 1.0


def sample_mask(obs: jnp.ndarray, rng: jax.random.PRNGKey, mrt: float) -> jnp.ndarray:
    return obs * jax.random.bernoulli(rng, p=1.0 - mrt, shape=obs.shape)


def mc_expect_obs(obs: jnp.ndarray, rng: jax.random.PRNGKey, nsmp: int, mrt: float) -> jnp.ndarray:
    return jnp.mean(jax.vmap(lambda k: sample_mask(obs, k, mrt))(jax.random.split(rng, nsmp)), axis=0)


def init_network(env, cfg, rng):
    rng, *irngs = jax.random.split(rng, 3)
    eobs = jnp.zeros(env.obssh())
    anet = Continuous_ActorCritic(action_dim=env.actsh()[0], activation=cfg["ACTIVATION"])
    ast = TrainState.create(apply_fn=anet.apply, params=anet.init(irngs[0], eobs), tx=optax.adam(learning_rate=cfg["ACTOR_LR"], eps=1e-5))
    cnet = Critic(activation=cfg["ACTIVATION"])
    cst = TrainState.create(apply_fn=cnet.apply, params=cnet.init(irngs[1], eobs), tx=optax.adam(learning_rate=cfg["CRITIC_LR"], eps=1e-5))
    return anet, ast, cnet, cst, rng


def init_env(env, eprm, cfg, rng):
    rng, rrng = jax.random.split(rng)
    ests, obs = rspe(jax.random.split(rrng, cfg["NUM_ENVS"]), env, eprm, cfg["NUM_ENVS"])
    return ests, obs, rng


def make_gae_func(cnet, cfg):
    def cgae(cparams, tbatch, lobs):
        tvals = jax.vmap(jax.vmap(cnet.apply, in_axes=(None, 0)), in_axes=(None, 0))(cparams, tbatch.obs)
        lvals = jax.vmap(cnet.apply, in_axes=(None, 0))(cparams, lobs)

        def gae_advance(carry, info):
            gae, nval = carry
            trn, val = info
            dlt = trn.reward + cfg["GAMMA"] * nval * (1 - trn.done) - val
            gae = dlt + cfg["GAMMA"] * cfg["GAE_LAMBDA"] * (1 - trn.done) * gae
            return (gae, val), gae

        _, advs = jax.lax.scan(gae_advance, (jnp.zeros_like(lvals), lvals), (tbatch, tvals), reverse=True, unroll=16)
        return advs, advs + tvals
    return cgae


def make_reset_fn(env, eprm):
    def mrenv(done, st, obs, key):
        nst, nobs = env.reset(key, eprm)
        return jax.lax.cond(done, lambda: (nst, nobs.obs), lambda: (st, obs))
    return mrenv


def gather_acts(ests, lobs, anet, cnet, ast, cst, env, eprm, cfg, rng):
    nagt = env.nagt
    nenv = cfg["NUM_ENVS"]
    rng, *arngs = jax.random.split(rng, nagt + 1)
    aacts, avals, alps = [], [], []
    for aidx in range(nagt):
        akeys = jax.random.split(arngs[aidx], nenv)
        pi = jax.vmap(anet.apply, in_axes=(None, 0))(ast.params, lobs)
        vals = jax.vmap(cnet.apply, in_axes=(None, 0))(cst.params, lobs)
        acts = jax.vmap(lambda pi, k: pi.sample(seed=k))(pi, akeys)
        lps = jax.vmap(lambda pi, a: pi.log_prob(a))(pi, acts)
        aacts.append(acts)
        avals.append(vals)
        alps.append(lps)
    return aacts, avals, alps, rng


def gather_acts_b(ests, lobs, anet, cnet, ast, cst, env, eprm, cfg, rng, nmcs, mrt):
    nagt = env.nagt
    nenv = cfg["NUM_ENVS"]
    rng, mcrng, *arngs = jax.random.split(rng, nagt + 2)
    mobs = jax.vmap(lambda o, k: mc_expect_obs(o, k, nmcs, mrt))(lobs, jax.random.split(mcrng, nenv))
    aacts, avals, alps = [], [], []
    for aidx in range(nagt):
        akeys = jax.random.split(arngs[aidx], nenv)
        pi = jax.vmap(anet.apply, in_axes=(None, 0))(ast.params, mobs)
        vals = jax.vmap(cnet.apply, in_axes=(None, 0))(cst.params, lobs)
        acts = jax.vmap(lambda pi, k: pi.sample(seed=k))(pi, akeys)
        lps = jax.vmap(lambda pi, a: pi.log_prob(a))(pi, acts)
        aacts.append(acts)
        avals.append(vals)
        alps.append(lps)
    return aacts, avals, alps, rng


def step_and_reset(ests, aacts, env, eprm, cfg, rng):
    nenv = cfg["NUM_ENVS"]
    nests, nobs, rews, dns, infs = sspe(ests, tuple(aacts), env, eprm, nenv)
    rng, rrng = jax.random.split(rng)
    rkeys = jax.random.split(rrng, nenv)
    mres = make_reset_fn(env, eprm)
    fsts, fobs = jax.vmap(mres)(dns, nests, nobs.obs, rkeys)
    return fsts, fobs, rews, dns, infs, rng


def build_transit(aacts, avals, alps, lobs, rews, dns, infs, nagt):
    return tuple(
        MultiAgentTransition(done=dns, action=aacts[i], value=avals[i], reward=rews[i], log_prob=alps[i], obs=lobs, agent_id=i, info=infs)
        for i in range(nagt)
    )


def ppo_actor_loss(aparams, cparams, anet, cnet, cgae, cfg, nagt, lobs, atrjs):
    tloss = 0.0
    for aidx in range(nagt):
        trj = atrjs[aidx]
        advs, _ = cgae(cparams, trj, lobs)
        obsf = trj.obs.reshape(-1, trj.obs.shape[-1])
        actf = trj.action.reshape(-1, trj.action.shape[-1])
        lpf = trj.log_prob.reshape(-1)
        advf = advs.reshape(-1)
        adst = jax.vmap(anet.apply, in_axes=(None, 0))(aparams, obsf)
        lps = jax.vmap(lambda d, a: d.log_prob(a))(adst, actf)
        prat = jnp.exp(lps - lpf)
        clp = jnp.clip(prat, 1 - cfg["CLIP_EPS"], 1 + cfg["CLIP_EPS"])
        tloss += jnp.mean(jnp.minimum(prat * advf, clp * advf))
    return -tloss / nagt


def bayes_act_loss(aparams, cparams, anet, cnet, cgae, cfg, nagt, lobs, atrjs, rng, nmcs, mrt):
    tloss = 0.0
    for aidx in range(nagt):
        trj = atrjs[aidx]
        advs, _ = cgae(cparams, trj, lobs)
        obsf = trj.obs.reshape(-1, trj.obs.shape[-1])
        actf = trj.action.reshape(-1, trj.action.shape[-1])
        lpf = trj.log_prob.reshape(-1)
        advf = advs.reshape(-1)
        rng, mcrng = jax.random.split(rng)
        eobs = jax.vmap(lambda o, k: mc_expect_obs(o, k, nmcs, mrt))(obsf, jax.random.split(mcrng, obsf.shape[0]))
        adst = jax.vmap(anet.apply, in_axes=(None, 0))(aparams, eobs)
        lps = jax.vmap(lambda d, a: d.log_prob(a))(adst, actf)
        prat = jnp.exp(lps - lpf)
        clp = jnp.clip(prat, 1 - cfg["CLIP_EPS"], 1 + cfg["CLIP_EPS"])
        tloss += jnp.mean(jnp.minimum(prat * advf, clp * advf))
    return -tloss / nagt


def critic_loss_fn(cparams, cnet, cgae, atrjs, lobs, nagt):
    tot = 0.0
    for i in range(nagt):
        trj = atrjs[i]
        _, tgts = cgae(cparams, trj, lobs)
        obsf = trj.obs.reshape(-1, trj.obs.shape[-1])
        tgtf = tgts.reshape(-1)
        vals = jax.vmap(cnet.apply, in_axes=(None, 0))(cparams, obsf)
        tot += jnp.mean(jnp.square(tgtf - vals))
    return tot / nagt


def build_metrics(linfo, atb, ainfs, nagt):
    mets = {"actor_loss": linfo[0], "critic_loss": linfo[1]}
    for i in range(nagt):
        mets[f"episode_reward_agent_{i}"] = jnp.mean(jnp.sum(atb[i].reward, axis=0))
    for k, acts in ainfs.items():
        mets[k] = jnp.mean(jnp.sum(acts, axis=0))
    return mets


def build_output(rst, amets, nagt):
    return {
        "runner_state": rst,
        "metrics": amets,
        "agent_rewards": {f"agent_{i}": amets[f"episode_reward_agent_{i}"] for i in range(nagt)},
        "num_agents": nagt,
    }


def nash(env, eprm, cfg: dict, rng: jax.random.PRNGKey):
    nenv = cfg["NUM_ENVS"]
    nagt = env.nagt

    anet, ast, cnet, cst, rng = init_network(env, cfg, rng)
    ests, obs, rng = init_env(env, eprm, cfg, rng)
    cgae = make_gae_func(cnet, cfg)

    def update_step(rst, unused):
        def env_step_func(rst, unused):
            ast, cst, ests, lobs, rng = rst
            aacts, avals, alps, rng = gather_acts(ests, lobs, anet, cnet, ast, cst, env, eprm, cfg, rng)
            fsts, fobs, rews, dns, infs, rng = step_and_reset(ests, aacts, env, eprm, cfg, rng)
            acls = capmc(jnp.stack(aacts), env, nenv)
            atrn = build_transit(aacts, avals, alps, lobs, rews, dns, infs, nagt)
            return (ast, cst, fsts, fobs, rng), (atrn, acls)

        rst, trj_data = jax.lax.scan(env_step_func, rst, None, cfg["NUM_STEPS"])
        atb, ainfs = trj_data
        ast, cst, ests, lobs, rng = rst

        def update_minib(tst, binfo):
            ast, cst, rng = tst
            atb, lobs = binfo
            for _ in range(cfg.get("NESTED_UPDATES", 1)):
                closs, cgrd = jax.value_and_grad(critic_loss_fn)(cst.params, cnet, cgae, atb, lobs, nagt)
                cst = cst.apply_gradients(grads=cgrd)
            aloss, agrd = jax.value_and_grad(ppo_actor_loss)(ast.params, cst.params, anet, cnet, cgae, cfg, nagt, lobs, atb)
            ast = ast.apply_gradients(grads=agrd)
            return (ast, cst, rng), (aloss, closs)

        tst, linfo = update_minib((ast, cst, rng), (atb, lobs))
        ast, cst, rng = tst
        rst = (ast, cst, ests, lobs, rng)
        return rst, build_metrics(linfo, atb, ainfs, nagt)

    rng, _rng = jax.random.split(rng)
    rst = (ast, cst, ests, obs.obs, _rng)
    rst, amets = jax.lax.scan(update_step, rst, None, cfg["NUM_UPDATES"])
    return build_output(rst, amets, nagt)


def bnash(env, eprm, cfg: dict, rng: jax.random.PRNGKey, nmcs: int = MCS, mrt: float = MRT):
    nenv = cfg["NUM_ENVS"]
    nagt = env.nagt

    anet, ast, cnet, cst, rng = init_network(env, cfg, rng)
    ests, obs, rng = init_env(env, eprm, cfg, rng)
    cgae = make_gae_func(cnet, cfg)

    def update_step(rst, unused):
        def env_step_func(rst, unused):
            ast, cst, ests, lobs, rng = rst
            aacts, avals, alps, rng = gather_acts_b(ests, lobs, anet, cnet, ast, cst, env, eprm, cfg, rng, nmcs, mrt)
            fsts, fobs, rews, dns, infs, rng = step_and_reset(ests, aacts, env, eprm, cfg, rng)
            acls = capmc(jnp.stack(aacts), env, nenv)
            atrn = build_transit(aacts, avals, alps, lobs, rews, dns, infs, nagt)
            return (ast, cst, fsts, fobs, rng), (atrn, acls)

        rst, trj_data = jax.lax.scan(env_step_func, rst, None, cfg["NUM_STEPS"])
        atb, ainfs = trj_data
        ast, cst, ests, lobs, rng = rst

        def update_minib(tst, binfo):
            ast, cst, rng = tst
            atb, lobs = binfo
            for _ in range(cfg.get("NESTED_UPDATES", 1)):
                closs, cgrd = jax.value_and_grad(critic_loss_fn)(cst.params, cnet, cgae, atb, lobs, nagt)
                cst = cst.apply_gradients(grads=cgrd)
            rng, lrng = jax.random.split(rng)
            aloss, agrd = jax.value_and_grad(bayes_act_loss)(ast.params, cst.params, anet, cnet, cgae, cfg, nagt, lobs, atb, lrng, nmcs, mrt)
            ast = ast.apply_gradients(grads=agrd)
            return (ast, cst, rng), (aloss, closs)

        tst, linfo = update_minib((ast, cst, rng), (atb, lobs))
        ast, cst, rng = tst
        rst = (ast, cst, ests, lobs, rng)
        return rst, build_metrics(linfo, atb, ainfs, nagt)

    rng, _rng = jax.random.split(rng)
    rst = (ast, cst, ests, obs.obs, _rng)
    rst, amets = jax.lax.scan(update_step, rst, None, cfg["NUM_UPDATES"])
    return build_output(rst, amets, nagt)

