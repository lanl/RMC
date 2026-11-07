import os, sys
import numpy as np
from functools import partial
from typing import Dict

import jax
import jax.numpy as jnp

from io_utils import (
    ConfigMA, 
    load_ConfigMA, 
    load_json, 
    get_activation_function, 
    get_schedule_function, 
    get_schedule_diffusion_function,
)
from density_targets import build_density_class

from rmc import (
    BaseLogDensity, 
    DenoisingDiffusionSampler, 
    HMC, 
    LiouvilleFlow, 
    PathIntegralSampler, 
    SMC, 
    SVGD
)


def build_initial_distribution(mean0: float, std0: float, dim: int):
    """Build the initial distribution for sampling."""
    mean0_arr = mean0 * jnp.ones((1, dim))
    cov0_arr = jnp.diagflat((std0**2 * jnp.ones((dim,)))).reshape((1, dim, dim))
    initialdist = partial(jax.random.multivariate_normal, mean=mean0_arr, cov=cov0_arr)
    return initialdist

def construct_sampling_object_and_override_conf(method: str, sampler_config: Dict, Dcl: BaseLogDensity):
    overrides = {}
    if method == "hmc":
        overrides["density_cl"] = Dcl
        overrides["sample_shape"] = tuple(sampler_config["sample_shape"])
        overrides["initial_sampler_cl"] = build_initial_distribution(
                    mean0=sampler_config["mean0"], std0=sampler_config["std0"], dim=sampler_config["dim"]
        )
        # Override configuration
        sampler_config.update(overrides)
        # Create sampler object
        smp_obj = HMC(sampler_config)
    elif method == "smc":
        overrides["density_cl"] = Dcl
        overrides["sample_shape"] = tuple(sampler_config["sample_shape"])
        nsamples = sampler_config["nsamples"]
        T = sampler_config["T"]
        # Define tempering
        sched = jnp.linspace(0, 1, T + 1)
        tempering_fn = lambda tstep: sched[tstep]   
        overrides["tempering_fn"] = tempering_fn
        # Override configuration
        sampler_config.update(overrides)
        # Create sampler object
        smp_obj = SMC(nsamples, T, sampler_config)
    elif method == "svgd":
        nsamples = sampler_config["nsamples"]
        overrides["density_cl"] = Dcl
        overrides["sample_shape"] = tuple(sampler_config["sample_shape"])
        # Override configuration
        sampler_config.update(overrides)
        # Create sampler object
        smp_obj = SVGD(nsamples, sampler_config)
    elif method == "lfis":
        overrides["activation_func"] = get_activation_function(sampler_config["activation_func"])
        schedule = get_schedule_function(sampler_config["schedule"])
        # Override configuration
        sampler_config.update(overrides)
        # Create sampler object
        smp_obj = LiouvilleFlow(sampler_config, Dcl, schedule)
    elif method == "pis":
        overrides["activation_func"] = get_activation_function(sampler_config["activation_func"])
        h = sampler_config["h"]
        T = sampler_config["T"]
        # Override configuration
        sampler_config.update(overrides)
        # Create sampler object
        smp_obj = PathIntegralSampler(sampler_config, Dcl, h, T)
    elif method == "dds":
        overrides["activation_func"] = get_activation_function(sampler_config["activation_func"])
        sigma = sampler_config["sigma"]
        K = sampler_config["K"]
        beta_schedule = get_schedule_diffusion_function(sampler_config["beta_schedule"])
        # Override configuration
        sampler_config.update(overrides)
        # Create sampler object
        smp_obj = DenoisingDiffusionSampler(sampler_config, Dcl, sigma, K, beta_schedule)
    else:
        print(f"Error: Unknown method '{method}'", file=sys.stderr)
        sys.exit(1)

    return smp_obj


def main():
    config_path = sys.argv[1] if len(sys.argv) > 1 else "config.json"

    try:
        config: ConfigMA = load_ConfigMA(config_path)
    except (FileNotFoundError, ValueError, KeyError, TypeError) as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    for tgt in config.targets:
        # Build target density
        print(f"Target: {tgt}")
        Dcl = build_density_class(tgt)

        # Read configuration for sampling method
        config_name = f"config_{tgt}.json"
        print(f"Reading configuration for {tgt} from {config_name}")
        config_path = config.path_configs / config_name
        smp_conf = load_json(config_path)
        smp_conf["root_path"] = config.path_out / tgt

        # Construct sampling object
        smp_obj = construct_sampling_object_and_override_conf(config.method, smp_conf, Dcl)
        print(f"{config.method} object constructed") 

        # Train model if applicable
        if hasattr(smp_obj, "train"):
            print(f"Training {config.method} model for {tgt}")
            smp_obj.train()

        # Run sampler
        samples = None
        if config.method == "hmc":
            smp_obj.sample()
            samples = np.array(smp_obj.qpath)
        elif config.method == "smc":
            smp_obj.sample()
            samples = np.array(smp_obj.hmc_.qall)
        elif config.method == "svgd":
            samples = np.array(smp_obj.sample())
        elif config.method == "lfis":
            key = jax.random.PRNGKey(smp_conf["seed"])
            key, subkey = jax.random.split(key)
            withw = True
            particle_path_, _, _ = smp_obj.sample(smp_conf["nsamples"], withw, subkey)
            samples = np.array(particle_path_)
        elif config.method == "pis":
            key = jax.random.PRNGKey(smp_conf["seed"])
            key, subkey = jax.random.split(key)
            particle_path_, _ = smp_obj.sample(smp_conf["nsamples"], subkey)
            samples = np.array(particle_path_)
        elif config.method == "dds":
            key = jax.random.PRNGKey(smp_conf["seed"])
            key, subkey = jax.random.split(key)
            particle_path_ = smp_obj.sample(smp_conf["nsamples"], subkey)
            samples = np.array(particle_path_)

        # Save samples to output file
        if samples is not None:
            print(f"Collected samples: {samples.shape}")
            os.makedirs(smp_conf["root_path"], exist_ok=True)
            np.save(smp_conf["root_path"] / f"samples_{tgt}.npy", samples)


if __name__ == "__main__":
    main()