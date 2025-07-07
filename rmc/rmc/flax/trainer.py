# -*- coding: utf-8 -*-

"""Definition of functions for training Flax neural network."""

from functools import partial
from typing import Any, Callable, List, Optional
from jax.typing import ArrayLike

import jax
import optax

import jax.numpy as jnp

from flax import nnx
from flax import jax_utils
from flax.training.train_state import TrainState

from .nn_config_dict import NNConfigDict, DataSetDict

PyTree = Any

def mse_loss(output: ArrayLike, labels: ArrayLike) -> float:
    """Compute Mean Squared Error (MSE) loss for training via Optax.

    Args:
        output: Comparison signal.
        labels: Reference signal.

    Returns:
        MSE between `output` and `labels`.
    """
    mse = optax.l2_loss(output, labels)
    return jnp.mean(mse)
    
def build_optimizer(config: NNConfigDict,
                    learning_rate_fn: optax._src.base.Schedule,
                    ):
    if config["opt_type"] == "SGD":
        # Stochastic Gradient Descent optimiser
        if "momentum" in config:
            tx = optax.sgd(
                learning_rate=learning_rate_fn, momentum=config["momentum"], nesterov=True
            )
        else:
            tx = optax.sgd(learning_rate=learning_rate_fn)
    elif config["opt_type"] == "ADAM":
        # Adam optimiser
        tx = optax.adam(
            learning_rate=learning_rate_fn,
        )
    elif config["opt_type"] == "ADAMW":
        # Adam with weight decay regularization
        tx = optax.adamw(
            learning_rate=learning_rate_fn,
        )
    else:
        raise NotImplementedError(
            f"Optimizer specified {config['opt_type']} has not been included."
        )
    return tx
    
@jax.jit
def train_step(state: TrainState,
               batch_in: ArrayLike,
               batch_out: ArrayLike,
               criterion: Callable):
    """Train for a single step."""
    
    def loss_fn(params):
        model = state.static.merge(params)
        output = model(batch_in)
        loss = criterion(output, batch_out)
        return loss
        
    grad_fn = jax.value_and_grad(loss_fn)
    loss, grads = grad_fn(state.params)
    grads = lax.pmean(grads, axis_name="batch")
    loss = lax.pmean(loss, axis_name="batch")

    # Update parameters
    state = state.apply_gradients(grads=grads)
    return state, loss
    

def train(config: NNConfigDict,
          model: Callable,
          train_ds: DataSetDict,
          test_ds: Optional[DataSetDict] = None,
    ):
    """Train Flax model.
    
    Function for training a Flax neural network model. This uses data parallel 
    training assuming sharded batched data.

    Args:
        config: Hyperparameter configuration.
        model: Flax model to train.
        train_ds: Dictionary of training data (includes images and
                labels).
        test_ds: Dictionary of testing data (includes images and
                labels). No eval function is run if no test data
                is provided.
    """
    # Configure seed
    if "seed" not in config:
        key = jax.random.PRNGKey(0)
    else:
        key = jax.random.PRNGKey(config["seed"])

    # Build scheduler
    if "lr_schedule" in config:
        lr_schedule_fn = config["lr_schedule"]
    else:
        lr_schedule_fn = optax.constant_schedule(config["base_lr"])
            
    # Build optimizer
    tx = build_optimizer(config, lr_schedule_fn)
            
    # Build criterion
    if "criterion" in config:
        criterion = config["criterion"]
    else:
        criterion = mse_loss
            
        
    # Splits the model into State and GraphDef pytree objects
    # (representing parameters and the graph definition)
    params, static = model.split(nnx.Param)
        
    # Initialize training state
    state = TrainState.create(apply_fn=None,
                              params=params,
                              tx=tx,
                              static=static,
            )

    # For parallel training
    state = jax_utils.replicate(state)
    p_train_step = jax.pmap(partial(train_step, criterion=criterion), axis_name="batch",)

    # Configure batching
    ndata = train_ds["input"].shape[0]
    batch_size = config["batch_size"]
    batches = ndata // batch_size
        
    # Configure sharding
    ndevloc = jax.local_device_count()
    shp_in = train_ds["input"].shape[1:]
    dt_in_shape = (ndevloc, -1, *shp_in)
    shp_out = train_ds["label"].shape[1:]
    dt_out_shape = (ndevloc, -1, *shp_out)

    # Execute training loop
    for epoch in range(config["max_epochs"]):
        avg_loss = 0.
        num_items = 0

        key, subkey = jax.random.split(key)
        perms = jax.random.permutation(subkey, ndata)

        for i in range(batches):
            batch_in = train_ds["input"][perms[i*batch_size:(i+1)*batch_size]].reshape(data_in_shape)
            batch_out = train_ds["label"][perms[i*batch_size:(i+1)*batch_size]].reshape(data_out_shape)
                
            state, loss = p_train_step(state, batch_in, batch_out)

            avg_loss += loss.mean() * batch_in.shape[0]
            num_items += batch.shape[0]

        # Only to figure out current learning rate, which cannot be stored in stateless optax.
        lr = lr_schedule_fn(state.step)
        # Print the averaged training loss so far.
        print(f"Epoch: {epoch}, Average Loss: {avg_loss / num_items}, lr: {lr}")
        
