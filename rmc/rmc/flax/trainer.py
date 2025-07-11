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
    
    
def build_optax_optimizer(config: NNConfigDict,
                    learning_rate_fn: optax._src.base.Schedule,
                    ):
    """Build optax optimizer to include in NNX optimizer.

    Args:
        config: Configuration dictionary for neural network training.
        learning_rate_fn: Optax learning rate scheduler.
        
    Returns:
        Optax optimizer object.
    """
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
    
    
def loss_fn(criterion: Callable, model: Callable, batch: DataSetDict):
    output = model(batch["input"])
    loss = criterion(output, batch["label"])
    return loss
    
    
@nnx.jit
def train_step(model: Callable, criterion: Callable, optimizer: nnx.Optimizer, metrics: nnx.MultiMetric, batch: DataSetDict):
    """Train for a single step."""
    grad_fn = nnx.value_and_grad(loss_fn)
    loss, grads = grad_fn(criterion, model, batch)
    metrics.update(loss=loss)  # In-place updates.
    optimizer.update(grads)  # In-place updates.


@nnx.jit
def eval_step(model: Callable, criterion: Callable, metrics: nnx.MultiMetric, batch: DataSetDict):
    loss = loss_fn(criterion, model, batch)
    metrics.update(loss=loss)  # In-place updates.
  

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
    # Build scheduler
    if "lr_schedule" in config:
        lr_schedule_fn = config["lr_schedule"]
    else:
        lr_schedule_fn = optax.constant_schedule(config["base_lr"])
            
    # Build nnx optimizer
    tx = build_optax_optimizer(config, lr_schedule_fn)
    optimizer = nnx.Optimizer(model, tx)
    nnx.display(optimizer)
            
    # Build criterion
    if "criterion" in config:
        criterion = config["criterion"]
    else:
        criterion = mse_loss
            
    # Build metrics
    metrics = nnx.MultiMetric(
        loss=nnx.metrics.Average('loss'),
    )
    metrics_history = {
        "train_loss": [],
        "test_loss": [],
    }
    
    # Configure batching and logging
    ndata = train_ds["input"].shape[0]
    batch_size = config["batch_size"]
    nbatches = ndata // batch_size
    eval_every = config["eval_every"]
    
    # Execute training loop
    train_epochs = config["max_epochs"]
    for epoch in range(train_epochs):
        key, subkey = jax.random.split(key)
        perms = jax.random.permutation(subkey, ndata)
        
        batch = {}

        for i in range(nbatches):
            batch["input"] = train_ds["input"][perms[i*batch_size:(i+1)*batch_size]]
            batch["label"] = train_ds["label"][perms[i*batch_size:(i+1)*batch_size]]
            
            train_step(model, criterion, optimizer, metrics, batch)
            
        if epoch > 0 and (epoch % eval_every == 0 or epoch == train_epochs - 1):  # One training epoch has passed.

            # Log the training metrics.
            for metric, value in metrics.compute().items():  # Compute the metrics.
                metrics_history[f'train_{metric}'].append(value)  # Record the metrics.
                metrics.reset()  # Reset the metrics for the test set.
                
            # Only to figure out current learning rate, which cannot be stored in stateless optax.
            lr = lr_schedule_fn(state.step)
            
            if test_ds is not None:
                ntestbatches = test_ds["input"].shape[0] // batch_size
                test_batch = {}
                # Compute the metrics on the test set after each training epoch.
                for i in range(ntestbatches):
                    test_batch["input"] = test_ds["input"][i*batch_size:(i+1)*batch_size]
                    test_batch["label"] = test_ds["label"][i*batch_size:(i+1)*batch_size]
                    eval_step(model, criterion, metrics, test_batch)

                # Log the test metrics.
                for metric, value in metrics.compute().items():
                    metrics_history[f'test_{metric}'].append(value)
                    metrics.reset()  # Reset the metrics for the next training epoch.
                    # Print the averaged training loss so far.
                    print(f"Epoch: {epoch}, Loss-train: {metrics_history['train_loss']}, lr: {lr}, Loss-test: {metrics_history['test_loss']}")
            else:
                # Print the averaged training loss so far.
                print(f"Epoch: {epoch}, Loss-train: {metrics_history['train_loss']}, lr: {lr}")

