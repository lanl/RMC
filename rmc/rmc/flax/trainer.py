# -*- coding: utf-8 -*-

"""Definition of functions for training Flax neural network."""

from functools import partial
from typing import Any, Callable, List, Optional
from jax.typing import ArrayLike

import jax
import optax

import jax.numpy as jnp

from jax.experimental import mesh_utils

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
    
    
def loss_fn(model: Callable, criterion: Callable, x: ArrayLike, y: ArrayLike):
    """Loss function definition.
    
    Args:
        model: Model to train.
        criterion: Criterion that defines loss function.
        x: Input (features) array.
        y: Output (labels) array.
    """
    output = model(x)
    loss = criterion(output, y)
    return loss
    
    
#@nnx.jit
def train_step(model: Callable, criterion: Callable, optimizer: nnx.Optimizer, metrics: nnx.MultiMetric, x: ArrayLike, y: ArrayLike):
    """Train for a single step.
    
    This function uses data and a criterion to optimize model parameters. It returns
    the current loss in the training set.
    
    Args:
        model: Model to train.
        criterion: Criterion to use for training.
        optimizer: NNX optimizer object used to train model.
        metrics: Dictionary of metrics to evaluate.
        x: Input (features) array.
        y: Output (labels) array.
        
    Returns:
        Loss evaluated.
    """
    grad_fn = nnx.value_and_grad(loss_fn)
    loss, grads = grad_fn(model, criterion, x, y)
    optimizer.update(grads)  # In-place updates.
    #optimizer.update(model, grads)  # In-place updates.
    metrics.update(loss=loss)  # In-place updates.
    return loss
    


@nnx.jit
def eval_step(model: Callable, criterion: Callable, metrics: nnx.MultiMetric, x: ArrayLike, y: ArrayLike):
    """Evaluate for a single step.
    
    This function uses data and a criterion to evaluate performance of current model.
    It returns the current loss evaluated in the testing set.
    
    Args:
        model: Model to train.
        criterion: Criterion to use for training.
        metrics: Dictionary of metrics to evaluate.
        x: Input (features) array.
        y: Output (labels) array.
        
    Returns:
        Loss evaluated.
    """

    loss = loss_fn(model, criterion, x, y)
    metrics.update(loss=loss)  # In-place updates.
    
    return loss
  
  
def iterate_dataset(ds: DataSetDict, steps: int, batch_size: int, subkey: ArrayLike, shuffle: bool=False,):
    """Yield chunks of dataset for training/evaluating ML model.
    
    Yield a number of `steps` chunks of the dataset each of size `batch_size`.
    
    Args:
        ds: Data set to iterate. It is a dictionary where `input` keyword defines the
            input (feature) data and `label` keyword defines the output data.
        steps: Number of data chunks to collect.
        batch_size: Number of samples in each chunk.
        subkey: JAX random generation.
        shuffle: If true, the data is randomly ordered. Otherwise, the data is
            returned with the ordering of the original dataset.
        
    Returns:
        Input and output arrays.
    """
    ndata = ds["input"].shape[0]
    #print(f"Inside iterate_dataset --> ds['input'].shape: {ds['input'].shape}")
    #print(f"Inside iterate_dataset --> ds['label'].shape: {ds['label'].shape}")
    
    if shuffle:
        perms = jax.random.permutation(subkey, ndata)
    else:
        perms = jnp.range(ndata)
    for i in range(steps):
        x = ds["input"][perms[i*batch_size:(i+1)*batch_size]]
        y = ds["label"][perms[i*batch_size:(i+1)*batch_size]]
        yield x, y



def train(config: NNConfigDict,
          model: Callable,
          key: ArrayLike,
          train_ds: DataSetDict,
          test_ds: Optional[DataSetDict] = None,
    ):
    """Train Flax model.
    
    Function for training a Flax neural network model. This uses data parallel 
    training assuming sharded batched data.

    Args:
        config: Hyperparameter configuration.
        model: Flax model to train.
        key: JAX random generation.
        train_ds: Dictionary of training data (includes images and
                labels).
        test_ds: Dictionary of testing data (includes images and
                labels). No eval function is run if no test data
                is provided.
    """
    # Create mesh + shardings
    num_devices = jax.local_device_count()
    mesh = jax.sharding.Mesh(
            mesh_utils.create_device_mesh((num_devices,)), ("data",)
    )
    model_sharding = jax.NamedSharding(mesh, jax.sharding.PartitionSpec())
    data_sharding = jax.NamedSharding(mesh, jax.sharding.PartitionSpec("data"))
    
    # Build scheduler
    if "lr_schedule" in config:
        lr_schedule_fn = config["lr_schedule"]
    else:
        lr_schedule_fn = optax.constant_schedule(config["base_lr"])
            
    # Build nnx optimizer
    tx = build_optax_optimizer(config, lr_schedule_fn)
    optimizer = nnx.Optimizer(model, tx)
    #optimizer = nnx.Optimizer(model, tx, wrt=nnx.Param)
    #nnx.display(optimizer)
            
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
    
    # Replicate state
    state = nnx.state((model, optimizer))
    state = jax.device_put(state, model_sharding)
    nnx.update((model, optimizer), state)
    
    # Configure batching and logging
    ndata = train_ds["input"].shape[0]
    batch_size = config["batch_size"]
    nbatches = ndata // batch_size
    eval_every = config["eval_every"]
    
    # Execute training loop
    train_epochs = config["max_epochs"]
    key, subkey1, subkey2 = jax.random.split(key, 3)
    for epoch in range(train_epochs):
        for (x, y) in iterate_dataset(train_ds, nbatches, batch_size, subkey1, True):
            # Shard data
            x, y = jax.device_put((x, y), data_sharding)
            # Train
            loss = train_step(model, criterion, optimizer, metrics, x, y)
            #print(f"Epoch: {epoch} --> In minibatch: Loss-train: {loss}")
    
        if epoch > 0 and (epoch % eval_every == 0 or epoch == train_epochs - 1):  # One training epoch has passed.

            # Log the training metrics.
            for metric, value in metrics.compute().items():  # Compute the metrics.
                metrics_history[f'train_{metric}'].append(value)  # Record the metrics.
                metrics.reset()  # Reset the metrics for the test set.
                
            # Only to figure out current learning rate, which cannot be stored in stateless optax.
            lr = lr_schedule_fn(optimizer.step)
            
            if test_ds is not None:
                ntestbatches = test_ds["input"].shape[0] // batch_size
                for i, (x, y) in enumerate(iterate_dataset(test_ds, ntestbatches, batch_size, subkey2)):
                    # Shard data
                    x, y = jax.device_put((x, y), data_sharding)
                    # Evaluate
                    loss = eval_step(model, criterion, metrics, x, y)

                # Log the test metrics.
                for metric, value in metrics.compute().items():
                    metrics_history[f'test_{metric}'].append(value)
                    metrics.reset()  # Reset the metrics for the next training epoch.
                    # Print the averaged training loss so far.
                    print(f"Epoch: {epoch}, Loss-train: {metrics_history['train_loss']}, lr: {lr}, Loss-test: {metrics_history['test_loss']}")
            else:
                # Print the averaged training loss so far.
                print(f"Epoch: {epoch}, Loss-train: {metrics_history['train_loss'][-1]}, lr: {lr}")

    # dereplicate state
    state = nnx.state((model, optimizer))
    state = jax.device_get(state)
    nnx.update((model, optimizer), state)
    return model, metrics_history['train_loss'][-1]
