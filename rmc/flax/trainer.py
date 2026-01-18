# -*- coding: utf-8 -*-

"""Definition of functions for training Flax neural network."""

import pickle
from pathlib import Path
from typing import Any, Callable, Optional

import jax
import jax.numpy as jnp
from jax.experimental import mesh_utils
from jax.typing import ArrayLike

import optax
from flax import nnx
from flax.serialization import from_state_dict, to_state_dict

from .nn_config_dict import DataSetDict, NNConfigDict

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


def build_optax_optimizer(
    config: NNConfigDict,
    learning_rate_fn: optax._src.base.Schedule,
):
    """Build optax optimizer to include in NNX optimizer.

    Args:
        config: Configuration dictionary for neural network training.
        learning_rate_fn: Optax learning rate scheduler.

    Returns:
        Optax optimizer object.
    """
    # Build optimizer
    if config["opt_type"] == "SGD":
        # Stochastic Gradient Descent optimiser
        if "momentum" in config:
            opt_core = partial(optax.sgd, momentum=config["momentum"], nesterov=True)
        else:
            opt_core = optax.sgd
    elif config["opt_type"] == "ADAM":
        # Adam optimiser
        opt_core = optax.adam
    elif config["opt_type"] == "ADAMW":
        # Adam with weight decay regularization
        opt_core = optax.adamw
    else:
        raise NotImplementedError(
            f"Optimizer specified {config['opt_type']} has not been included."
        )

    # Build optax optimizer to be able to get lr later
    tx = optax.inject_hyperparams(opt_core)(learning_rate=learning_rate_fn)

    return tx


def loss_fn(model: Callable, criterion: Callable, x: ArrayLike, y: ArrayLike):
    """Loss function definition.

    Args:
        model: Model to train.
        criterion: Criterion that defines loss function.
        x: Input (features) array.
        y: Output (labels) array.
    """
    # output = model(x)
    # loss = criterion(output, y)
    loss = criterion(model, x, y)
    return loss


@nnx.jit(static_argnums=(1, 6))
def train_step(
    model: Callable,
    criterion: Callable,
    optimizer: nnx.Optimizer,
    metrics: nnx.MultiMetric,
    x: ArrayLike,
    y: ArrayLike,
    has_aux: bool = False,
):
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
        has_aux:  Indicates whether loss fun returns a pair where the first element
            is considered the output of the mathematical function to be differentiated
            and the second element is auxiliary data.

    Returns:
        Loss evaluated.
    """
    grad_fn = nnx.value_and_grad(loss_fn, has_aux=has_aux)
    loss, grads = grad_fn(model, criterion, x, y)
    # optimizer.update(grads)  # In-place updates.
    # optimizer.update(model, grads)  # In-place updates.
    if has_aux:
        # optimizer.update(model, grads, value=loss[1])  # In-place updates.
        optimizer.update(grads, value=loss[1])  # In-place updates.
        metrics.update(loss=loss[0], auxloss=loss[1])  # In-place updates.
        return loss[0]
    else:
        # optimizer.update(model, grads, value=loss)  # In-place updates.
        optimizer.update(grads, value=loss)  # In-place updates.
        metrics.update(loss=loss)  # In-place updates.
    return loss


@nnx.jit(static_argnums=(1, 5))
def eval_step(
    model: Callable,
    criterion: Callable,
    metrics: nnx.MultiMetric,
    x: ArrayLike,
    y: ArrayLike,
    has_aux: bool = False,
):
    """Evaluate for a single step.

    This function uses data and a criterion to evaluate performance of current model.
    It returns the current loss evaluated in the testing set.

    Args:
        model: Model to train.
        criterion: Criterion to use for training.
        metrics: Dictionary of metrics to evaluate.
        x: Input (features) array.
        y: Output (labels) array.
        has_aux:  Indicates whether loss fun returns a pair where the first element
            is considered the output of the mathematical function to be differentiated
            and the second element is auxiliary data.

    Returns:
        Loss evaluated.
    """

    loss = loss_fn(model, criterion, x, y)
    # metrics.update(loss=loss)  # In-place updates.
    if has_aux:
        metrics.update(loss=loss[0], auxloss=loss[1])  # In-place updates.
        return loss[0]
    else:
        metrics.update(loss=loss)  # In-place updates.

    return loss


def iterate_dataset(
    ds: DataSetDict,
    steps: int,
    batch_size: int,
    subkey: ArrayLike,
    shuffle: bool = False,
):
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

    if shuffle:
        perms = jax.random.permutation(subkey, ndata)
    else:
        perms = jnp.arange(ndata)
    for i in range(steps):
        x = ds["input"][perms[i * batch_size : (i + 1) * batch_size]]
        y = ds["label"][perms[i * batch_size : (i + 1) * batch_size]]
        yield x, y


def train(
    config: NNConfigDict,
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
    mesh = jax.sharding.Mesh(mesh_utils.create_device_mesh((num_devices,)), ("data",))
    model_sharding = jax.NamedSharding(mesh, jax.sharding.PartitionSpec())
    data_sharding = jax.NamedSharding(mesh, jax.sharding.PartitionSpec("data"))

    # Build scheduler
    if "lr_schedule" in config:
        lr_schedule_fn = config["lr_schedule"]
    else:
        lr_schedule_fn = optax.constant_schedule(config["base_lr"])

    # Build nnx optimizer
    tx = build_optax_optimizer(config, lr_schedule_fn)
    optimizer = nnx.Optimizer(model, tx, wrt=nnx.Param)
    # nnx.display(optimizer)

    # Build criterion
    if "criterion" in config:
        criterion = config["criterion"]
    else:
        criterion = mse_loss

    # Build metrics
    if config["has_aux"]:
        metrics = nnx.MultiMetric(
            loss=nnx.metrics.Average("loss"),
            auxloss=nnx.metrics.Average("auxloss"),
        )
        metrics_history = {
            "train_loss": [],
            "test_loss": [],
            "train_auxloss": [],
            "test_auxloss": [],
        }
    else:
        metrics = nnx.MultiMetric(
            loss=nnx.metrics.Average("loss"),
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
    best_loss = 1e8

    min_epoch = 1  # 10

    patience_counter = 0
    for epoch in range(train_epochs + 1):
        for x, y in iterate_dataset(train_ds, nbatches, batch_size, subkey1, True):
            # Shard data
            x, y = jax.device_put((x, y), data_sharding)
            # Train
            model.train()  # Switch to train mode
            loss = train_step(model, criterion, optimizer, metrics, x, y, config["has_aux"])
            # print(f"Epoch: {epoch} --> In minibatch: Loss-train: {loss}")

        if epoch > 0 and (
            epoch % eval_every == 0 or epoch == train_epochs
        ):  # One training epoch has passed.

            # Log the training metrics.
            for metric, value in metrics.compute().items():  # Compute the metrics.
                metrics_history[f"train_{metric}"].append(value)  # Record the metrics.
            metrics.reset()  # Reset the metrics for the test set.

            # Get current learning rate from optax optimizer (configured to store it).
            lr = optimizer.opt_state.hyperparams["learning_rate"].value

            if test_ds is not None:
                ntestbatches = test_ds["input"].shape[0] // batch_size
                model.eval()  # Switch to eval mode
                for i, (x, y) in enumerate(
                    iterate_dataset(test_ds, ntestbatches, batch_size, subkey2)
                ):
                    # Shard data
                    x, y = jax.device_put((x, y), data_sharding)
                    # Evaluate
                    loss = eval_step(model, criterion, metrics, x, y, config["has_aux"])

                # Log the test metrics.
                for metric, value in metrics.compute().items():
                    metrics_history[f"test_{metric}"].append(value)
                metrics.reset()  # Reset the metrics for the next training epoch.
                # Print the averaged training loss so far.
                if config["has_aux"]:
                    print(
                        f"Epoch: {epoch:>5d}, Loss-train: {metrics_history['train_loss'][-1]:>13.10f}, AuxLoss-train: {metrics_history['train_auxloss'][-1]:>10.7f}, lr: {lr:>9.8f}, Loss-test: {metrics_history['test_loss'][-1]:>13.10f}, AuxLoss-test: {metrics_history['test_auxloss'][-1]:>10.7f}"
                    )
                else:
                    print(
                        f"Epoch: {epoch:>5d}, Loss-train: {metrics_history['train_loss'][-1]:>13.10f}, lr: {lr:>9.8f}, Loss-test: {metrics_history['test_loss'][-1]:>13.10f}"
                    )
            else:
                # Print the averaged training loss so far.
                if config["has_aux"]:
                    print(
                        f"Epoch: {epoch:>5d}, Loss-train: {metrics_history['train_loss'][-1]:>13.10f}, AuxLoss-train: {metrics_history['train_auxloss'][-1]:>10.7f}, lr: {lr:>9.8f}"
                    )
                else:
                    print(
                        f"Epoch: {epoch:>5d}, Loss-train: {metrics_history['train_loss'][-1]:>13.10f}, lr: {lr:>9.8f}"
                    )
        if loss < best_loss:
            best_loss = loss
            patience_counter = 0

        if len(metrics_history["train_loss"]) > 1:  # epoch > min_epoch:
            # Early stopping
            if config["has_aux"]:
                if metrics_history["train_auxloss"][-1] < config["max_loss"] or metrics_history[
                    "train_loss"
                ][-1] < (config["max_loss"] / 10.0):
                    break
            else:
                if metrics_history["train_loss"][-1] < config["max_loss"]:
                    break

            # adjust learning rate
            # patience_counter += 1
            # if patience_counter > patience:
            #    print("===Reducing LR====")
            # optimizer.param_groups[0]['lr'] = optimizer.param_groups[0]['lr']/2
            #    patience_counter = 0
            # if optimizer.param_groups[0]['lr'] < lr/40:
            #    print('===LR too small, stop traning ====')
            #    break

    # dereplicate state
    state = nnx.state((model, optimizer))
    state = jax.device_get(state)
    nnx.update((model, optimizer), state)
    if config["has_aux"]:
        return model, metrics_history["train_loss"][-1], metrics_history["train_auxloss"][-1]

    return model, metrics_history["train_loss"][-1]


def save_model(model: Callable, file_path: str, file_name: str):
    """Save Flax model.

    Function for saving a Flax NNX neural network model.

    Args:
        model: Flax model to save.
        file_path: Absolute path where model is to be saved.
    """
    # Create path
    path = Path(file_path + "/" + file_name + ".pkl")
    path.parent.mkdir(parents=True, exist_ok=True)

    # Get the model state (e.g. parameter values)
    state = nnx.state(model)

    # Convert to a pure dictionary
    # This removes VariableState objects and leaves only Arrays
    pure_dict = to_state_dict(state)

    # Save the dictionary (e.g., as a .npy file)
    # with open(path, "wb") as f:
    #    jnp.save(f, pure_dict)

    with open(path, "wb") as pickle_file:
        pickle.dump(pure_dict, pickle_file)


def load_model(model: Callable, file_path: str, file_name: str):
    """Load Flax model.

    Function for loading a Flax NNX neural network model.

    Args:
        model: Flax model to load.
        file_path: Absolute path where model is saved.
        file_name: Filename to load from.
    """
    # Use model instance to serve as the target structure
    state_target = nnx.state(model)

    # Load the saved pure dictionary
    path = Path(file_path + "/" + file_name + ".pkl")
    # with open(path, "rb") as f:
    #    loaded_pure_dict = jnp.load(f, allow_pickle=True).item()
    with open(path, "rb") as pickle_file:
        loaded_pure_dict = pickle.load(pickle_file)

    # print(f"Loaded dict: {loaded_pure_dict}")

    # Restore the state into the target structure
    restored_state = from_state_dict(state_target, loaded_pure_dict)

    # Update the model with the loaded state
    nnx.update(model, restored_state)

    return model
