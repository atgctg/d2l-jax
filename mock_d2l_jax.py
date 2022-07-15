from matplotlib_inline import backend_inline
import matplotlib.pyplot as plt
import time
import jax
from jax import random, numpy as jnp
import torch
from torch.utils import data
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds


def use_svg_display():
    """Use the svg format to display a plot in Jupyter.

    Defined in :numref:`sec_calculus`"""
    backend_inline.set_matplotlib_formats("svg")


def set_figsize(figsize=(3.5, 2.5)):
    """Set the figure size for matplotlib.

    Defined in :numref:`sec_calculus`"""
    use_svg_display()
    plt.rcParams["figure.figsize"] = figsize


def set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend):
    """Set the axes for matplotlib.

    Defined in :numref:`sec_calculus`"""
    axes.set_xlabel(xlabel)
    axes.set_ylabel(ylabel)
    axes.set_xscale(xscale)
    axes.set_yscale(yscale)
    axes.set_xlim(xlim)
    axes.set_ylim(ylim)
    if legend:
        axes.legend(legend)
    axes.grid()


def plot(
    X,
    Y=None,
    xlabel=None,
    ylabel=None,
    legend=None,
    xlim=None,
    ylim=None,
    xscale="linear",
    yscale="linear",
    fmts=("-", "m--", "g-.", "r:"),
    figsize=(3.5, 2.5),
    axes=None,
):
    """Plot data points.

    Defined in :numref:`sec_calculus`"""
    if legend is None:
        legend = []

    set_figsize(figsize)
    axes = axes if axes else plt.gca()

    # Return True if `X` (tensor or list) has 1 axis
    def has_one_axis(X):
        return (
            hasattr(X, "ndim")
            and X.ndim == 1
            or isinstance(X, list)
            and not hasattr(X[0], "__len__")
        )

    if has_one_axis(X):
        X = [X]
    if Y is None:
        X, Y = [[]] * len(X), X
    elif has_one_axis(Y):
        Y = [Y]
    if len(X) != len(Y):
        X = X * len(Y)
    axes.cla()
    for x, y, fmt in zip(X, Y, fmts):
        if len(x):
            axes.plot(x, y, fmt)
        else:
            axes.plot(y, fmt)
    set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend)


# 3.1
class Timer:  # @save
    """Record multiple running times."""

    def __init__(self):
        self.times = []
        self.start()

    def start(self):
        """Start the timer."""
        self.tik = time.time()

    def stop(self):
        """Stop the timer and record the time in a list."""
        self.times.append(time.time() - self.tik)
        return self.times[-1]

    def avg(self):
        """Return the average time."""
        return sum(self.times) / len(self.times)

    def sum(self):
        """Return the sum of time."""
        return sum(self.times)

    def cumsum(self):
        """Return the accumulated time."""
        return np.array(self.times).cumsum().tolist()


# 3.2

def synthetic_data(w, b, num_examples, key):  # @save
    """Generate y = Xw + b + noise."""
    key1, key2 = random.split(key)
    X = random.normal(key1, (num_examples, len(w))) * 1 + 0
    y = jnp.matmul(X, w) + b
    y += random.normal(key2, y.shape) * 0.01 + 0
    return X, y.reshape((-1, 1))


def linreg(X, w, b):  # @save
    """The linear regression model."""
    return jnp.matmul(X, w) + b


def squared_loss(y_hat, y):  # @save
    """Squared loss."""
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2

@jax.jit
def sgd(params, grads, lr):  # @save
    """Minibatch stochastic gradient descent."""
    params = jax.tree_util.tree_map(lambda p, g: p - lr * g, params, grads)
    return params


# 3.3

# Since JAX doesn't have a dataloader, we'll use PyTorch's
def load_array(data_arrays, batch_size, is_train=True):  # @save
    """Construct a PyTorch data iterator."""
    # convert JAX arrays to PyTorch tensors
    data_arrays = (torch.from_numpy(np.array(x)) for x in data_arrays)

    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size, shuffle=is_train)


# 3.5

def get_fashion_mnist_labels(labels):  # @save
    """Return text labels for the Fashion-MNIST dataset."""
    text_labels = [
        "t-shirt",
        "trouser",
        "pullover",
        "dress",
        "coat",
        "sandal",
        "shirt",
        "sneaker",
        "bag",
        "ankle boot",
    ]
    return [text_labels[int(i)] for i in labels]


def show_images(imgs, num_rows, num_cols, titles=None, scale=1.5):  # @save
    """Plot a list of images."""
    figsize = (num_cols * scale, num_rows * scale)
    _, axes = plt.subplots(num_rows, num_cols, figsize=figsize)
    axes = axes.flatten()
    for i, (ax, img) in enumerate(zip(axes, imgs)):
        ax.imshow(img)
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        if titles:
            ax.set_title(titles[i])
    return axes


def load_data_fashion_mnist(batch_size, resize=None):  # @save
    """Download the Fashion-MNIST dataset and then load it into memory."""
    mnist_train, mnist_test = tf.keras.datasets.fashion_mnist.load_data()
    # Divide all numbers by 255 so that all pixel values are between
    # 0 and 1, add a batch dimension at the last. And cast label to int32
    process = lambda X, y: (tf.expand_dims(X, axis=3) / 255, tf.cast(y, dtype="int32"))
    resize_fn = lambda X, y: (
        tf.image.resize_with_pad(X, resize, resize) if resize else X,
        y,
    )
    return (
        tfds.as_numpy(
            tf.data.Dataset.from_tensor_slices(process(*mnist_train))
            .batch(batch_size)
            .shuffle(len(mnist_train[0]))
            .map(resize_fn)
        ),
        tfds.as_numpy(
            tf.data.Dataset.from_tensor_slices(process(*mnist_test))
            .batch(batch_size)
            .map(resize_fn)
        ),
    )
