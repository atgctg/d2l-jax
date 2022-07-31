import sys
import collections
import inspect
import time
from IPython import display

from matplotlib_inline import backend_inline
import matplotlib.pyplot as plt

import numpy as np

import jax
from jax import numpy as jnp, random, grad, vmap, jit
from flax import linen as nn
from flax.training.train_state import TrainState
import optax

import tensorflow as tf
import tensorflow_datasets as tfds

d2l = sys.modules[__name__]


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


# 3.2 (make sure these are in sync with oo-design.ipynb)

def add_to_class(Class):  # @save
    def wrapper(obj):
        setattr(Class, obj.__name__, obj)

    return wrapper


class HyperParameters:  # @save
    def save_hyperparameters(self, ignore=[]):
        """Save function arguments into class attributes."""
        frame = inspect.currentframe().f_back
        _, _, _, local_vars = inspect.getargvalues(frame)
        self.hparams = {
            k: v
            for k, v in local_vars.items()
            if k not in set(ignore + ["self"]) and not k.startswith("_")
        }
        for k, v in self.hparams.items():
            setattr(self, k, v)


class ProgressBoard(HyperParameters):
    """Plot data points in animation.

    Defined in :numref:`sec_oo-design`"""

    def __init__(
        self,
        xlabel=None,
        ylabel=None,
        xlim=None,
        ylim=None,
        xscale="linear",
        yscale="linear",
        ls=["-", "--", "-.", ":"],
        colors=["C0", "C1", "C2", "C3"],
        fig=None,
        axes=None,
        figsize=(3.5, 2.5),
        display=True,
    ):
        self.save_hyperparameters()

    def draw(self, x, y, label, every_n=1):
        """Defined in :numref:`sec_utils`"""
        Point = collections.namedtuple("Point", ["x", "y"])
        if not hasattr(self, "raw_points"):
            self.raw_points = collections.OrderedDict()
            self.data = collections.OrderedDict()
        if label not in self.raw_points:
            self.raw_points[label] = []
            self.data[label] = []
        points = self.raw_points[label]
        line = self.data[label]
        points.append(Point(x, y))
        if len(points) != every_n:
            return
        mean = lambda x: sum(x) / len(x)
        line.append(Point(mean([p.x for p in points]), mean([p.y for p in points])))
        points.clear()
        if not self.display:
            return
        use_svg_display()
        if self.fig is None:
            self.fig = plt.figure(figsize=self.figsize)
        plt_lines, labels = [], []
        for (k, v), ls, color in zip(self.data.items(), self.ls, self.colors):
            plt_lines.append(
                plt.plot([p.x for p in v], [p.y for p in v], linestyle=ls, color=color)[
                    0
                ]
            )
            labels.append(k)
        axes = self.axes if self.axes else plt.gca()
        if self.xlim:
            axes.set_xlim(self.xlim)
        if self.ylim:
            axes.set_ylim(self.ylim)
        if not self.xlabel:
            self.xlabel = self.x
        axes.set_xlabel(self.xlabel)
        axes.set_ylabel(self.ylabel)
        axes.set_xscale(self.xscale)
        axes.set_yscale(self.yscale)
        axes.legend(plt_lines, labels)
        display.display(self.fig)
        display.clear_output(wait=True)


class Module(nn.Module):
    """Defined in :numref:`sec_oo-design`"""
    plot_train_per_epoch: int = 2
    plot_valid_per_epoch: int = 1
    board: ProgressBoard = ProgressBoard()

    def loss(self, y_hat, y):
        raise NotImplementedError

    def __call__(self, X, *args, **kwargs):
        raise NotImplementedError

    def plot(self, key, value, train):
        """Plot a point in animation."""
        assert hasattr(self, "trainer"), "Trainer is not inited"
        self.board.xlabel = "epoch"
        if train:
            x = self.trainer.train_batch_idx / self.trainer.num_train_batches
            n = self.trainer.num_train_batches / self.plot_train_per_epoch
        else:
            x = self.trainer.epoch + 1
            n = self.trainer.num_val_batches / self.plot_valid_per_epoch

        self.board.draw(x, value, ("train_" if train else "val_") + key, every_n=int(n))

    def training_step(self, params, batch):
        loss, grads = jax.value_and_grad(self.loss)(params, *batch[:-1], batch[-1])
        self.plot("loss", loss, train=True)
        return loss, grads

    def validation_step(self, params, batch):
        loss = self.loss(params, *batch[:-1], batch[-1])
        self.plot("loss", loss, train=False)

    def configure_optimizers(self):
        raise NotImplementedError

    def configure_optimizers(self):
        """Defined in :numref:`sec_classification`"""
        return optax.sgd(learning_rate=self.lr)


# 3.3


class DataModule(HyperParameters):
    """Defined in :numref:`sec_oo-design`"""

    def __init__(self, root="../data"):
        self.save_hyperparameters()

    def get_dataloader(self, train):
        raise NotImplementedError

    def train_dataloader(self):
        return self.get_dataloader(train=True)

    def val_dataloader(self):
        return self.get_dataloader(train=False)

    def get_tensorloader(self, tensors, train, indices=slice(0, None)):
        """Defined in :numref:`sec_synthetic-regression-data`"""
        tensors = tuple(a[indices] for a in tensors)
        shuffle_buffer = tensors[0].shape[0] if train else 1
        return tfds.as_numpy(
            tf.data.Dataset.from_tensor_slices(tensors)
            .shuffle(buffer_size=shuffle_buffer)
            .batch(self.batch_size)
        )


class Trainer(HyperParameters):
    """Defined in :numref:`sec_oo-design`"""

    def __init__(self, max_epochs, num_gpus=0, gradient_clip_val=0):
        self.save_hyperparameters()
        assert num_gpus == 0, "No GPU support yet"

    def prepare_data(self, data):
        self.train_dataloader = data.train_dataloader()
        self.val_dataloader = data.val_dataloader()
        self.num_train_batches = len(self.train_dataloader)
        self.num_val_batches = (
            len(self.val_dataloader) if self.val_dataloader is not None else 0
        )

    def prepare_model(self, model):
        model.trainer = self
        model.board.xlim = [0, self.max_epochs]
        self.model = model

    def fit(self, params, model, data):
        self.prepare_data(data)
        self.prepare_model(model)
        self.optim = model.configure_optimizers()
        self.state = TrainState.create(
            apply_fn=model.apply, params=params, tx=self.optim
        )

        self.epoch = 0
        self.train_batch_idx = 0
        self.val_batch_idx = 0
        for self.epoch in range(self.max_epochs):
            self.fit_epoch()

    def prepare_batch(self, batch):
        """Defined in :numref:`sec_linear_scratch`"""
        return batch

    def fit_epoch(self):
        """Defined in :numref:`sec_linear_scratch`"""

        for batch in self.train_dataloader:
            _, grads = self.model.training_step(
                self.state.params, self.prepare_batch(batch)
            )

            # todo: clip
            # if self.gradient_clip_val > 0:
            #     grads = self.clip_gradients(self.gradient_clip_val, grads)

            self.state = self.state.apply_gradients(grads=grads)
            self.train_batch_idx += 1

        if self.val_dataloader is None:
            return

        for batch in self.val_dataloader:
            self.model.validation_step(self.state.params, self.prepare_batch(batch))
            self.val_batch_idx += 1

    # todo: clip
    # def clip_gradients(self, grad_clip_val, grads):
    #     """Defined in :numref:`sec_rnn-scratch`"""
    #     grad_clip_val = tf.constant(grad_clip_val, dtype=tf.float32)
    #     new_grads = [tf.convert_to_tensor(grad) if isinstance(
    #         grad, tf.IndexedSlices) else grad for grad in grads]
    #     norm = tf.math.sqrt(sum((tf.reduce_sum(grad ** 2)) for grad in new_grads))
    #     if tf.greater(norm, grad_clip_val):
    #         for i, grad in enumerate(new_grads):
    #             new_grads[i] = grad * grad_clip_val / norm
    #         return new_grads
    #     return grads


# Use TensorFlow's dataloader, since JAX doesn't have one
class SyntheticRegressionData(DataModule, HyperParameters):  # @save
    def __init__(
        self, key, w, b, noise=0.01, num_train=1000, num_val=1000, batch_size=32
    ):
        super().__init__()
        self.save_hyperparameters()
        n = num_train + num_val
        key1, key2 = jax.random.split(key, 2)
        self.X = jax.random.normal(key1, (n, len(w)))
        noise = jax.random.normal(key2, (n, 1)) * noise
        self.y = jnp.matmul(self.X, w.reshape((-1, 1))) + b + noise

    def get_dataloader(self, train):
        i = slice(0, self.num_train) if train else slice(self.num_train, None)
        return self.get_tensorloader((self.X, self.y), train, i)


# Since JAX doesn't have a dataloader, we'll use PyTorch's
# def load_array(data_arrays, batch_size, is_train=True):  # @save
#     """Construct a PyTorch data iterator."""
#     # convert JAX arrays to PyTorch tensors
#     data_arrays = (torch.from_numpy(np.array(x)) for x in data_arrays)

#     dataset = data.TensorDataset(*data_arrays)
#     return data.DataLoader(dataset, batch_size, shuffle=is_train)


###############################################################################

# 3.5

# def get_fashion_mnist_labels(labels):  # @save
#     """Return text labels for the Fashion-MNIST dataset."""
#     text_labels = [
#         "t-shirt",
#         "trouser",
#         "pullover",
#         "dress",
#         "coat",
#         "sandal",
#         "shirt",
#         "sneaker",
#         "bag",
#         "ankle boot",
#     ]
#     return [text_labels[int(i)] for i in labels]


# def show_images(imgs, num_rows, num_cols, titles=None, scale=1.5):  # @save
#     """Plot a list of images."""
#     figsize = (num_cols * scale, num_rows * scale)
#     _, axes = plt.subplots(num_rows, num_cols, figsize=figsize)
#     axes = axes.flatten()
#     for i, (ax, img) in enumerate(zip(axes, imgs)):
#         ax.imshow(img)
#         ax.axes.get_xaxis().set_visible(False)
#         ax.axes.get_yaxis().set_visible(False)
#         if titles:
#             ax.set_title(titles[i])
#     return axes


# def load_data_fashion_mnist(batch_size, resize=None):  # @save
#     """Download the Fashion-MNIST dataset and then load it into memory."""
#     mnist_train, mnist_test = tf.keras.datasets.fashion_mnist.load_data()
#     # Divide all numbers by 255 so that all pixel values are between
#     # 0 and 1, add a batch dimension at the last. And cast label to int32
#     process = lambda X, y: (tf.expand_dims(X, axis=3) / 255, tf.cast(y, dtype="int32"))
#     resize_fn = lambda X, y: (
#         tf.image.resize_with_pad(X, resize, resize) if resize else X,
#         y,
#     )
#     return (
#         tfds.as_numpy(
#             tf.data.Dataset.from_tensor_slices(process(*mnist_train))
#             .batch(batch_size)
#             .shuffle(len(mnist_train[0]))
#             .map(resize_fn)
#         ),
#         tfds.as_numpy(
#             tf.data.Dataset.from_tensor_slices(process(*mnist_test))
#             .batch(batch_size)
#             .map(resize_fn)
#         ),
#     )


# 4.2

def show_images(imgs, num_rows, num_cols, titles=None, scale=1.5):
    """Plot a list of images.

    Defined in :numref:`sec_utils`"""
    figsize = (num_cols * scale, num_rows * scale)
    _, axes = d2l.plt.subplots(num_rows, num_cols, figsize=figsize)
    axes = axes.flatten()
    for i, (ax, img) in enumerate(zip(axes, imgs)):
        try:
            img = d2l.numpy(img)
        except:
            pass
        ax.imshow(img)
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        if titles:
            ax.set_title(titles[i])
    return axes
# 6.7

def cpu():  #@save
    return jax.devices('cpu')[0]

def gpu(i=0):  #@save
    return jax.devices('gpu')[i]

def num_gpus():  #@save
    try:
        return jax.device_count('gpu')
    except:
        return 0

def try_gpu(i=0):  #@save
    """Return gpu(i) if exists, otherwise return cpu()."""
    if num_gpus() >= i + 1:
        return gpu(i)
    return cpu()

def try_all_gpus():  #@save
    """Return all available GPUs, or [cpu(),] if no GPU exists."""
    return [gpu(i) for i in range(num_gpus())]

@d2l.add_to_class(d2l.Trainer)  #@save
def __init__(self, max_epochs, num_gpus=0, gradient_clip_val=0):
    self.save_hyperparameters()
    self.gpus = [d2l.gpu(i) for i in range(min(num_gpus, d2l.num_gpus()))]

@d2l.add_to_class(d2l.Trainer)  #@save
def prepare_batch(self, batch):
    if self.gpus:
        batch = [jax.device_put(a, device=self.gpus[0]) for a in batch]
    return batch