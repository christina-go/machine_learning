# from "Machine Learning with Python for Everyone" by Mark E Fenner - CNG

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import pandas as pd
import patsy

import itertools as it
import collections as co
import functools as ft
import os.path as osp

import glob
import textwrap

import warnings

warnings.filterwarnings("ignore")
# some warnings are stubborn in the extreme, we don't want them in this exercise


def warn(*args, **kwargs):
    pass


warnings.warn = warn

# config related

np.set_printoptions(precision=4, suppress=True)
pd.options.display.float_format = '{:20,.4f}'.format

# there are good reasons not to do this in any real production code
# for our purposes, we do

np.random.seed(42)

# default is [6.4, 4.8] (4:3)
mpl.rcParams['figure.figsize'] = [4.0, 3.0]

# turn on latex tables
pd.set_option('display.latex.repr', True)
# monkey-patch for centering Out[] Dataframes


def _repr_latex_(self):
    return "{\centering\n%s\n\medskip}" % self.to_latex()


pd.DataFrame._repr_latex = _repr_latex_

# only used once
markers = it.cycle(['+', '^', 'o', '_', '*', 'd', 'x', 's'])

# handy helper for displaying stuff
from IPython.display import Image

#
# sklearn's packaging is very java-esque :(
#

from sklearn import (cluster, datasets, decomposition, discriminant_analysis,
                     dummy, ensemble, feature_selection as ftr_sel,
                     linear_model, metrics, model_selection as skms, multiclass
                     as skmulti, naive_bayes, neighbors, pipeline,
                     preprocessing as skpre, svm, tree)

# the punch line is to predict for a large grid of data points
# http://scikit-learn.org/stable/auto_examples/neighbors
# /plot_classification.html


def plot_boundary(ax, data, tgt, model, dims, grid_step=0.1):
    # grab a 2D view of the data and get limits
    twoD = data[:, list(dims)]
    min_x1, min_x2 = np.min(twoD, axis=0) + 2 * grid_step
    max_x1, max_x2 = np.max(twoD, axis=0) - grid_step

    # make a grid of points and predict at them
    xs, ys = np.mgrid[min_x1:max_x1:grid_step, min_x2:max_x2:grid_step]

    grid_points = np.c_[xs.ravel(), ys.ravel()]
    # warning: non-cv fit
    preds = model.fit(twoD, tgt).predict(grid_points).reshape(xs.shape)

    # plot the predictions at the grid points
    ax.pcolormesh(xs, ys, preds, cmap=plt.cm.coolwarm)
    ax.set_xlim(min_x1, max_x1)  # - grid_step)
    ax.set_ylim(min_x2, max_x2)  # - grid_step)


def plot_separator(model, xs, ys, label='', ax=None):
    ''' xs, ys are 1-D b/c contour and decision_function use comparable packaging'''

    if ax is None:
        ax = plt.gca()

        xy = np.cartesian_product(xs, ys)
        z_shape = (xs.size, ys.size)  # using .size since 1D
        zs = model.decision_function(xy).reshape(z_shape)

        contours = ax.contour(xs, ys, zs,
                              colors='k', levels=[0],
                              linestyles=['-'])

        fmt = {contours.levels[0]: label}
        labels = ax.clabel(contours, fmt=fmt, inline_spacing=10)
        [l.set_rotation(-90) for l in labels]


def high_school_style(ax):
    ' helper to define an axis to look like a typical school plot '
    ax.spines['left'].set_position(('data', 0.0))
    ax.spines['bottom'].set_position(('data', 0.0))
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    def make_ticks(lims):
        lwr, upr = sorted(lims)  #x/ylims can be inverted in mpl
        lwr = np.round(lwr).astype('int')  # can return np objs
        upr = np.round(upr).astype('int')

        if lwr * upr < 0:
            return list(range(lwr, 0)) + list(range(1, upr + 1))
        else:
            return list(range(lwr, upr + 1))

    import matplotlib.ticker as ticker
    xticks = make_ticks(ax.get_xlim())
    yticks = make_ticks(ax.get_ylim())

    ax.xaxis.set_major_locator(ticker.FixedLocator(xticks))
    ax.yaxis.set_major_locator(ticker.FixedLocator(yticks))

    ax.set_aspect('equal')


def get_model_name(model):
    ' return name of model (class) as a string '
    return str(model.__class__).split('.')[-1][:-2]


def rdot(w, x):
    ' apply np.dot on swapped args '
    return np.dot(x, w)


from sklearn.base import BaseEstimator, ClassifierMixin


class DLDA(BaseEstimator, ClassifierMixin):
    def __init__(self):
        pass

    def fit(self, train_ftrs, train_tgts):
        self.uniq_tgts = np.unique(train_tgts)
        self.means, self.priors = {}, {}

        self.var = train_ftrs.var(axis=0)  # biased
        for tgt in self.uniq_tgts:
            cases = train_ftrs[train_tgts == tgt]
            self.means[tgt] = cases.mean(axis=0)
            self.priors[tgt] = len(cases) / len(train_ftrs)

        return self

    def predict(self, test_ftrs):
        disc = np.empty((test_ftrs.shape[0]))

        for tgt in self.uniq_tgts:
            # technically. the maha_dist is sqrt() of this:
            mahalanobis_dists = ((test_ftrs - self.means[tgt])**2 / self.var)
            disc[:, tgt] = (-np.sum(mahalanobis_dists, axis=1) +
                            2 * np.log(self.priors[tgt]))

        return np.argmax(disc, axis=1)


def plot_lines_and_projections(axes, lines, points, xs):
    data_xs, data_ys = points[:, 0], points[:, 1]
    mean = np.mean(points, axis=0, keepdims=True)
    centered_data = points - mean

    for (m, b), ax in zip(lines, axes):
        mb_line = m * xs + b
        v_line = np.array([[1, 1 / m if m else 0]])

        ax.plot(data_xs, data_ys, 'r.')  # uncentered
        ax.plot(xs, mb_line, 'y')  # uncentered
        ax.plot(*mean.T, 'ko')

        # centered data makes this math much easier!
        # this is length on yellow line from red to blue
        # distance from mean to projected point
        y_lengths = centered_data.dot(v_line.T) / v_line.dot(v_line.T)
        projs = y_lengths.dot(v_line)

        # decenter (back to original coordinates)
        final = projs + mean
        ax.plot(*final.T, 'b')

        # connect points to projections
        from matplotlib import collections as mc
        proj_lines = mc.LineCollection(zip(points, final))
        ax.add_collection(proj_lines)

        hypots = zip(points, np.broadcast_to(mean, points.shape))
        mean_lines = mc.LineCollection(zip(points, final))
        ax.add_collection(mean_lines)


# adding an orientation would be nice
def sane_quiver(vs, ax=None, colors=None, origin=(0, 0)):
    ''' plot row vectors from origin '''
    vs = np.asarray(vs)
    assert vs.ndim == 2 and vs.shape[1] == 2  # ensure column vectors
    n = vs.shape[0]
    if not ax: ax = plt.gca()

    # zs = np.zeros(n)
    # zs = np.broadcast_to(origin, vs.shape)
    orig_x, orig_y = origin

    U = vs.T[0]  # column to rows, row[0] is xs
    V = vs.T[1]
    
    #making everything the same length for the quiver
    X = [orig_x for x in range(len(U))]
    Y = [orig_y for x in range(len(V))]

    props = {"angles": 'xy', 'scale': 1, 'scale_units': 'xy'}
    ax.quiver(X, Y, U, V, color='r', **props)

    ax.set_aspect('equal')
    # ax.set_axis_off()
    _min, _max = min(vs.min(), 0) - 1, max(0, vs.max()) + 1
    ax.set_xlim(_min, _max)
    ax.set_ylim(_min, _max)


def reweight(examples, weights):
    ''' convert weights to counts of examples using approximately two significant digits of weights.

        there are probably 100 reasons not to do this like this.
        top 2:
        1. boosting may require more precise values (or using randomization) to keep things unbiased
        2. this *really* expands the dataset to a significant degree (wastes resources)

    '''
    from math import gcd
    from functools import reduce

    # who needs repeated the least?
    min_wgt = min(weights)
    min_replicate = 1 / min_wgt  # eg .25 -> 4

    # compute naive duplication to 2 decimal places
    counts = (min_replicate * weights * 100).astype(np.int64)

    # trim duplication if we can
    our_gcd = reduce(gcd, counts)
    counts = counts // our_gcd

    # repeat is picky about type
    return np.repeat(examples, counts, axis=0)


#examples = np.array([1, 10, 20])
#weights = np.array([.25, .33, 1-(.25+.33)])
# print(pd.Series(reweight(examples, weights)))


def enumerate_outer(outer_seq):
    ''' repeat the outer idx based on len of inner '''
    return np.repeat(*zip(*enumerate(map(len, outer_seq))))


def np_array_fromiter(itr, shape, dtype=np.float64):
    ''' helper since np.fromiter only does 1D '''
    arr = np.empty(shape, dtype=dtype)
    for idx, itm in enumerate(itr):
        arr[idx] = itm
    return arr


# how do you figure out arcane code?
# work inside out, small inputs, pay attention to datatypes.
# try outter and righter calls with simpler inputs

# the difference with a "raw" np.meshgrid call is we stack these up in two columns
# of results (ie we make a table out of the pair arrays)


def np_cartesian_product(*arrays):
    ''' some numpy kung-fu to produce all
    possible combinations of input arrays '''

    ndim = len(arrays)
    return np.stack(np.meshgrid(*arrays), axis=-1).reshape(-1, ndim)
