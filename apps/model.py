#!/usr/bin/env python
# -*- coding: utf-8 -*-

from functools import partial

import numpy as np
from scipy import sparse, stats


def identity(x):
    return x


def _getrvs(dist, **kwargs):
    distribution = getattr(stats, dist)
    return partial(distribution(**kwargs).rvs)


class Input:
    def __init__(self, N_u, N_x, input_scale, seed=None):
        if seed is not None:
            np.random.seed(seed=seed)
        self.Win = np.random.uniform(-input_scale, input_scale, (N_x, N_u))

    def __call__(self, u):
        return np.dot(self.Win, u)


class Reservoir:
    def __init__(
        self,
        N_x,
        density,
        rho,
        activation_func,
        leaking_rate,
        seed=None,
        randomize_initial_state=False,
    ):
        self.N_x = N_x
        self.density = density
        self.rho = rho
        self.activation_func = activation_func
        self.alpha = leaking_rate
        self.seed = seed
        if seed is not None:
            np.random.seed(self.seed)
        self.W = self.make_connection(N_x, density, rho)
        self.x = np.zeros(N_x)
        if randomize_initial_state:
            self.reset_reservoir_state(randomize_initial_state)

    def make_connection(self, N_x, density, rho):
        rvs = _getrvs("uniform")
        connection = sparse.random(N_x, N_x, density=density, data_rvs=rvs)
        W = connection.toarray()

        rec_scale = 1.0
        W *= np.random.uniform(-rec_scale, rec_scale, (N_x, N_x))

        eigv_list = np.linalg.eig(W)[0]
        sp_radius = np.max(np.abs(eigv_list))

        W *= rho / sp_radius

        return W

    def resample_connection(self):
        self.W = self.make_connection(self.N_x, self.density, self.rho)

    def __call__(self, x_in):
        # self.x = self.x.reshape(-1, 1)
        self.x = (1.0 - self.alpha) * self.x + self.alpha * self.activation_func(
            np.dot(self.W, self.x) + x_in
        )
        return self.x

    def reset_reservoir_state(self, randomize_initial_state=False):
        if randomize_initial_state:
            self.x = np.random.uniform(-1, 1, self.N_x)
        else:
            self.x = np.zeros(self.N_x)


class Output:
    def __init__(self, N_x, N_y, seed=None):
        if seed is not None:
            np.random.seed(seed=seed)
        self.Wout = np.random.normal(size=(N_y, N_x))

    def __call__(self, x):
        return np.dot(self.Wout, x)

    def setweight(self, Wout_opt):
        self.Wout = Wout_opt


class Feedback:
    def __init__(self, N_y, N_x, fb_scale, seed=None):
        if seed is not None:
            np.random.seed(seed=seed)
        self.Wfb = np.random.uniform(-fb_scale, fb_scale, (N_x, N_y))

    def __call__(self, y):
        return np.dot(self.Wfb, y)


class Pseudoinv:
    def __init__(self, N_x, N_y):
        self.X = np.empty((N_x, 0))
        self.D = np.empty((N_y, 0))

    def __call__(self, d, x):
        x = np.reshape(x, (-1, 1))
        d = np.reshape(d, (-1, 1))
        self.X = np.hstack((self.X, x))
        self.D = np.hstack((self.D, d))

    def get_Wout_opt(self):
        Wout_opt = np.dot(self.D, np.linalg.pinv(self.X))
        return Wout_opt


class Tikhonov:
    def __init__(self, N_x, N_y, beta):
        self.beta = beta
        self.X_XT = np.zeros((N_x, N_x))
        self.D_XT = np.zeros((N_y, N_x))
        self.N_x = N_x

    def __call__(self, d, x):
        x = np.reshape(x, (-1, 1))
        d = np.reshape(d, (-1, 1))
        self.X_XT += np.dot(x, x.T)
        self.D_XT += np.dot(d, x.T)

    def get_Wout_opt(self):
        X_pseudo_inv = np.linalg.inv(self.X_XT + self.beta * np.identity(self.N_x))
        Wout_opt = np.dot(self.D_XT, X_pseudo_inv)
        return Wout_opt


class RLS:
    def __init__(self, N_x, N_y, delta, lam, update):
        self.delta = delta
        self.lam = lam
        self.update = update
        self.P = (1.0 / self.delta) * np.eye(N_x, N_x)
        self.Wout = np.zeros([N_y, N_x])

    def __call__(self, d, x):
        x = np.reshape(x, (-1, 1))
        d = np.reshape(d, (-1, 1))
        for _ in np.arange(self.update):
            v = d - np.dot(self.Wout, x)
            gain = 1 / self.lam * np.dot(self.P, x)
            gain = gain / (1 + 1 / self.lam * np.dot(np.dot(x.T, self.P), x))
            self.P = 1 / self.lam * (self.P - np.dot(np.dot(gain, x.T), self.P))
            self.Wout += np.dot(v, gain.T)

        return self.Wout


class ESN:
    def __init__(
        self,
        N_u,
        N_y,
        N_x,
        density=0.05,
        input_scale=1.0,
        rho=0.95,
        activation_func=np.tanh,
        fb_scale=None,
        fb_seed=None,
        noise_level=None,
        leaking_rate=1.0,
        output_func=identity,
        inv_output_func=identity,
        classification=False,
        average_window=None,
    ):
        self.Input = Input(N_u, N_x, input_scale)
        self.Reservoir = Reservoir(N_x, density, rho, activation_func, leaking_rate)
        self.Output = Output(N_x, N_y)
        self.N_u = N_u
        self.N_y = N_y
        self.N_x = N_x
        self.y_prev = np.zeros(N_y)
        self.output_func = output_func
        self.inv_output_func = inv_output_func
        self.classification = classification

        if fb_scale is None:
            self.Feedback = None
        else:
            self.Feedback = Feedback(N_y, N_x, fb_scale, fb_seed)

        if noise_level is None:
            self.noise = None
        else:
            # np.random.seed(seed=0)
            self.noise = np.random.uniform(-noise_level, noise_level, (self.N_x, 1))

        if classification:
            if average_window is None:
                raise ValueError("Window for time average is not given!")
            else:
                self.window = np.zeros((average_window, N_x))

    def train(self, U, D, optimizer, trans_len=None):
        train_len = len(U)
        if trans_len is None:
            trans_len = 0
        Y = []

        for n in range(train_len):
            x_in = self.Input(U[n])

            if self.Feedback is not None:
                x_back = self.Feedback(self.y_prev)
                x_in += x_back

            if self.noise is not None:
                x_in += self.noise

            x = self.Reservoir(x_in)

            if self.classification:
                self.window = np.append(self.window, x.reshape(1, -1), axis=0)
                self.window = np.delete(self.window, 0, 0)
                x = np.average(self.window, axis=0)

            d = D[n]
            d = self.inv_output_func(d)

            if n > trans_len:
                optimizer(d, x)

            y = self.Output(x)
            Y.append(self.output_func(y))
            self.y_prev = d

        self.Output.setweight(optimizer.get_Wout_opt())

        return np.array(Y)

    def predict(self, U, return_X=False):
        test_len = len(U)
        Y_pred = []
        X = np.empty((0, self.N_x))

        for n in range(test_len):
            x_in = self.Input(U[n])

            if self.Feedback is not None:
                x_back = self.Feedback(self.y_prev)
                x_in += x_back

            x = self.Reservoir(x_in)
            X = np.vstack((X, x))

            if self.classification:
                self.window = np.append(self.window, x.reshape(1, -1), axis=0)
                self.window = np.delete(self.window, 0, 0)
                x = np.average(self.window, axis=0)

            y_pred = self.Output(x)
            Y_pred.append(self.output_func(y_pred))
            self.y_prev = y_pred

        if return_X:
            return np.array(Y_pred), X
        return np.array(Y_pred)

    def run(self, U):
        test_len = len(U)
        Y_pred = []
        y = U[0]

        for _ in range(test_len):
            x_in = self.Input(y)

            if self.Feedback is not None:
                x_back = self.Feedback(self.y_prev)
                x_in += x_back

            x = self.Reservoir(x_in)

            y_pred = self.Output(x)
            Y_pred.append(self.output_func(y_pred))
            y = y_pred
            self.y_prev = y

        return np.array(Y_pred)

    def adapt(self, U, D, optimizer, warmup=-1):
        data_len = len(U)
        Y_pred = []
        Wout_abs_mean = []

        Wout = optimizer.Wout
        for n in np.arange(0, data_len, 1):
            x_in = self.Input(U[n])
            x = self.Reservoir(x_in)
            d = D[n]
            d = self.inv_output_func(d)

            if n > warmup:
                Wout = optimizer(d, x)

            y = np.dot(Wout, x)
            Y_pred.append(y)
            Wout_abs_mean.append(np.mean(np.abs(Wout)))

        self.Output.setweight(Wout)

        return np.array(Y_pred), np.array(Wout_abs_mean)

    def reset_reservoir_state(self, randomize_initial_state=False):
        self.Reservoir.reset_reservoir_state(randomize_initial_state)


class reBASICS:
    def __init__(
        self,
        N_u,
        N_y,
        N_x,
        N_module,
        density=0.05,
        input_scale=1.0,
        rho=0.95,
        activation_func=np.tanh,
        noise_level=None,
        leaking_rate=1.0,
        output_func=identity,
        inv_output_func=identity,
    ):
        self.Input = Input(N_u, N_x, input_scale)
        self.Reservoirs = [
            Reservoir(
                N_x,
                density,
                rho,
                activation_func,
                leaking_rate,
                randomize_initial_state=True,
            )
            for _ in range(N_module)
        ]
        self.Output = Output(N_module, N_y)
        self.N_u = N_u
        self.N_y = N_y
        self.N_x = N_x
        self.N_module = N_module
        self.y_prev = np.zeros(N_y)
        self.output_func = output_func
        self.inv_output_func = inv_output_func
        self.noise_level = noise_level

        if noise_level is None:
            self.noise = None
        else:
            # np.random.seed(seed=0)
            self.noise = np.random.uniform(-noise_level, noise_level, (self.N_x, 1))

    def _is_active(self, reservoir, U, threshold, span):
        X = np.array([reservoir(self.Input(u)) for u in U])
        X = X[span[0] : span[1]]
        x = X[:, 0]
        return np.max(x) - np.min(x) > threshold
        # for x in X.T:
        #     if np.max(x) - np.min(x) < threshold:
        #         return False
        # return True

    def resample_inactive_module(self, U, threshold=0.01, span=(0, None)):
        for i, r in enumerate(self.Reservoirs):
            while not self._is_active(r, U, threshold, span):
                r.resample_connection()
                print("Resampled module %d" % i)

    def adapt(self, U, D, optimizer, warmup=-1):
        data_len = len(U)
        Y_pred = []
        Wout_abs_mean = []

        Wout = optimizer.Wout
        for n in np.arange(0, data_len, 1):
            x_in = self.Input(U[n])
            if self.noise_level:
                x_in += self.noise_level * np.random.normal(size=(self.N_x,))
            states = [r(x_in) for r in self.Reservoirs]
            x = np.array([s[0] for s in states])
            d = D[n]
            d = self.inv_output_func(d)

            if n > warmup:
                Wout = optimizer(d, x)

            y = np.dot(Wout, x)
            Y_pred.append(y)
            Wout_abs_mean.append(np.mean(np.abs(Wout)))

        self.Output.setweight(Wout)

        return np.array(Y_pred), np.array(Wout_abs_mean)

    def predict(self, U, return_X=False):
        test_len = len(U)
        Y_pred = []
        X = np.empty((0, self.N_module))

        for n in range(test_len):
            x_in = self.Input(U[n])
            if self.noise_level:
                x_in += self.noise_level * np.random.normal(size=(self.N_x,))
            states = [r(x_in) for r in self.Reservoirs]
            x = np.array([s[0] for s in states])
            X = np.vstack((X, x))

            y_pred = self.Output(x)
            Y_pred.append(self.output_func(y_pred))
            self.y_prev = y_pred

        if return_X:
            return np.array(Y_pred), X
        return np.array(Y_pred)

    def reset_reservoir_state(self, randomize_initial_state=False):
        for r in self.Reservoirs:
            r.reset_reservoir_state(randomize_initial_state)
