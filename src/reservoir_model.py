import numpy as np
import pandas as pd


class ReservoirComputing:
    def __init__(
        self,
        n_reservoir=100,
        spectral_radius=0.95,
        leak_rate=1.0,
        input_scaling=0.1,
        regularization=1e-6,
        random_seed=23,
        weights_path=None,
        init_value=0,
    ):
        """
        Initialize the Reservoir Computing model (Echo State Network style).

        :param n_reservoir: Number of reservoir neurons.
        :param spectral_radius: Spectral radius for scaling the reservoir's weight matrix.
        :param leak_rate: Leaking rate for leaky integration (if < 1.0).
        :param input_scaling: Scaling factor for the input matrix.
        :param regularization: Ridge penalty (Tikhonov) for training the output weights.
        :param random_seed: For reproducibility.
        :param load_weights: If True, loads pre-trained weights from 'weights_path'.
        :param weights_path: Path to npz file with saved weights (Win, W, Wout).
        :param init_value: first observation of initial train time series.
        """
        self.n_reservoir = n_reservoir
        self.spectral_radius = spectral_radius
        self.leak_rate = leak_rate
        self.input_scaling = input_scaling
        self.regularization = regularization
        self.random_seed = random_seed
        self.init_value = init_value

        np.random.seed(self.random_seed)

        self.W_in = None
        self.W = None
        self.W_out = None

        if weights_path is not None:
            self._load_weights(weights_path)

    def _load_weights(self, path):
        """
        Load weights from an npz file (Win, W, Wout).
        """
        data = np.load(path, allow_pickle=True)
        self.W_in = data["W_in"]
        self.W = data["W"]
        self.W_out = data["W_out"]

    def save_weights(self, path):
        """
        Save weights to an npz file.
        """
        np.savez(path, W_in=self.W_in, W=self.W, W_out=self.W_out)

    def _init_weights(self, n_input, n_output):
        """
        Initialize the input and reservoir weights (only if not loaded).
        Spectral radius is adjusted for stability.
        """
        # Input weights
        # shape: (n_reservoir, n_input)
        self.W_in = (
            (np.random.rand(self.n_reservoir, n_input) - 0.5) * 2.0 * self.input_scaling
        )

        # Reservoir weights
        # shape: (n_reservoir, n_reservoir)
        W_raw = np.random.rand(self.n_reservoir, self.n_reservoir) - 0.5
        # Ensure largest absolute eigenvalue is spectral_radius
        rhoW = max(abs(np.linalg.eigvals(W_raw)))
        self.W = (W_raw / rhoW) * self.spectral_radius

        # Output weights (trained by ridge regression)
        # shape: (n_output, n_reservoir + n_input + 1)
        self.W_out = None

    def _update_state(self, state, input_vector):
        """
        Update reservoir state with the leaky-integration approach:
        r(t+1) = (1 - leak_rate)*r(t) + leak_rate * tanh( W_in * input + W * r(t) ).
        """
        pre_activation = np.dot(self.W_in, input_vector) + np.dot(self.W, state)
        new_state = (1.0 - self.leak_rate) * state + self.leak_rate * np.tanh(
            pre_activation
        )
        return new_state

    def fit(self, X_train, Y_train, washout=50):
        """
        Train the reservoir using ridge regression on the (reservoir_state, input) -> output mapping.

        :param X_train: shape (T, n_input)
        :param Y_train: shape (T, n_output)
        :param washout: number of initial steps to discard from training to let reservoir 'wash out'.
        """
        # X_train and Y_train must have the same time dimension T
        T, n_input = X_train.shape
        _, n_output = Y_train.shape

        # If W_in or W are not set, initialize them
        if self.W_in is None or self.W is None:
            self._init_weights(n_input, n_output)

        # Collect states
        reservoir_states = []
        # We'll store also the input & bias as part of extended state
        # so final feature is [1, input, reservoir_state]

        r = np.zeros(self.n_reservoir)
        for t in range(T):
            # update
            r = self._update_state(r, X_train[t])
            if t >= washout:
                # extended state
                ext_state = np.concatenate(
                    [np.ones(1), X_train[t], r]
                )  # shape = (1 + n_input + n_reservoir,)
                reservoir_states.append(ext_state)

        # Build big matrices to solve W_out
        M = np.array(reservoir_states)  # shape (T - washout, 1 + n_input + n_reservoir)
        # target
        Y_target = Y_train[washout:]  # shape (T - washout, n_output)

        # Ridge regression: W_out = Y_target^T * M^T * inv(M * M^T + reg*I)
        # but more easily done with (M^T M + reg I)^(-1) M^T Y_target
        # M shape => (samples, features), Y_target => (samples, n_output)

        regI = self.regularization * np.eye(M.shape[1])
        # (features, n_output) = (features, samples) dot (samples, n_output)
        self.W_out = np.linalg.solve(M.T.dot(M) + regI, M.T.dot(Y_target))
        # final shape => (1 + n_input + n_reservoir, n_output)

    def predict(self, X_init, forecast_steps=12, generative=True):
        """
        Make predictions in generative mode or single-step mode.

        :param X_init: shape (T, n_input), last known training inputs
                       or a single state to start generative predictions.
        :param forecast_steps: how many steps to predict forward
        :param generative: if True, each new output is fed back as next input
        :return: predictions of shape (forecast_steps, n_output)
        """

        if self.W_out is None:
            raise ValueError("Model is not trained. Call fit(...) first.")

        # we'll keep track of reservoir state
        r = np.zeros(self.n_reservoir)

        # "warm up" the reservoir with the last chunk from X_init
        for t in range(X_init.shape[0]):
            r = self._update_state(r, X_init[t])

        # The last known input for generative mode
        current_input = X_init[-1].copy()

        preds = []

        for step in range(forecast_steps):
            # extended state
            ext_state = np.concatenate([np.ones(1), current_input, r])
            # linear readout
            y_out = ext_state.dot(self.W_out)  # shape (n_output,)

            preds.append(y_out)

            # generative = feed the predicted y_out as the next input
            if generative:
                next_input = y_out  # shape (n_output,) -> must match input dimension if univariate
            else:
                # if not generative, user must provide external input?
                next_input = current_input  # placeholder

            # update reservoir state
            r = self._update_state(r, next_input)
            current_input = next_input

        return np.array(preds)

    # Back to real from log and get coef
    def back_to_real_and_coef(self, loged_array, last_train_idx):
        to_real = np.exp(loged_array.cumsum() + np.log(self.init_value))
        k = to_real[-1] / 300228.6 # Данные за 18 декабря 2024 - базовый уровень. to_real[last_train_idx]

        return to_real, k

def forecast_index(datadir='data/dom_index.csv', forecast=52, path_to_weights=None): 
    #data=data_preprocessing(path_to_data=datadir)
    data=pd.read_csv(datadir)
    series = data["log_return_close"].dropna().values
    if path_to_weights is not None:
        m = ReservoirComputing(weights_path=path_to_weights)
        pred = m.predict(
            np.array([[series[-1]]]), forecast_steps=forecast, generative=True
        )
        logged_array = np.concatenate((series, pred), axis=None)
        return m.back_to_real_and_coef(logged_array, len(data))[1]
    else:
        m = ReservoirComputing(
            n_reservoir=100,
            spectral_radius=0.95,
            leak_rate=0.9,
            input_scaling=0.1,
            regularization=1e-6,
            random_seed=42,
            init_value=data["CLOSE"].values[0],
        )
        X_full = []
        Y_full = []
        for i in range(1, len(series)):
            X_full.append([series[i - 1]])  # input is the previous time step's value
            Y_full.append([series[i]])  # target is the current time step's value

        X_full = np.array(X_full)
        Y_full = np.array(Y_full)

        m.fit(X_full, Y_full, washout=30)  # e.g., 30-step washout
        #m.save_weights("reservoir_weights.npz")
        pred = m.predict(
            np.array([[series[-1]]]), forecast_steps=forecast, generative=True
        )
        logged_array = np.concatenate((series, pred), axis=None)
        return m.back_to_real_and_coef(logged_array, len(data))[1]
    
if __name__ == "__main__":
    datadir='../data/dom_index.csv'
    path_to_weights = '../models/'   
    k=forecast_index(datadir, 105, path_to_weights=None)
    print(k)
    
    