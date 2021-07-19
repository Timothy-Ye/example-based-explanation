import numpy as np
import scipy.optimize
import tensorflow as tf


class ExplainableModel(object):
    """
    Core wrapper class around a TensorFlow 2 model,
    for producing example-based explanations of the model's predictions.
    """

    def __init__(
        self,
        model=None,
        feature_model=None,
        model_parameters=None,
        train_x=None,
        train_y=None,
        num_train_points=None,
        test_x=None,
        test_y=None,
        num_test_points=None,
        loss_fn=None,
        l2=0.0,
        dtype=np.float64,
        ihvp_method="cg",
        cg_damping=0.0,
        cg_tol=1e-05,
        cg_maxiter=100,
        lissa_scaling=1.0,
        lissa_samples=1,
        lissa_depth=1000,
        verbose=False,
    ):
        self.model = model
        self.feature_model = feature_model

        if model is not None and model_parameters is None:
            self.model_parameters = model.trainable_variables
        else:
            self.model_parameters = model_parameters

        self.train_x = train_x
        self.train_y = train_y
        self.num_train_points = num_train_points

        # Check lengths of train_x and train_y match.
        if train_x is not None and train_y is not None:
            if len(train_x) != len(train_y):
                raise ValueError(
                    "train_x and train_y have different lengths: train_x has length {}, train_y has length {}".format(
                        len(train_x), len(train_y)
                    )
                )

        # Infer num_train_points if appropriate.
        if num_train_points is None and train_x is not None:
            self.num_train_points = len(train_x)

        self.test_x = test_x
        self.test_y = test_y
        self.num_test_points = num_test_points

        # Check lengths of test_x and test_y match.
        if test_x is not None and test_y is not None:
            if len(test_x) != len(test_y):
                raise ValueError(
                    "test_x and test_y have different lengths: test_x has length {}, test_y has length {}".format(
                        len(test_x), len(test_y)
                    )
                )

        # Infer num_test_points if appropriate.
        if num_test_points is None and test_x is not None:
            self.num_test_points = len(test_x)

        self.loss_fn = loss_fn
        self.l2 = l2
        self.dtype = dtype
        self.ihvp_method = ihvp_method
        self.cg_damping = cg_damping
        self.cg_tol = cg_tol
        self.cg_maxiter = cg_maxiter
        self.lissa_scaling = lissa_scaling
        self.lissa_samples = lissa_samples
        self.lissa_depth = lissa_depth
        self.verbose = verbose

        self.train_grads = {}
        self.test_grads = {}

        self.train_ihvps = {}
        self.test_ihvps = {}

        self.train_feats = {}
        self.test_feats = {}

        self.alpha_vals = {}

    # TODO: Clean up usage of dtypes.

    def flatten_tensors(self, tensors):
        """
        Converts a list of TensorFlow tensors into a 1D NumPy array.
        """

        return np.concatenate([tf.reshape(t, -1) for t in tensors])

    def reshape_vector(self, vector):
        """
        Converts a 1D NumPy array (i.e. vector) in a list of TensorFlow tensors with the same shape as model_parameters
        """

        # idxs used for splitting the flat vector, so idx 0 is prepended.
        # dtype is explicit in case model_parameters are empty.
        idxs = np.append(
            [0], np.cumsum([tf.size(t) for t in self.model_parameters], dtype=np.int32)
        )

        # Check dimensions of x match model parameters.
        total_size = idxs[-1]
        if len(vector) != total_size:
            raise ValueError(
                "vector and model_parameters have different lengths: vector has length {}, model_parameters has length {}".format(
                    len(vector), total_size
                )
            )

        # Reshape x into list of tensors.
        return [
            tf.reshape(
                vector[idxs[i] : idxs[i + 1]], tf.shape(self.model_parameters[i])
            )
            for i in range(len(self.model_parameters))
        ]

    def get_train_grad(self, train_idx):
        """
        Gets gradient of the loss w.r.t. model parameters at a training point.
        """

        if train_idx in self.train_grads:
            return self.train_grads[train_idx]

        # TODO: Check necessary parameters are specified (e.g. train_x).

        with tf.GradientTape() as tape:
            pred = self.model(self.train_x[np.newaxis, train_idx])
            loss = self.loss_fn(self.train_y[np.newaxis, train_idx], pred)

        train_grad = tape.gradient(
            loss,
            self.model_parameters,
            unconnected_gradients=tf.UnconnectedGradients.ZERO,
        )

        # L2 regularisation can be added separately from loss_fn.
        self.train_grads[train_idx] = self.flatten_tensors(
            train_grad
        ) + 2 * self.l2 * self.flatten_tensors(self.model_parameters)

        return self.train_grads[train_idx]

    def get_test_grad(self, test_idx):
        """
        Gets gradient of the loss w.r.t. model parameters at a test point.
        """

        if test_idx in self.test_grads:
            return self.test_grads[test_idx]

        with tf.GradientTape() as tape:
            pred = self.model(self.test_x[np.newaxis, test_idx])
            loss = self.loss_fn(self.test_y[np.newaxis, test_idx], pred)

        test_grad = tape.gradient(
            loss,
            self.model_parameters,
            unconnected_gradients=tf.UnconnectedGradients.ZERO,
        )

        # L2 regularisation can be added separately from loss_fn.
        self.test_grads[test_idx] = self.flatten_tensors(
            test_grad
        ) + 2 * self.l2 * self.flatten_tensors(self.model_parameters)

        return self.test_grads[test_idx]

    def get_ihvp_cg(self, vector):
        """
        Gets the inverse Hessian-vector product using conjugate gradient method.
        """

        # Helper function which calculates Hessian-vector product from vector x.
        def get_hvp(x):

            # Calculate HVP using back-over-back auto-diff.
            with tf.GradientTape() as outer_tape:
                with tf.GradientTape() as inner_tape:
                    pred = self.model(self.train_x)
                    loss = self.loss_fn(self.train_y, pred)

                grads = inner_tape.gradient(
                    loss,
                    self.model_parameters,
                    unconnected_gradients=tf.UnconnectedGradients.ZERO,
                )

            hvp = outer_tape.gradient(
                grads,
                self.model_parameters,
                output_gradients=self.reshape_vector(x),
                unconnected_gradients=tf.UnconnectedGradients.ZERO,
            )

            # L2 regularisation and damping added at the end.
            return self.flatten_tensors(hvp) + (2.0 * self.l2 + self.cg_damping) * x

        def cg_loss_fn(x):

            hvp = get_hvp(x)
            return 0.5 * np.dot(x, hvp) - np.dot(x, vector)

        def cg_jac_fn(x):

            hvp = get_hvp(x)
            return hvp - vector

        # TODO: Add verbose callback.

        result = scipy.optimize.minimize(
            cg_loss_fn,
            vector,
            method="CG",
            jac=cg_jac_fn,
            options={
                "gtol": self.cg_tol,
                "maxiter": self.cg_maxiter,
                "disp": self.verbose,
            },
        )

        return result.x

    def get_train_ihvp(self, train_idx):
        """
        Gets inverse Hessian-vector product at a training point.
        """

        if train_idx in self.train_ihvps:
            return self.train_ihvps[train_idx]

        # train_grad used in both CG and LiSSA.
        train_grad = self.get_train_grad(train_idx)

        # Using conjugate gradient method for IHVP.
        if self.ihvp_method == "cg":
            self.train_ihvps[train_idx] = self.get_ihvp_cg(train_grad)
        elif self.ihvp_method == "lissa":
            raise NotImplementedError(
                "LiSSA has not yet been implemented for calculating IHVPs."
            )
        else:
            raise ValueError(
                "'{}' is not a supported method of calculating IHVPs.".format(
                    self.ihvp_method
                )
            )

        return self.train_ihvps[train_idx]

    def get_test_ihvp(self, test_idx):
        """
        Gets inverse Hessian-vector product at a test point.
        """

        if test_idx in self.test_ihvps:
            return self.test_ihvps[test_idx]

        # test_grad used in both CG and LiSSA.
        test_grad = self.get_test_grad(test_idx)

        # Using conjugate gradient method for IHVP.
        if self.ihvp_method == "cg":
            self.test_ihvps[test_idx] = self.get_ihvp_cg(test_grad)
        elif self.ihvp_method == "lissa":
            raise NotImplementedError(
                "LiSSA has not yet been implemented for calculating IHVPs."
            )
        else:
            raise ValueError(
                "'{}' is not a supported method of calculating IHVPs.".format(
                    self.ihvp_method
                )
            )

        return self.test_ihvps[test_idx]

    def get_train_feat(self, train_idx):
        """
        Gets features (i.e. output from feature_model) at a training point.
        """

        if train_idx in self.train_feats:
            return self.train_feats[train_idx]

        self.train_feats[train_idx] = self.feature_model(
            self.train_x[np.newaxis, train_idx]
        ).reshape(-1)

        return self.train_feats[train_idx]

    def get_test_feat(self, test_idx):
        """
        Gets features (i.e. output from feature_model) at a test point.
        """

        if test_idx in self.test_feats:
            return self.test_feats[test_idx]

        self.test_feats[test_idx] = self.feature_model(
            self.test_x[np.newaxis, test_idx]
        ).reshape(-1)

        return self.test_feats[test_idx]

    def get_alpha_val(self, train_idx):
        """
        Gets the alpha value (for representer theorem) at a training point.
        """

        if train_idx in self.alpha_vals:
            return self.alpha_vals[train_idx]

        with tf.GradientTape() as tape:
            pred = self.model(self.train_x[np.newaxis, train_idx])
            loss = self.loss_fn(self.train_y[np.newaxis, train_idx], pred)

        grad = tape.gradient(
            loss,
            pred,
            unconnected_gradients=tf.UnconnectedGradients.ZERO,
        )

        # grad.numpy()[0] gives NumPy array of desired dimension.
        self.alpha_vals[train_idx] = -grad.numpy()[0] / (
            2 * self.l2 * self.num_train_points
        )

        return self.alpha_vals[train_idx]

    def get_influence(self, train_idx, test_idx, use_s_test=False):
        """
        Gets the influence of a training point on a test point.
        """

        if use_s_test:
            return np.dot(self.get_train_grad(train_idx), self.get_test_ihvp(test_idx))
        else:
            return np.dot(self.get_train_ihvp(train_idx), self.get_test_grad(test_idx))

    def get_theta_relatif(self, train_idx, test_idx):
        """
        Gets the theta-relative influence of a training point on a test point.
        """

        return np.dot(
            self.get_train_ihvp(train_idx), self.get_test_grad(test_idx)
        ) / np.linalg.norm(self.get_train_ihvp(train_idx))

    def get_l_relatif(self, train_idx, test_idx):
        """
        Gets the l-relative influence of a training point on a test point.
        """

        return np.dot(
            self.get_train_ihvp(train_idx), self.get_test_grad(test_idx)
        ) / np.sqrt(
            np.dot(self.get_train_ihvp(train_idx), self.get_train_grad(train_idx))
        )

    def get_representer_value(self, train_idx, test_idx):
        """
        Gets the representer value for a training point given a test point.
        """

        return self.get_alpha_val(train_idx) * np.dot(
            self.get_train_feat(train_idx), self.get_test_feat(test_idx)
        )

    def get_grad_cos(self, train_idx, test_idx):
        """
        Gets the gradient cosine value for a training point and a test point.
        """

        return np.dot(self.get_train_grad(train_idx), self.get_test_grad(test_idx)) / (
            np.linalg.norm(self.get_train_grad(train_idx))
            * np.linalg.norm(self.get_test_grad(test_idx))
        )

    def get_influence_parameters(self, train_idx, epsilon=None):
        """
        Gets the predicted parameters for if a training point was upweighted by epsilon,
        approximated using influence functions.
        """

        # By default, set epsilon to approximate leave-one-out retraining.
        if epsilon is None:
            epsilon = -1.0 / self.num_train_points

        return self.reshape_vector(
            - epsilon * self.get_train_ihvp(train_idx)
            + self.flatten_tensors(self.model_parameters)
        )

    def save_train_grads(self, file, compressed=True):
        """
        Saves the currently stored train_grads to a .npz file.
        """

        # Array names must be strings, not indices.
        dict = {str(k): v for k, v in self.train_grads.items()}

        if compressed:
            np.savez_compressed(file, **dict)
        else:
            np.savez(file, **dict)

    def load_train_grads(self, file):
        """
        Loads train_grads from a .npz file.
        """

        data = np.load(file)

        for f in data.files:
            self.train_grads[int(f)] = data[f]

    def save_test_grads(self, file, compressed=True):
        """
        Saves the currently stored test_grads to a .npz file.
        """

        dict = {str(k): v for k, v in self.test_grads.items()}

        if compressed:
            np.savez_compressed(file, **dict)
        else:
            np.savez(file, **dict)

    def load_test_grads(self, file):
        """
        Loads test_grads from a .npz file.
        """

        data = np.load(file)

        for f in data.files:
            self.test_grads[int(f)] = data[f]

    def save_train_ihvps(self, file, compressed=True):
        """
        Saves the currently stored train_ihvps to a .npz file.
        """

        dict = {str(k): v for k, v in self.train_ihvps.items()}

        if compressed:
            np.savez_compressed(file, **dict)
        else:
            np.savez(file, **dict)

    def load_train_ihvps(self, file):
        """
        Loads train_ihvps from a .npz file.
        """

        data = np.load(file)

        for f in data.files:
            self.train_ihvps[int(f)] = data[f]

    def save_test_ihvps(self, file, compressed=True):
        """
        Saves the currently stored test_ihvps to a .npz file.
        """

        dict = {str(k): v for k, v in self.test_ihvps.items()}

        if compressed:
            np.savez_compressed(file, **dict)
        else:
            np.savez(file, **dict)

    def load_test_ihvps(self, file):
        """
        Loads test_ihvps from a .npz file.
        """

        data = np.load(file)

        for f in data.files:
            self.test_ihvps[int(f)] = data[f]

    def save_train_feats(self, file, compressed=True):
        """
        Saves the currently stored train_feats to a .npz file.
        """

        dict = {str(k): v for k, v in self.train_feats.items()}

        if compressed:
            np.savez_compressed(file, **dict)
        else:
            np.savez(file, **dict)

    def load_train_feats(self, file):
        """
        Loads train_feats from a .npz file.
        """

        data = np.load(file)

        for f in data.files:
            self.train_feats[int(f)] = data[f]

    def save_test_feats(self, file, compressed=True):
        """
        Saves the currently stored test_feats to a .npz file.
        """

        dict = {str(k): v for k, v in self.test_feats.items()}

        if compressed:
            np.savez_compressed(file, **dict)
        else:
            np.savez(file, **dict)

    def load_test_feats(self, file):
        """
        Loads test_feats from a .npz file.
        """

        data = np.load(file)

        for f in data.files:
            self.test_feats[int(f)] = data[f]

    def save_alpha_vals(self, file, compressed=True):
        """
        Saves the currently stored alpha_vals to a .npz file.
        """

        dict = {str(k): v for k, v in self.alpha_vals.items()}

        if compressed:
            np.savez_compressed(file, **dict)
        else:
            np.savez(file, **dict)

    def load_alpha_vals(self, file):
        """
        Loads alpha_vals from a .npz file.
        """

        data = np.load(file)

        for f in data.files:
            self.alpha_vals[int(f)] = data[f]

    def clear_cached_values(self):
        """
        Clears all the saved gradients, IHVPs, etc., from the ExplainableModel instance.
        """

        self.train_grads = {}
        self.test_grads = {}

        self.train_ihvps = {}
        self.test_ihvps = {}

        self.train_feats = {}
        self.test_feats = {}

        self.alpha_vals = {}

    # TODO: Add verbose messages.
    # TODO: Implement LiSSA for inverse HVP calculation.
