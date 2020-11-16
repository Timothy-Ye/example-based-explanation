import numpy as np
import tensorflow as tf
import scipy.optimize

# Takes a flat vector and a TF module, and returns a tensor with shape identical to the module's trainable variables.
def reshape_flat_vector(flat_vector, model):

    # Check the overall lengths match.
    length = np.sum([len(tf.reshape(t, [-1])) for t in model.trainable_variables])
    if len(flat_vector) != length:
        raise ValueError(
            "Length of flat vector is "
            + len(flat_vector)
            + ", while module has "
            + length
            + " trainable variables."
        )

    # Reshape flat_vector.
    reshaped_flat_vector = []
    i = 0
    for t in model.trainable_variables:
        var_length = len(tf.reshape(t, [-1]))
        reshaped_flat_vector.append(
            tf.reshape(flat_vector[i : i + var_length], tf.shape(t))
        )
        i += var_length

    return reshaped_flat_vector


# Calculate the Hessian Vector Product, where the Hessian is the loss over training data w.r.t. model parameters.
def get_hvp(vector, model, training_samples, training_labels, loss_fn, scaling):

    # Calculate HVP using back-over-back auto-diff.
    with tf.GradientTape() as outer_tape:
        with tf.GradientTape() as inner_tape:
            predicted_labels = model(training_samples)
            loss = loss_fn(training_labels, predicted_labels) * scaling

        grads = inner_tape.gradient(
            loss,
            model.trainable_variables,
            unconnected_gradients=tf.UnconnectedGradients.ZERO,
        )

    hvp = outer_tape.gradient(
        grads,
        model.trainable_variables,
        output_gradients=vector,
        unconnected_gradients=tf.UnconnectedGradients.ZERO,
    )

    return hvp


# Calculate the gradient of loss at a training point w.r.t. model parameters.
def get_training_gradient(model, training_sample, training_label, loss_fn, scaling):

    with tf.GradientTape() as tape:
        model_label = model(training_sample)
        loss = loss_fn(training_label, model_label) * scaling

    training_gradient = tape.gradient(
        loss,
        model.trainable_variables,
        unconnected_gradients=tf.UnconnectedGradients.ZERO,
    )

    return training_gradient


# Return the loss function to feed into Conjugate Gradient optimisation.
def get_cg_loss_fn(
    model,
    training_samples,
    training_labels,
    loss_fn,
    scaling,
    damping,
    training_gradient,
):
    def cg_loss_fn(x):

        # Need to reshape vector before passing into get_hvp().
        reshaped_vector = reshape_flat_vector(x, model)

        hvp = get_hvp(
            reshaped_vector, model, training_samples, training_labels, loss_fn, scaling
        )

        # Easier to flatten and just use np.dot().
        flat_hvp = np.concatenate([tf.reshape(t, [-1]) for t in hvp]) + damping * x
        flat_training_gradient = np.concatenate(
            [tf.reshape(t, [-1]) for t in training_gradient]
        )

        return 0.5 * np.dot(flat_hvp, x) - np.dot(flat_training_gradient, x)

    return cg_loss_fn


# Return the gradient function to feed into Conjugate Gradient optimisation.
def get_cg_jac_fn(
    model,
    training_samples,
    training_labels,
    loss_fn,
    scaling,
    damping,
    training_gradient,
):
    def cg_jac_fn(x):

        # Need to reshape vector before passing into get_hvp().
        reshaped_vector = reshape_flat_vector(x, model)

        hvp = get_hvp(
            reshaped_vector, model, training_samples, training_labels, loss_fn, scaling
        )

        # Easier to flatten and just use np.dot().
        flat_hvp = np.concatenate([tf.reshape(t, [-1]) for t in hvp]) + damping * x
        flat_training_gradient = np.concatenate(
            [tf.reshape(t, [-1]) for t in training_gradient]
        )

        return flat_hvp - flat_training_gradient

    return cg_jac_fn


# Return the inverse HVP of loss Hessian and training gradient using Conjugate Gradient method.
def get_inverse_hvp_cg(
    model,
    training_samples,
    training_labels,
    loss_fn,
    scaling,
    damping,
    training_gradient,
    verbose,
):

    cg_loss_fn = get_cg_loss_fn(
        model,
        training_samples,
        training_labels,
        loss_fn,
        scaling,
        damping,
        training_gradient,
    )
    cg_jac_fn = get_cg_jac_fn(
        model,
        training_samples,
        training_labels,
        loss_fn,
        scaling,
        damping,
        training_gradient,
    )

    # Use training gradient as starting point for CG.
    flat_training_gradient = np.concatenate(
        [tf.reshape(t, [-1]) for t in training_gradient]
    )

    cg_callback = None
    if verbose:

        def verbose_cg_callback(xk):
            print(
                "CG Loss: ",
                cg_loss_fn(xk),
                "; CG Jac Norm:",
                np.linalg.norm(cg_jac_fn(xk)),
            )
            return

        cg_callback = verbose_cg_callback

    result = scipy.optimize.minimize(
        cg_loss_fn,
        flat_training_gradient,
        method="CG",
        jac=cg_jac_fn,
        callback=cg_callback,
        options={"maxiter": 100, "disp": verbose},
    )

    return result.x


# Calculate the gradient of loss at a test point w.r.t. model parameters.
def get_test_gradient(model, test_sample, test_label, loss_fn):

    with tf.GradientTape() as tape:
        model_label = model(test_sample)
        loss = loss_fn(test_label, model_label)

    test_gradient = tape.gradient(
        loss,
        model.trainable_variables,
        unconnected_gradients=tf.UnconnectedGradients.ZERO,
    )

    return test_gradient


# Overall function for finding influence at training and test point pair.
def get_influence(
    model,
    training_samples,
    training_labels,
    training_sample,
    training_label,
    test_sample,
    test_label,
    loss_fn=None,
    scaling=1.0,
    damping=0.0,
    verbose=False,
):

    if loss_fn is None:
        loss_fn = model.loss

    training_gradient = get_training_gradient(
        model, training_sample, training_label, loss_fn, scaling
    )

    test_gradient = get_test_gradient(model, test_sample, test_label, loss_fn)
    flat_test_gradient = np.concatenate([tf.reshape(t, [-1]) for t in test_gradient])

    inverse_hvp = get_inverse_hvp_cg(
        model,
        training_samples,
        training_labels,
        loss_fn,
        scaling,
        damping,
        training_gradient,
        verbose,
    )

    influence = np.dot(inverse_hvp, flat_test_gradient)

    return influence
