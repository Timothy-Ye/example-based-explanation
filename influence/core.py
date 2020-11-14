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
        unconnected_gradient=tf.UnconnectedGradients.ZERO,
    )

    return hvp


# Calculate the gradient of loss over a training point w.r.t. model parameters.
def get_training_gradient(model, training_sample, training_label, loss_fn, scaling):

    with tf.GradientTape() as tape:
        model_label = model(training_sample)
        loss = loss_fn(training_label, model_label) * scaling

    training_grad = tape.gradient(
        loss,
        model.trainable_variables,
        unconnected_gradients=tf.UnconnectedGradients.ZERO,
    )

    return training_grad


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

    return cg_loss_fn

