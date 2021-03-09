import numpy as np

import influence.influence_model
import influence.influence_with_s_test
import representer.representer_model

class UnifiedModel(object):
    """Wrapper class which represents a TensorFlow model using one or more other model classes, to simplify interaction."""

    def __init__(
        self,
        use_influence_functions=True,
        use_relatif=True,
        use_representer_points=True,
        **kwargs,
    ):

        self.use_influence_functions = use_influence_functions
        self.use_relatif = use_relatif
        self.use_representer_points = use_representer_points
        
        if self.use_relatif and not self.use_influence_functions:
            raise ValueError("Cannot use RelatIF without using influence functions.")
        
        # Mandatory arguments for all models.
        training_inputs = kwargs.pop("training_inputs")
        training_labels = kwargs.pop("training_labels")
        test_inputs = kwargs.pop("test_inputs")
        loss_fn = kwargs.pop("loss_fn")

        if use_influence_functions:
            # Mandatory arguments for influence model.
            model = kwargs.pop("model")
            test_labels = kwargs.pop("test_labels")

            # Optional arguments for influence model.
            if "parameters" in kwargs:
                parameters = kwargs.pop("parameters")
            else: 
                parameters = model.trainable_variables
            if "scaling" in kwargs:
                scaling = kwargs.pop("scaling")
            else:
                scaling = 1.0
            if "damping" in kwargs:
                damping = kwargs.pop("damping")
            else:
                damping = 0.0
            if "verbose" in kwargs:
                verbose = kwargs.pop("verbose")
            else:
                verbose = False
            if "dtype" in kwargs:
                dtype = kwargs.pop("dtype")
            else:
                dtype = np.float32
            if "method" in kwargs:
                method = kwargs.pop("method")
            else:
                method = "cg"
            if "cg_tol" in kwargs:
                cg_tol = kwargs.pop("cg_tol")
            else:
                cg_tol = 1e-05            
            if "lissa_samples" in kwargs:
                lissa_samples = kwargs.pop("lissa_samples")
            else:
                lissa_samples = 1
            if "lissa_depth" in kwargs:
                lissa_depth = kwargs.pop("lissa_depth")
            else:
                lissa_depth = 1000

            if "use_s_test" in kwargs:
                use_s_test = kwargs.pop("use_s_test")
            else:
                use_s_test = False
            
            # Check we aren't using s_test with RelatIF.
            if use_s_test and self.use_relatif:
                raise ValueError("Cannot use s_test trick with RelatIF.")

            # Instantiate appropriate influence model.
            if use_s_test:
                self.influence_model = influence.influence_with_s_test.InfluenceWithSTest(
                    model,
                    training_inputs,
                    training_labels,
                    test_inputs,
                    test_labels,
                    loss_fn,
                    parameters,
                    scaling,
                    damping,
                    verbose,
                    dtype,
                    method,
                    cg_tol,
                    lissa_samples,
                    lissa_depth
                )
            else:
                self.influence_model = influence.influence_model.InfluenceModel(
                    model,
                    training_inputs,
                    training_labels,
                    test_inputs,
                    test_labels,
                    loss_fn,
                    parameters,
                    scaling,
                    damping,
                    verbose,
                    dtype,
                    method,
                    cg_tol,
                    lissa_samples,
                    lissa_depth
                )

        if self.use_representer_points:
            # Mandatory arguments for representer model.
            feature_model = kwargs.pop("feature_model")
            prediction_network = kwargs.pop("prediction_network")

            # Optional arguments for representer model.
            if "l2" in kwargs:
                l2 = kwargs.pop("l2")
            else:
                l2 = 0.01
            if "num_training_points" in kwargs:
                num_training_points = kwargs.pop("num_training_points")
            else:
                num_training_points = len(training_inputs)

            # Instantiate appropriate representer model.
            self.representer_model = representer.representer_model.RepresenterModel(
                feature_model,
                prediction_network,
                training_inputs,
                training_labels,
                test_inputs,
                loss_fn,
                l2,
                num_training_points,
            )

        if kwargs:
            print("Unexpected keyword arguments provided: {}".format(list(kwargs)))
    
    def get_influence_value(self, training_idx, test_idx):

        if not self.use_influence_functions:
            raise RuntimeError("Cannot use influence functions when use_influence_functions is set to False.")

        return self.influence_model.get_influence_on_loss(training_idx, test_idx)

    def get_theta_relatif_value(self, training_idx, test_idx):

        if not self.use_relatif:
            raise RuntimeError("Cannot use RelatIF when use_relatif is set to False.")

        return self.influence_model.get_theta_relatif(training_idx, test_idx)

    def get_l_relatif_value(self, training_idx, test_idx):

        if not self.use_relatif:
            raise RuntimeError("Cannot use RelatIF when use_relatif is set to False.")

        return self.influence_model.get_l_relatif(training_idx, test_idx)

    def get_representer_value(self, training_idx, test_idx):

        if not self.use_representer_points:
            raise RuntimeError("Cannot use Representer points when use_representer_points is set to False.")

        return self.representer_model.get_representer_value(training_idx, test_idx)