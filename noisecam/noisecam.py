from typing import Union

import numpy as np
import tensorflow as tf
import keras.backend as K
from scipy.ndimage.interpolation import zoom

from . import ModelVisualization
from .utils import is_mixed_precision, normalize, zoom_factor
from .utils.model_modifiers import ExtractIntermediateLayerForGradcam as ModelModifier

class Noisecam(ModelVisualization):

    def __call__(self,
                 score,
                 seed_input,
                 penultimate_layer=None,
                 seek_penultimate_conv_layer=True,
                 gradient_modifier=None,
                 activation_modifier=lambda cam: K.relu(cam),
                 training=False,
                 expand_cam=True,
                 normalize_cam=True,
                 unconnected_gradients=tf.UnconnectedGradients.NONE) -> Union[np.ndarray, list]:
        
        # Preparing
        scores = self._get_scores_for_multiple_outputs(score)
        seed_inputs = self._get_seed_inputs_for_multiple_inputs(seed_input)

        # Processing gradcam
        model = ModelModifier(penultimate_layer, seek_penultimate_conv_layer)(self.model)

        with tf.GradientTape(watch_accessed_variables=False) as tape:
            tape.watch(seed_inputs)
            outputs = model(seed_inputs, training=training)
            outputs, penultimate_output = outputs[:-1], outputs[-1]
            score_values = self._calculate_scores(outputs, scores)
        grads = tape.gradient(score_values,
                              penultimate_output,
                              unconnected_gradients=unconnected_gradients)
        
        # When mixed precision enabled
        if is_mixed_precision(model):
            grads = tf.cast(grads, dtype=model.variable_dtype)
            penultimate_output = tf.cast(penultimate_output, dtype=model.variable_dtype)
            score_values = [tf.cast(v, dtype=model.variable_dtype) for v in score_values]

        score_values = sum(tf.math.exp(o) for o in score_values)
        score_values = tf.reshape(score_values, score_values.shape + (1, ) * (grads.ndim - 1))

        # first_deriv == g^{kc}_{ij}
        # second_deriv == {g^{kc}_{ij}}^2
        # third_deriv == {g^{kc}_{ij}}^3
        if gradient_modifier is not None:
            grads = gradient_modifier(grads)
        first_derivative = score_values * grads
        second_derivative = first_derivative * grads
        third_derivative = second_derivative * grads

        global_sum = K.sum(penultimate_output,
                           axis=tuple(np.arange(len(penultimate_output.shape))[1:-1]),
                           keepdims=True)
        
        alpha_denom = second_derivative * 2.0 + third_derivative * global_sum
        alpha_denom = alpha_denom + tf.cast((second_derivative == 0.0), second_derivative.dtype)
        alphas = second_derivative / alpha_denom

        alpha_normalization_constant = K.sum(alphas,
                                             axis=tuple(np.arange(len(alphas.shape))[1:-1]),
                                             keepdims=True)
        alpha_normalization_constant = alpha_normalization_constant + tf.cast(
            (alpha_normalization_constant == 0.0), alpha_normalization_constant.dtype)
        alphas = alphas / alpha_normalization_constant

        # weighting co-efficients (alphas)----above
        # global weight (w)----below -> deep_linearization_weights
        # spatial weights -> weights

        if activation_modifier is None:
            weights = first_derivative
        else:
            weights = activation_modifier(first_derivative)

        deep_linearization_weights = weights * alphas # global weight
        deep_linearization_weights = K.sum(
            deep_linearization_weights,
            axis=tuple(np.arange(len(deep_linearization_weights.shape))[1:-1]),
            keepdims=True)
        
        # spatial weights
        spatial_weights = K.relu(grads)
        
        # global weights - spatial weights
        # compact with tensorflow 1.2.1 (subtract function in different class)
        #noise_weights = tf.subtract(deep_linearization_weights, spatial_weights)
        # compact with tensorflow 2.13.0
        noise_weights = tf.math.subtract(deep_linearization_weights, spatial_weights)

        #cam = K.sum(noise_weights * penultimate_output, axis=-1)
        cam = np.sum(np.multiply(penultimate_output, noise_weights), axis=-1)


        if activation_modifier is not None:
            cam = activation_modifier(cam)

        if not expand_cam:
            if normalize_cam:
                cam = normalize(cam)
            return cam
        
        # Visualizing
        factors = (zoom_factor(cam.shape, X.shape) for X in seed_inputs)
        cam = [zoom(cam, factor, order=1) for factor in factors]
        if normalize_cam:
            cam = [normalize(x) for x in cam]
        if len(self.model.inputs) == 1 and not isinstance(seed_input, list):
            cam = cam[0]
        return cam
        