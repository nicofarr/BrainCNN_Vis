
# coding: utf-8
# In this file, functions for new types of visualization using Keras-vis

import numpy as np

import os 

from keras.models import load_model
import h5py
from scipy.stats import pearsonr
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import Dense, Dropout, Flatten
from keras.layers.advanced_activations import LeakyReLU
from keras import optimizers, callbacks, regularizers, initializers
from E2E_conv import *



# In[39]:

### Here we define an input modifier that only forces the result to be a symmetric matrix

from vis import input_modifiers as inpu

def symet(X):
    return 0.5*(X + X.T)

def iden(W):
    return W

binarizer = inpu.InputModifier()

binarizer.post = symet
binarizer.pre = iden


### Here we define the Loss function to estimate ternary activations in a layer 

from vis.losses import Loss

class EstimateTernaryInput(Loss):
    """A loss function that estimates Ternary activations (+1,0,-1) of a set of filters within a particular layer.
    
    One might also use this to generate an input image that is easily interpretable wrt to outputs on the final
    `keras.layers.Dense` layer.
    """
    def __init__(self, img_input):
        """
        Args:
            layer: The keras layer whose filters need to be maximized. This can either be a convolutional layer
                or a dense layer.
            
        """
        #super(ActivationMaximization, self).__init__()
        self.name = "Ternary Activation"
        self.img = img_input
        #self.filter_indices = utils.listify(filter_indices)

    def build_loss(self):
        img = K.cast(self.img,'float32')
        
        img = K.reshape(img,(64,64))

        loss = 0.
        
        loss += K.sum(K.pow(img-1,2) * K.pow(img+1,2) * K.pow(img,2))

        return loss


# In[41]:


from vis.regularizers import LPNorm,TotalVariation
from vis.losses import ActivationMaximization

from vis.utils import utils

from vis.visualization import visualize_activation_with_losses


class DynamicOptimizerCallback(object):
    """Abstract class for defining callbacks for use with [Optimizer.minimize](vis.optimizer#optimizerminimize).
    """
        
    #def _updatealpha(self,opt):
    #    #### TO IMPLEMENT 
    ##    if i < 100 :
    #        alphanew = self.alphastart
    #    else:
    ##        alphanew = self.alphastart + (i-100) * (self.alphaend - self.alphastart) / 100
    #    return alphanew
    
    def callback(self, i, opt, named_losses, overall_loss, grads, wrt_value):
        """This function will be called within [optimizer.minimize](vis.optimizer.md#minimize).
        Args:
            i: The optimizer iteration.
            named_losses: List of `(loss_name, loss_value)` tuples.
            overall_loss: Overall weighted loss.
            grads: The gradient of input image with respect to `wrt_value`.
            wrt_value: The current `wrt_value`.
        """
        #print("previous alpha : %f" % K.eval(opt.alpha))
        K.set_value(opt.alpha,K.eval(opt.alpha) * 1.03)
        #print("updated alpha : %f" % K.eval(opt.alpha))
        #raise NotImplementedError()

    def on_end(self):
        """Called at the end of optimization process. This function is typically used to cleanup / close any
        opened resources at the end of optimization.
        """
        pass


from vis.callbacks import OptimizerCallback,pprint

class Print_dyn(OptimizerCallback):
    """Callback to print values during optimization.
    """
    def callback(self, i, opt,named_losses, overall_loss, grads, wrt_value):
        print('Iteration: {}, named_losses: {}, overall loss: {}'
              .format(i + 1, pprint.pformat(named_losses), overall_loss))
        
        cur_alpha = K.eval(opt.alpha)
        
        #print('Iteration: {}, named_losses rel alpha: {}, overall loss: {}'
        #      .format(i + 1, pprint.pformat(named_losses/ cur_alpha), overall_loss))


# In[89]:


from vis.callbacks import Print
from vis.grad_modifiers import get

_PRINT_CALLBACK = Print_dyn()

def _identity(x):
    return x



class OptimizerDynamic(object):

    def __init__(self, input_tensor, losses, input_range=(0, 255) ,alpha=1e-6,wrt_tensor=None, norm_grads=True):
        """Creates an optimizer that minimizes weighted loss function.
        Args:
            input_tensor: An input tensor of shape: `(samples, channels, image_dims...)` if `image_data_format=
                channels_first` or `(samples, image_dims..., channels)` if `image_data_format=channels_last`.
            losses: List of ([Loss](vis.losses#Loss), weight) tuples.
            input_range: Specifies the input range as a `(min, max)` tuple. This is used to rescale the
                final optimized input to the given range. (Default value=(0, 255))
            wrt_tensor: Short for, with respect to. This instructs the optimizer that the aggregate loss from `losses`
                should be minimized with respect to `wrt_tensor`.
                `wrt_tensor` can be any tensor that is part of the model graph. Default value is set to None
                which means that loss will simply be minimized with respect to `input_tensor`.
            norm_grads: True to normalize gradients. Normalization avoids very small or large gradients and ensures
                a smooth gradient gradient descent process. If you want the actual gradient
                (for example, visualizing attention), set this to false.
        """
        self.input_tensor = input_tensor
        self.input_range = input_range
        self.loss_names = []
        self.loss_functions = []
        self.wrt_tensor = self.input_tensor if wrt_tensor is None else wrt_tensor

        self.alpha = K.variable(alpha)
        overall_loss = None
        for curi, (loss, weight) in enumerate(losses):
            # Perf optimization. Don't build loss function with 0 weight.
            if weight != 0:
                ### test on curi 
                if curi==3:
                    loss_fn = self.alpha * loss.build_loss()
                    #print("initial value : " K.eval(self.alpha))
                else:
                    loss_fn = weight * loss.build_loss()
                overall_loss = loss_fn if overall_loss is None else overall_loss + loss_fn
                self.loss_names.append(loss.name)
                self.loss_functions.append(loss_fn)

        # Compute gradient of overall with respect to `wrt` tensor.
        grads = K.gradients(overall_loss, self.wrt_tensor)[0]
        if norm_grads:
            grads = grads / (K.sqrt(K.mean(K.square(grads))) + K.epsilon())

        # The main function to compute various quantities in optimization loop.
        self.compute_fn = K.function([self.input_tensor, K.learning_phase()],
                                     self.loss_functions + [overall_loss, grads, self.wrt_tensor])

    def _rmsprop(self, grads, cache=None, decay_rate=0.95):
        """Uses RMSProp to compute step from gradients.
        Args:
            grads: numpy array of gradients.
            cache: numpy array of same shape as `grads` as RMSProp cache
            decay_rate: How fast to decay cache
        Returns:
            A tuple of
                step: numpy array of the same shape as `grads` giving the step.
                    Note that this does not yet take the learning rate into account.
                cache: Updated RMSProp cache.
        """
        if cache is None:
            cache = np.zeros_like(grads)
        cache = decay_rate * cache + (1 - decay_rate) * grads ** 2
        step = -grads / np.sqrt(cache + K.epsilon())
        return step, cache

    def _get_seed_input(self, seed_input):
        """Creates a random `seed_input` if None. Otherwise:
            - Ensures batch_size dim on provided `seed_input`.
            - Shuffle axis according to expected `image_data_format`.
        """
        desired_shape = (1, ) + K.int_shape(self.input_tensor)[1:]
        if seed_input is None:
            return utils.random_array(desired_shape, mean=np.mean(self.input_range),
                                      std=0.05 * (self.input_range[1] - self.input_range[0]))

        # Add batch dim if needed.
        if len(seed_input.shape) != len(desired_shape):
            seed_input = np.expand_dims(seed_input, 0)

        # Only possible if channel idx is out of place.
        if seed_input.shape != desired_shape:
            seed_input = np.moveaxis(seed_input, -1, 1)
        return seed_input.astype(K.floatx())

    def minimize(self, seed_input=None, max_iter=200,
                 input_modifiers=None, grad_modifier=None,
                 callbacks=None, verbose=True):
        """Performs gradient descent on the input image with respect to defined losses.
        Args:
            seed_input: An N-dim numpy array of shape: `(samples, channels, image_dims...)` if `image_data_format=
                channels_first` or `(samples, image_dims..., channels)` if `image_data_format=channels_last`.
                Seeded with random noise if set to None. (Default value = None)
            max_iter: The maximum number of gradient descent iterations. (Default value = 200)
            input_modifiers: A list of [InputModifier](vis.input_modifiers#inputmodifier) instances specifying
                how to make `pre` and `post` changes to the optimized input during the optimization process.
                `pre` is applied in list order while `post` is applied in reverse order. For example,
                `input_modifiers = [f, g]` means that `pre_input = g(f(inp))` and `post_input = f(g(inp))`
            grad_modifier: gradient modifier to use. See [grad_modifiers](vis.grad_modifiers.md). If you don't
                specify anything, gradients are unchanged. (Default value = None)
            callbacks: A list of [OptimizerCallback](vis.callbacks#optimizercallback) instances to trigger.
            verbose: Logs individual losses at the end of every gradient descent iteration.
                Very useful to estimate loss weight factor(s). (Default value = True)
        Returns:
            The tuple of `(optimized input, grads with respect to wrt, wrt_value)` after gradient descent iterations.
        """
        seed_input = self._get_seed_input(seed_input)
        input_modifiers = input_modifiers or []
        grad_modifier = _identity if grad_modifier is None else get(grad_modifier)

        callbacks = callbacks or []
        if verbose:
            callbacks.append(_PRINT_CALLBACK)

        cache = None
        best_loss = float('inf')
        best_input = None

        grads = None
        wrt_value = None

        all_losses = []
        for i in range(max_iter):
            # Apply modifiers `pre` step
            for modifier in input_modifiers:
                seed_input = modifier.pre(seed_input)

            # 0 learning phase for 'test'
            computed_values = self.compute_fn([seed_input, 0])
            losses = computed_values[:len(self.loss_names)]
            named_losses = list(zip(self.loss_names, losses))
            overall_loss, grads, wrt_value = computed_values[len(self.loss_names):]

            # TODO: theano grads shape is inconsistent for some reason. Patch for now and investigate later.
            if grads.shape != wrt_value.shape:
                grads = np.reshape(grads, wrt_value.shape)

            # Apply grad modifier.
            grads = grad_modifier(grads)

            # Trigger callbacks
            for c in (callbacks):
                c.callback(i, self, named_losses, overall_loss, grads, wrt_value)

            # Gradient descent update.
            # It only makes sense to do this if wrt_tensor is input_tensor. Otherwise shapes wont match for the update.
            if self.wrt_tensor is self.input_tensor:
                step, cache = self._rmsprop(grads, cache)
                seed_input += step

            # Apply modifiers `post` step
            for modifier in reversed(input_modifiers):
                seed_input = modifier.post(seed_input)

            all_losses.append(named_losses)
            if overall_loss < best_loss:
                best_loss = overall_loss.copy()
                best_input = seed_input.copy()

        # Trigger on_end
        for c in callbacks:
            c.on_end()

        img = best_input[0]
        #img = utils.deprocess_input(best_input[0], self.input_range)

        return img, grads, wrt_value,all_losses,named_losses,overall_loss


def visualize_activation_with_losses_dynamic(input_tensor, losses, wrt_tensor=None,alpha=1e-6,
                                     seed_input=None, input_range=(0, 255),
                                     **optimizer_params):
    """Generates the `input_tensor` that minimizes the weighted `losses`. This function is intended for advanced
    use cases where a custom loss is desired.
    Args:
        input_tensor: An input tensor of shape: `(samples, channels, image_dims...)` if `image_data_format=
            channels_first` or `(samples, image_dims..., channels)` if `image_data_format=channels_last`.
        wrt_tensor: Short for, with respect to. The gradients of losses are computed with respect to this tensor.
            When None, this is assumed to be the same as `input_tensor` (Default value: None)
        losses: List of ([Loss](vis.losses#Loss), weight) tuples.
        seed_input: Seeds the optimization with a starting image. Initialized with a random value when set to None.
            (Default value = None)
        input_range: Specifies the input range as a `(min, max)` tuple. This is used to rescale the
            final optimized input to the given range. (Default value=(0, 255))
        optimizer_params: The **kwargs for optimizer [params](vis.optimizer#optimizerminimize). Will default to
            reasonable values when required keys are not found.
    Returns:
        The model input that minimizes the weighted `losses`.
    """
    
    ### Configure the Dynamic Callback 
    dyn_cb = DynamicOptimizerCallback()
    
    # Default optimizer kwargs.
    optimizer_params = utils.add_defaults_to_kwargs({
        'seed_input': seed_input,
        'max_iter': 200,
        'callbacks' : [dyn_cb,],
        'verbose': False
    }, **optimizer_params)

    opt = OptimizerDynamic(input_tensor, losses, input_range, alpha,wrt_tensor=wrt_tensor)
    img,grads,_,all_losses,named_losses,overall_loss = opt.minimize(**optimizer_params)

    # If range has integer numbers, cast to 'uint8'
    #if isinstance(input_range[0], int) and isinstance(input_range[1], int):
    #    img = np.clip(img, input_range[0], input_range[1]).astype('uint8')

    if K.image_data_format() == 'channels_first':
        img = np.moveaxis(img, 0, -1)
    return img,all_losses,named_losses,overall_loss


# In[91]:


from vis.regularizers import LPNorm,TotalVariation
from vis.losses import ActivationMaximization
from vis.utils import utils

from vis.visualization import visualize_activation_with_losses

def visualize_activation_ternary_dynamic(model, layer_idx,alpha=1e-6,filter_indices=None, wrt_tensor=None,
                         seed_input=None, input_range=(-1, 1),
                         backprop_modifier=None, grad_modifier=None,
                         act_max_weight=1, lp_norm_weight=10, tv_weight=10,
                         **optimizer_params):
    """Generates the model input that maximizes the output of all `filter_indices` in the given `layer_idx`, and
    put it in ternary representation
    Args:
        model: The `keras.models.Model` instance. The model input shape must be: `(samples, channels, image_dims...)`
            if `image_data_format=channels_first` or `(samples, image_dims..., channels)` if
            `image_data_format=channels_last`.
        layer_idx: The layer index within `model.layers` whose filters needs to be visualized.
        filter_indices: filter indices within the layer to be maximized.
            If None, all filters are visualized. (Default value = None)
            For `keras.layers.Dense` layer, `filter_idx` is interpreted as the output index.
            If you are visualizing final `keras.layers.Dense` layer, consider switching 'softmax' activation for
            'linear' using [utils.apply_modifications](vis.utils.utils#apply_modifications) for better results.
        wrt_tensor: Short for, with respect to. The gradients of losses are computed with respect to this tensor.
            When None, this is assumed to be the same as `input_tensor` (Default value: None)
        seed_input: Seeds the optimization with a starting input. Initialized with a random value when set to None.
            (Default value = None)
        input_range: Specifies the input range as a `(min, max)` tuple. This is used to rescale the
            final optimized input to the given range. (Default value=(0, 255))
        backprop_modifier: backprop modifier to use. See [backprop_modifiers](vis.backprop_modifiers.md). If you don't
            specify anything, no backprop modification is applied. (Default value = None)
        grad_modifier: gradient modifier to use. See [grad_modifiers](vis.grad_modifiers.md). If you don't
            specify anything, gradients are unchanged (Default value = None)
        act_max_weight: The weight param for `ActivationMaximization` loss. Not used if 0 or None. (Default value = 1)
        lp_norm_weight: The weight param for `LPNorm` regularization loss. Not used if 0 or None. (Default value = 10)
        tv_weight: The weight param for `TotalVariation` regularization loss. Not used if 0 or None. (Default value = 10)
        alpha : regularization parameter for the ternarization
        optimizer_params: The **kwargs for optimizer [params](vis.optimizer#optimizerminimize). Will default to
            reasonable values when required keys are not found.
    Example:
        If you wanted to visualize the input image that would maximize the output index 22, say on
        final `keras.layers.Dense` layer, then, `filter_indices = [22]`, `layer_idx = dense_layer_idx`.
        If `filter_indices = [22, 23]`, then it should generate an input image that shows features of both classes.
    Returns:
        The model input that maximizes the output of `filter_indices` in the given `layer_idx`.
    """
    if backprop_modifier is not None:
        modifier_fn = get(backprop_modifier)
        model = modifier_fn(model)

    losses = [
        (ActivationMaximization(model.layers[layer_idx], filter_indices), act_max_weight),
        (LPNorm(model.input,1), lp_norm_weight),
        (TotalVariation(model.input), tv_weight),
        (EstimateTernaryInput(model.input), alpha)
    ]

    # Add grad_filter to optimizer_params.
    optimizer_params = utils.add_defaults_to_kwargs({
        'grad_modifier': grad_modifier,
        'input_modifiers' : [binarizer,],
    }, **optimizer_params)

    return visualize_activation_with_losses_dynamic(model.input, losses, wrt_tensor,alpha,
                                            seed_input, input_range, **optimizer_params)
