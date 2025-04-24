import tensorflow as tf

def route_args(router, args, depth):
    """
    Routes keyword arguments into f and g arguments as specified in router.
    """
    routed_args = [(dict(), dict()) for _ in range(depth)]
    matched_keys = [key for key in args.keys() if key in router]

    for key in matched_keys:
        val = args[key]
        for d, ((f_args, g_args), routes) in enumerate(
            zip(routed_args, router[key])
        ):
            # For each route, if flag is True, add value for that key.
            new_f_args, new_g_args = tuple(
                ({key: val} if route else {}) for route in routes
            )
            f_args.update(new_f_args)
            g_args.update(new_g_args)
            routed_args[d] = (f_args, g_args)
    return routed_args


class Deterministic(tf.keras.layers.Layer):
    """
    Allows recording/restoring RNG state.
    """
    def __init__(self, net, **kwargs):
        super().__init__(**kwargs)
        self.net = net
        self.rng_state = None
    
    def record_rng(self):
        self.rng_state = tf.identity(tf.random.get_global_generator().state)

    def call(self, *args, record_rng=False, set_rng=False, **kwargs):
        """
        Set/record RNG
        """
        if record_rng: self.record_rng()
        if not set_rng: return self.net(*args, **kwargs)
        else:
            gen = tf.random.get_global_generator()
            old_state = tf.identity(gen.state)
            if self.rng_state:
                gen.state.assign(self.rng_state)
            out = self.net(*args, **kwargs)
            gen.state.assign(old_state)
            return out


@tf.custom_gradient
def reversible_block_fn(x, f, g, f_args, g_args):
    """
    Implements the reversible block forward computation:

        Given input x, split into x1, x2.
        Compute:
            y1 = x1 + f(x2, **f_args)
            y2 = x2 + g(y1, **g_args)
        Return y = concat(y1, y2).

    The custom gradient recomputes the forward pass to avoid storing activations.
    This is a simplified version compared to the manual backward_pass in PyTorch.
    """
    # Assume x.shape = [batch, n, channels] and channel dimension is last.
    x1, x2 = tf.split(x, num_or_size_splits=2, axis=-1)
    # Forward pass
    y1 = x1 + f(x2, **f_args)
    y2 = x2 + g(y1, **g_args)
    y = tf.concat([y1, y2], axis=-1)

    def grad(dy):
        """
        In the backward pass we re-compute intermediate results.
        This implementation uses a GradientTape on the whole forward pass.
        """
        # Recompute forward pass from input x.
        with tf.GradientTape() as tape:
            tape.watch(x)
            x1_r, x2_r = tf.split(x, num_or_size_splits=2, axis=-1)
            y1_r = x1_r + f(x2_r, **f_args)
            y2_r = x2_r + g(y1_r, **g_args)
            y_r = tf.concat([y1_r, y2_r], axis=-1)
        # Compute the gradients w.r.t. x using the upstream gradient dy.
        dx = tape.gradient(y_r, x, output_gradients=dy)
        # The other arguments (f, g, f_args, g_args) are not trainable here.
        return dx, None, None, None, None

    return y, grad

class ReversibleBlock(tf.keras.layers.Layer):
    """
    A reversible block which uses two sub-networks f and g.
    It wraps f and g with a Deterministic layer
    and uses a custom gradient function to
    re-compute intermediates on the backward pass
    """
    def __init__(self, f, g, **kwargs):
        super().__init__(**kwargs)
        self.f = Deterministic(f)
        self.g = Deterministic(g)

    def call(self, x, f_args={}, g_args={}):
        # Use the custom reversible function
        return reversible_block_fn(x, self.f, self.g, f_args, g_args)


# Sequential (non-reversible) sequence layer
class SequentialSequence(tf.keras.layers.Layer):
    """
    A sequential block: applies pairs of functions f and g with residuals
      x = x + f(x, **f_args)
      x = x + g(x, **g_args)
    """
    def __init__(self, layers, args_route={}, layer_dropout=0.0, **kwargs):
        """
        layers: list of (f, g) tuples (each f, g is a callable/layer)
        args_route: dictionary mapping argument keys to a list of booleans for each layer.
        """
        super().__init__(**kwargs)
        self.layers = layers
        self.args_route = args_route
        self.layer_dropout = layer_dropout

    def call(self, x, **kwargs):
        args = route_args(self.args_route, kwargs, len(self.layers))
        for (f, g), (f_args, g_args) in zip(self.layers, args):
            x = x + f(x, **f_args)
            x = x + g(x, **g_args)
        return x

# Reversible Sequence layer (stacking reversible blocks)
class ReversibleSequence(tf.keras.layers.Layer):
    """
    Stacks reversible blocks and then averages the two output halves
    Mimics the PyTorch implementation:
        - Duplicate the input channels.
        - For each reversible block, update the concatenated representation.
        - Finally, split and average the two halves for the final output.
    """
    def __init__(self, blocks, args_route={}, **kwargs):
        """
        blocks: list of (f, g) tuples.
        args_route: similar to SequentialSequence.
        """
        super().__init__(**kwargs)
        self.args_route = args_route
        # List of reversible blocks
        self.blocks = [ReversibleBlock(f=f, g=g) for f, g in blocks]

    def call(self, x, **kwargs):
        # Duplicate x along the channel dimension.
        x = tf.concat([x, x], axis=-1)
        args = route_args(self.args_route, kwargs, len(self.blocks))
        # Reformat args for each reversible block.
        args = [{"f_args": a[0], "g_args": a[1]} for a in args]
        for block, arg in zip(self.blocks, args):
            x = block(x, **arg)
        # Split the channels and average the two halves.
        x1, x2 = tf.split(x, num_or_size_splits=2, axis=-1)
        return (x1 + x2) / 2