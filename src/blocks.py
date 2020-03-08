import inspect
from .utils import *
import tensorflow as tf
from tensorflow.keras import layers



class ResidualBlock(tf.keras.layers.Layer):

    def __init__(
        self, dilation_rate, nb_filters, kernel_size,
        padding, activation='relu', dropout_rate=0,
        kernel_initializer='he_normal', use_batch_norm=False,
        use_layer_norm=False, last_block=True, **kwargs):

        self.dilation_rate = dilation_rate
        self.nb_filters = nb_filters
        self.kernel_size = kernel_size
        self.padding = padding
        self.activation = activation
        self.dropout_rate = dropout_rate
        self.use_batch_norm = use_batch_norm
        self.use_layer_norm = use_layer_norm
        self.kernel_initializer = kernel_initializer
        self.last_block = last_block
        self.layers = []
        self.layers_outputs = []
        self.shape_match_conv = None
        self.res_output_shape = None
        self.final_activation = None

        super(ResidualBlock, self).__init__(**kwargs)

    def _add_and_activate_layer(self, layer):
        self.layers.append(layer)
        self.layers[-1].build(self.res_output_shape)
        self.res_output_shape = self.layers[-1].compute_output_shape(self.res_output_shape)

    def build(self, input_shape):

        with tf.keras.backend.name_scope(self.name):  # name scope used to make sure weights get unique names
            self.layers = []
            self.res_output_shape = input_shape

            for k in range(2):
                name = 'conv1D_{}'.format(k)
                with tf.keras.backend.name_scope(name):  # name scope used to make sure weights get unique names
                    self._add_and_activate_layer(
                        tf.keras.layers.Conv1D(
                            filters=self.nb_filters, kernel_size=self.kernel_size,
                            dilation_rate=self.dilation_rate, padding=self.padding,
                            name=name, kernel_initializer=self.kernel_initializer)
                        )

                if self.use_batch_norm:
                    self._add_and_activate_layer(tf.keras.layers.BatchNormalization())
                elif self.use_layer_norm:
                    self._add_and_activate_layer(tf.keras.layers.LayerNormalization())

                self._add_and_activate_layer(tf.keras.layers.Activation('relu'))
                self._add_and_activate_layer(tf.keras.layers.SpatialDropout1D(rate=self.dropout_rate))

            if not self.last_block:
                name = 'conv1D_{}'.format(k + 1)
                with tf.keras.backend.name_scope(name):
                    self.shape_match_conv = tf.keras.layers.Conv1D(
                        filters=self.nb_filters, kernel_size=1, padding='same',
                        name=name, kernel_initializer=self.kernel_initializer
                    )
            else:
                self.shape_match_conv = tf.keras.layers.Lambda(lambda x: x, name='identity')

            self.shape_match_conv.build(input_shape)
            self.res_output_shape = self.shape_match_conv.compute_output_shape(input_shape)

            self.final_activation = tf.keras.layers.Activation(self.activation)
            self.final_activation.build(self.res_output_shape)
            for layer in self.layers:
                self.__setattr__(layer.name, layer)

            super(ResidualBlock, self).build(input_shape)

    def call(self, inputs, training=None):
        x = inputs
        self.layers_outputs = [x]
        for layer in self.layers:
            training_flag = 'training' in dict(inspect.signature(layer.call).parameters)
            x = layer(x, training=training) if training_flag else layer(x)
            self.layers_outputs.append(x)
        x2 = self.shape_match_conv(inputs)
        self.layers_outputs.append(x2)
        res_x = layers.add([x2, x])
        self.layers_outputs.append(res_x)

        res_act_x = self.final_activation(res_x)
        self.layers_outputs.append(res_act_x)
        return [res_act_x, x]

    def compute_output_shape(self, input_shape):
        return [self.res_output_shape, self.res_output_shape]



class TCNLayer(tf.keras.layers.Layer):

    def __init__(
        self, nb_filters=64, kernel_size=2, nb_stacks=1,
        dilations=(1, 2, 4, 8, 16, 32), padding='causal',
        use_skip_connections=True, dropout_rate=0.0,
        return_sequences=False, activation='linear',
        kernel_initializer='he_normal', use_batch_norm=False,
        use_layer_norm=False, **kwargs):

        self.return_sequences = return_sequences
        self.dropout_rate = dropout_rate
        self.use_skip_connections = use_skip_connections
        self.dilations = dilations
        self.nb_stacks = nb_stacks
        self.kernel_size = kernel_size
        self.nb_filters = nb_filters
        self.activation = activation
        self.padding = padding
        self.kernel_initializer = kernel_initializer
        self.use_batch_norm = use_batch_norm
        self.use_layer_norm = use_layer_norm
        self.skip_connections = []
        self.residual_blocks = []
        self.layers_outputs = []
        self.main_conv1D = None
        self.build_output_shape = None
        self.lambda_layer = None
        self.lambda_ouput_shape = None

        if padding != 'causal' and padding != 'same':
            raise ValueError("Only 'causal' or 'same' padding are compatible for this layer.")

        if not isinstance(nb_filters, int):
            raise Exception()

        # initialize parent class
        super(TCNLayer, self).__init__(**kwargs)

    @property
    def receptive_field(self):
        assert_msg = 'The receptive field formula works only with power of two dilations.'
        assert all([is_power_of_two(i) for i in self.dilations]), assert_msg
        return self.kernel_size * self.nb_stacks * self.dilations[-1]

    def build(self, input_shape):
        self.main_conv1D = tf.keras.layers.Conv1D(
            filters=self.nb_filters, kernel_size=1,
            padding=self.padding, kernel_initializer=self.kernel_initializer
        )
        self.main_conv1D.build(input_shape)

        # member to hold current output shape of the layer for building purposes
        self.build_output_shape = self.main_conv1D.compute_output_shape(input_shape)

        # list to hold all the member ResidualBlocks
        self.residual_blocks = []
        total_num_blocks = self.nb_stacks * len(self.dilations)
        if not self.use_skip_connections:
            total_num_blocks += 1  # cheap way to do a false case for below

        for s in range(self.nb_stacks):
            for d in self.dilations:
                self.residual_blocks.append(
                    ResidualBlock(
                        dilation_rate=d, nb_filters=self.nb_filters,
                        kernel_size=self.kernel_size, padding=self.padding,
                        activation=self.activation, dropout_rate=self.dropout_rate,
                        use_batch_norm=self.use_batch_norm, use_layer_norm=self.use_layer_norm,
                        kernel_initializer=self.kernel_initializer,
                        last_block=len(self.residual_blocks) + 1 == total_num_blocks,
                        name='residual_block_{}'.format(len(self.residual_blocks))
                    )
                )
                # build newest residual block
                self.residual_blocks[-1].build(self.build_output_shape)
                self.build_output_shape = self.residual_blocks[-1].res_output_shape

        # this is done to force keras to add the layers in the list to self._layers
        for layer in self.residual_blocks:
            self.__setattr__(layer.name, layer)

        # Author: @karolbadowski.
        output_slice_index = int(self.build_output_shape.as_list()[1] / 2) if self.padding == 'same' else -1
        self.lambda_layer = tf.keras.layers.Lambda(lambda tt: tt[:, output_slice_index, :])
        self.lambda_ouput_shape = self.lambda_layer.compute_output_shape(self.build_output_shape)

    def compute_output_shape(self, input_shape):
        """
        Overridden in case keras uses it somewhere... no idea. Just trying to avoid future errors.
        """
        if not self.built:
            self.build(input_shape)
        if not self.return_sequences:
            return self.lambda_ouput_shape
        else:
            return self.build_output_shape

    def call(self, inputs, training=None):
        x = inputs
        self.layers_outputs = [x]
        try:
            x = self.main_conv1D(x)
            self.layers_outputs.append(x)
        except AttributeError:
            import sys
            sys.exit(0)
        self.skip_connections = []
        for layer in self.residual_blocks:
            x, skip_out = layer(x, training=training)
            self.skip_connections.append(skip_out)
            self.layers_outputs.append(x)

        if self.use_skip_connections:
            x = layers.add(self.skip_connections)
            self.layers_outputs.append(x)
        if not self.return_sequences:
            x = self.lambda_layer(x)
            self.layers_outputs.append(x)
        return x

    def get_config(self):
        """
        Returns the config of a the layer. This is used for saving and loading from a model
        :return: python dictionary with specs to rebuild layer
        """
        config = super(TCNLayer, self).get_config()
        config['nb_filters'] = self.nb_filters
        config['kernel_size'] = self.kernel_size
        config['nb_stacks'] = self.nb_stacks
        config['dilations'] = self.dilations
        config['padding'] = self.padding
        config['use_skip_connections'] = self.use_skip_connections
        config['dropout_rate'] = self.dropout_rate
        config['return_sequences'] = self.return_sequences
        config['activation'] = self.activation
        config['use_batch_norm'] = self.use_batch_norm
        config['use_layer_norm'] = self.use_layer_norm
        config['kernel_initializer'] = self.kernel_initializer
        return config