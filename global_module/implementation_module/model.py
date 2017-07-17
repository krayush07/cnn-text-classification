import tensorflow as tf
from global_module.settings_module import set_params, set_dir


class CNNClassification():
    def __init__(self, params, dir_obj):
        self.params = params
        self.dir_obj = dir_obj
        self.call_pipeline()

    def call_pipeline(self):
        self.create_placeholders()

    def create_placeholders(self):
        self.word_emb_matrix = tf.get_variable("word_embedding_matrix", shape=[self.params.vocab_size, self.params.EMB_DIM], dtype=tf.float32)
        self.word_input = tf.placeholder(name="word_input", shape=[self.params.BATCH_SIZE, self.params.MAX_SEQ_LEN], dtype=tf.int32)
        self.labels = tf.placeholder(name="labels", shape=[self.params.BATCH_SIZE], dtype=tf.float32)

    def convolution_layer(self, input, filter, stride, padding, activation, name):
        with tf.variable_scope(name):
            weights = tf.get_variable(name='weights',
                                      shape=filter,
                                      initializer=tf.random_uniform_initializer(minval=-0.1, maxval=0.1))

            biases = tf.get_variable(name='biases',
                                     shape=filter[-1],
                                     initializer=tf.constant_initializer(0.0))

            # TODO: check the data alignment (NHWC)
            conv = tf.nn.conv2d(name="convolution",
                                input=input,
                                filter=weights,
                                strides=stride,
                                padding=padding)

            if (activation == 'RELU'):
                return tf.nn.relu(tf.nn.bias_add(conv, biases), name="relu")
            elif (activation == 'TANH'):
                return tf.nn.tanh(tf.nn.bias_add(conv, biases), name="tanh")

    def max_pool(self, input, pool_size, stride, padding, name):
        with tf.variable_scope(name):
            return tf.nn.max_pool(name="avg_pool",
                                  value=input,
                                  ksize=pool_size,
                                  strides=stride,
                                  padding=padding)

    def avg_pool(self, input, pool_size, stride, padding, name):
        with tf.variable_scope(name):
            return tf.nn.avg_pool(name="avg_pool",
                                  value=input,
                                  ksize=pool_size,
                                  strides=stride,
                                  padding=padding)

    def pooling_layer(self, input, pool_size, stride, padding, pool_option, name):
        if (pool_option == 'MAX'):
            return self.max_pool(input, pool_size, stride, padding, name)
        elif (pool_option == 'AVG'):
            return self.avg_pool(input, pool_size, stride, padding, name)

    def conv_pool_block(self, layer_num, input):
        pooled_output = []
        for i in range(len(self.params.filder_width[layer_num])):
            filter_width = self.params.filder_width[layer_num][i]
            num_filters = self.params.num_filter[i]

            conv_output = self.convolution_layer(input=input,
                                                 filter=[filter_width, self.params.EMB_DIM, 1, num_filters],
                                                 stride=[1, 1, 1, 1],
                                                 padding=self.params.conv_padding,
                                                 activation=self.params.conv_activation,
                                                 name="conv_layer_" + str(layer_num))

            pool_output = self.pooling_layer(input=conv_output,
                                             pool_size=[1, self.params.pool_width, 1, 1],
                                             stride=[1, self.params.pool_stride, 1, 1],
                                             padding=self.params.pool_padding,
                                             pool_option=self.params.pool_option,
                                             name="pool_layer_" + str(layer_num))

            pooled_output.append(pool_output)

        return pooled_output

    def create_network_pipeline(self):
        self.input_matrix = tf.nn.embedding_lookup(self.word_emb_matrix, self.word_input, name="emb_lookup")
        self.sent_input_matrix = tf.expand_dims(self.input_matrix, -1)

        # TODO : complete the code
        #for i in range(len(self.params.num_filter)):



def main():
    params = set_params.ParamsClass(mode='TR')
    dir_obj = set_dir.Directory('TR')
    multi_view_obj = CNNClassification(params, dir_obj)


if __name__ == '__main__':
    main()
