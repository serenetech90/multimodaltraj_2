import tensorflow as tf

class g2k_lstm_mc():
    def __init__(self, in_features, out_size, obs_len, num_nodes, lambda_reg):
        # super(g2k_lstm_mcr).__init__()
        # self.relu = tf.nn.relu_layer()
        self.out_size = num_nodes
        self.lambda_reg = tf.Variable(lambda_reg, dtype=tf.float64)
        self.init_w = tf.initializers.random_normal(mean=0, stddev=1, seed=0, dtype=tf.float64)

        self.outputs = tf.placeholder_with_default(input=tf.random.normal(shape=[int(in_features.shape[0])+2,
                                             int(in_features.shape[0])],
                                               mean=0, stddev=1, seed=0, dtype=tf.float64),#dtype=tf.float64,
                                      shape=[int(in_features.shape[0])+2,
                                             int(in_features.shape[0])],name="outputs")

        self.visual_path = tf.placeholder_with_default(input=tf.random.normal(shape=[1, int(in_features.shape[0])],
                                               mean=0, stddev=1, seed=0, dtype=tf.float64), #dtype=tf.float64,
                                          shape=[1, in_features.shape[0]], name="visual_path")

        self.pred_path_band = tf.placeholder_with_default(input=tf.random.normal(shape=[2,12,num_nodes],
                                               mean=0, stddev=1, seed=0, dtype=tf.float64), #dtype=tf.float64,
                                             shape=[2,12,num_nodes], name="pred_path_band")

        self.ngh = tf.placeholder_with_default(
            input=tf.random.normal(shape=[int(in_features.shape[0]), obs_len],
                                   mean=0, stddev=1, seed=0, dtype=tf.float64),
            shape=[int(in_features.shape[0]), obs_len], name="ngh")

        with tf.variable_scope("krnl_weights"):
            self.weight_v = tf.Variable(name='weight_v', initial_value= \
                self.init_w(shape=(obs_len, int(in_features.shape[0])+2)),
                                        # shape=tf.shape(1,in_features.shape[1].value),
                                        dtype=tf.float64)

            self.bias_v = tf.Variable(name='bias_v', initial_value= \
                                      self.init_w(shape=(int(in_features.shape[0]),)),
                                      # shape=tf.shape(1,in_features.shape[1].value),
                                      dtype=tf.float64)

            self.weight_o = tf.Variable(name='weight_o', initial_value= \
                                        self.init_w(shape=(obs_len, num_nodes)),#int(in_features.shape[0])
                                        # shape=tf.shape(1,in_features.shape[1].value),
                                        dtype=tf.float64)

            self.weight_c = tf.Variable(name='weight_c', initial_value= \
                                        self.init_w(shape=(24, obs_len)),# 16 when pred_len = 8
                                        # shape=tf.shape(1,in_features.shape[1].value),
                                        dtype=tf.float64)

        self.forward()


    def forward(self):

        embedded_spatial_vislet = tf.Variable(tf.matmul(self.weight_v, self.outputs) + self.bias_v)  # 12x10
        # ngh_temp = tf.Variable((self.lambda_reg * self.ngh) )# 12x10
        ngh = tf.Variable(tf.matmul(embedded_spatial_vislet, self.ngh))
        _, self.cost = tf.gradients(ys=self.ngh, xs=[embedded_spatial_vislet, ngh],
                                    stop_gradients=embedded_spatial_vislet,
                                    unconnected_gradients='zero')

        self.cost = tf.squeeze(self.cost)
        # self.cost = tf.nn.relu(self.cost)

        self.temp_path = tf.Variable(tf.matmul(self.weight_c, self.cost))  # 16x10
        self.temp_path = tf.Variable(tf.matmul(self.temp_path, self.weight_o))  # 16xn

        self.pred_path_band = tf.reshape(self.temp_path, (2,12 , self.out_size))  # 2x12xn

