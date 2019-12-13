import tensorflow as tf

class g2k_lstm_mcr():
    def __init__(self, in_features, hidden_size, obs_len, num_nodes, lambda_reg, sess_g):
        with tf.Session(graph=sess_g) as sess:
            # super(g2k_lstm_mcr).__init__()
            # self.relu = tf.nn.relu_layer()
            self.out_size = tf.placeholder_with_default(input=num_nodes, shape=[], name='out_size')
            self.lambda_reg = lambda_reg
            self.init_w = tf.initializers.random_normal(mean=0, stddev=1, seed=0, dtype=tf.float64)
            # in_features 1x2xnum_nodes
            # find intersection between two points on polar grid
            self.outputs = tf.placeholder_with_default(input=tf.random.normal(shape=[int(in_features.shape[0])+2,
                                                 int(in_features.shape[0])],
                                                   mean=0, stddev=1, seed=0, dtype=tf.float64),#dtype=tf.float64,
                                          shape=[int(in_features.shape[0])+2,
                                                 int(in_features.shape[0])],name="outputs")

            self.rel_features = tf.placeholder_with_default(input=tf.random.normal(
                                                            shape=[2, int(in_features.shape[0])],
                                                            mean=0, stddev=1, seed=0, dtype=tf.float64),
                                                            shape=[2, int(in_features.shape[0])], name="rel_features")  # 2x10

            # dtype=tf.float64,
            self.visual_path = tf.placeholder_with_default(input=tf.random.normal(shape=[2, int(in_features.shape[0])],
                                                   mean=0, stddev=1, seed=0, dtype=tf.float64), #dtype=tf.float64,
                                                   shape=[2, in_features.shape[0]], name="visual_path")

            # self.pred_path_band = \
                # tf.placeholder_with_default(input= tf.random.normal(shape=[2,12,self.out_size.eval()],
                #                                               mean=0, stddev=1, seed=0, dtype=tf.float64),
                #                                               shape=[2, 12, None], name="pred_path_band")

            self.ngh = tf.placeholder_with_default(input=tf.random.normal(shape=[int(in_features.shape[0]), obs_len],
                                                   mean=0, stddev=1, seed=0, dtype=tf.float64),
                                                   shape=[int(in_features.shape[0]), obs_len], name="ngh")

            with tf.variable_scope("krnl_weights", reuse=True):
                self.cost = tf.Variable(name='cost', initial_value=
                sess_g.get_tensor_by_name(name='krnl_weights_21/cost:0'),
                                        # self.init_w(shape=[obs_len, obs_len],dtype=tf.float64),
                                        shape=[obs_len, obs_len])

                self.attn = tf.Variable(name='attn',
                                        initial_value= sess_g.get_tensor_by_name(name='krnl_weights_21/attn:0'),
                                        #self.init_w(shape=[int(in_features.shape[0]), int(in_features.shape[0])], dtype=tf.float64),
                                        shape=[int(in_features.shape[0]),int(in_features.shape[0])])

                self.weight_v = tf.Variable(name='weight_v', initial_value= \
                    sess_g.get_tensor_by_name(name='krnl_weights_21/weight_v:0'),
                                            # self.init_w(shape=(obs_len, int(in_features.shape[0])+2)),
                                            # shape=tf.shape(1,in_features.shape[1].value),
                                            dtype=tf.float64)

                self.bias_v = tf.Variable(name='bias_v', initial_value= \
                    sess_g.get_tensor_by_name(name='krnl_weights_21/bias_v:0'),
                                          # self.init_w(shape=(int(in_features.shape[0]),)),
                                          # shape=tf.shape(1,in_features.shape[1].value),
                                          dtype=tf.float64)

                self.weight_o = tf.Variable(name='weight_o', initial_value= \
                                            self.init_w(shape=(obs_len, num_nodes)),#int(in_features.shape[0])
                                            # shape=tf.shape(1,in_features.shape[1].value),
                                            dtype=tf.float64)
                self.weight_c = tf.Variable(name='weight_c', initial_value= \
                    sess_g.get_tensor_by_name(name='krnl_weights_21/weight_c:0'),
                                            # self.init_w(shape=(24, obs_len)),# 16 when pred_len = 8
                                            # shape=tf.shape(1,in_features.shape[1].value),
                                            dtype=tf.float64)

            with tf.variable_scope('krnl_embed', reuse=True):
                self.weight_r = tf.Variable(name='weight_r', initial_value= \
                    sess_g.get_tensor_by_name(name='krnl_embed_21/weight_r:0'),
                                              # self.init_w(shape=(obs_len, 2)),
                                              # shape=tf.shape(1,in_features.shape[1].value),
                                              dtype=tf.float64)
                # self.embed_vis = tf.Variable(name='embed_vis', initial_value= \
                #                             self.init_w(shape=(obs_len, int(in_features.shape[0]))),
                #                             # shape=tf.shape(1,in_features.shape[1].value),
                #                             dtype=tf.float64)

            # self.weight_r = tf.Variable(sess_g.get_tensor_by_name(name='krnl_embed_4/weight_r_1:0'))
            # self.weight_v = tf.Variable(sess_g.get_tensor_by_name(name='krnl_weights_4/weight_v_1:0'))
            # self.weight_c = tf.Variable(sess_g.get_tensor_by_name(name='krnl_weights_4/weight_c_1:0'))
            # # self.weight_o = tf.Variable(sess_g.get_tensor_by_name(name='krnl_weights_4/weight_o_1:0'))
            # self.attn = tf.Variable(sess_g.get_tensor_by_name(name='krnl_weights_4/attn_1:0'))
            # self.cost = tf.Variable(sess_g.get_tensor_by_name(name='krnl_weights_4/cost_1:0'))
            # self.bias_v = tf.Variable(sess_g.get_tensor_by_name(name='krnl_weights_4/bias_v_1:0'))

            self.hidden_states = tf.placeholder_with_default(
                                    input=tf.random.normal(
                                    shape=[int(in_features.shape[0]), hidden_size],
                                    mean=0, stddev=1, seed=0, dtype=tf.float64),
                                    shape=[int(in_features.shape[0]), hidden_size], name="hidden_states")  # 2x10

            self.forward()
            sess.close()

    def forward(self):
        # embed using MLP
        # self.embedded_spatial_vislet = tf.Variable(tf.matmul(self.weight_v, self.outputs) + self.bias_v)  # 12x10
        self.ngh = tf.Variable((self.lambda_reg * self.ngh))# 12x10
        if self.rel_features.shape[0] > 0:
            # embed_vis_feature = tf.Variable(tf.matmul(self.weight_r, self.rel_features))
            self.attn = tf.Variable(tf.matmul(self.ngh, tf.multiply(tf.matmul(self.weight_v, self.outputs) + self.bias_v,
                                                                    tf.matmul(self.weight_r, self.rel_features))))

            # self.attn = tf.Variable(tf.matmul( self.ngh, self.attn ))
            # self.attn = tf.Variable(tf.nn.softmax(self.attn))

        # temp = tf.Variable(
        self.cost = tf.Variable(tf.matmul(tf.matmul(self.weight_v, self.outputs) + self.bias_v
                                , self.ngh))

        # _, self.cost = tf.gradients(ys=self.ngh, xs=[embedded_spatial_vislet, ngh],
        #                             stop_gradients=embedded_spatial_vislet,
        #                             unconnected_gradients='zero')
        # Check if this proper transformation ??? can we gt prob from ngh features

        # self.cost = tf.squeeze(self.cost)
        # self.cost = tf.Variable(tf.nn.softmax(self.cost))
        self.temp_path = tf.Variable(tf.matmul(tf.matmul(self.weight_c, self.cost) , self.weight_o))  # 16x10
        # self.temp_path = tf.Variable(tf.matmul(self.temp_path, self.weight_o)) # 16xn
        self.pred_path_band = tf.reshape(self.temp_path, (2, 12, self.out_size.eval()))  # 2x12xn
