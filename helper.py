import torch.nn as nn
import torch
# from torch.utils.serialization import load_lua as llua
import tensorflow as tf
import tensorflow.contrib.rnn as rnn_t
from tensorflow.contrib.rnn.python.ops import rnn_cell as rnn
# import tensorflow.contrib.grid_rnn as grnn
# https://github.com/tensorflow/tensorflow/blob/r1.8/tensorflow/contrib/rnn/python/ops/rnn_cell.py
# implementation of GNN (Gated Neighborhood Network)
#  we make use of peephole connections as grid lstm emerged from highway networks

class neighborhood_vis_loc_encoder():
    #  TODO code
    #     define correlation between vislets and locations and let the higher correlations define
    #     the zone of interest (region of interest that is influential to pedestrian, encode temporal features
    #     about the active hotspot/ salient interaction spot)
    #     try rnn as encoder for features
    #     for now multiple cues are concatenated.
    def __init__(self, hidden_size,hidden_len, num_layers, grid_size, embedding_size,dropout=0):
        # super(neighborhood_vis_loc_encoder, self).__init__()
        # tf.Variable()

        reg_w = tf.contrib.layers.l2_regularizer(scale=0.1)
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size
        self.input = tf.placeholder(dtype=tf.float64, shape=[hidden_len,hidden_len], name="inputs")
        self.state_f00_b00_c = tf.placeholder(name='state_f00_b00_c', shape=(hidden_len, self.hidden_size), dtype=tf.float64)
        self.c_hidden_state = tf.placeholder(name='c_hidden_state', shape=(hidden_len, self.hidden_size), dtype=tf.float64)

        self.state_f00_b00_m = tf.placeholder(name='state_f00_b00_m', shape=(hidden_len, (grid_size * (grid_size/2))), dtype=tf.float64)
        self.num_freq_blocks = tf.placeholder(name='num_freq_blocks', dtype=tf.float32)

        self.rnn = rnn.GridLSTMCell(num_units=num_layers,
                                    feature_size=grid_size,
                                    frequency_skip=grid_size,
                                    use_peepholes=True,
                                    num_frequency_blocks=[int(grid_size/2)],
                                    share_time_frequency_weights=True,
                                    state_is_tuple=False,
                                    couple_input_forget_gates=True,
                                    reuse=tf.AUTO_REUSE)

        # self.rnn = rnn_t.LSTMCell(num_units=num_layers, name='nghood_lstm', use_peepholes=True,
        #                           initializer='normal', dynamic=True,
        #                           dtype=tf.float64)

        self.output = tf.placeholder(dtype=tf.float64, shape=[hidden_len, (grid_size * (grid_size/2))],
                                     name="output")
        # .add_weight(name='weight', shape=(2,hidden_size))
                                    # initial_value=init_w(shape=(2,hidden_size)),
                                    # trainable=True
                                    #regularizer= reg_w.l2(weights=init_w(shape=(2,hidden_size)))
        # self.rnn = nn.GRU(input_size, hidden_size, num_layers,
        #                   batch_first=True, bidirectional=False, dropout=dropout)
        # self.rnn.add_weight(name='weight', shape=(2,hidden_size),
        #                     trainable=True,initializer=init_w(shape=(2,hidden_size)))
        self.forward()

    def update_input_size(self, new_size):
        self.input = tf.placeholder(dtype=tf.float64, shape=[new_size, new_size], name="inputs")
        self.hidden_state = tf.placeholder(name='hidden_state', shape=(new_size, self.hidden_size), dtype=tf.float64)

    def forward(self):
        # vislet, location = *input[0], *input[1]
        # Combine both features
        # hidden = tf.convert_to_tensor(hidden, dtype=tf.float64)
        # input = tf.reshape
        # input = tf.nn.relu_layer(x=input , weights=tf)

        self.output, self.c_hidden_state = self.rnn(inputs=self.input, state=self.state_f00_b00_c)
        # self.input = tf.transpose(output)
        # self.hidden_state = tf.transpose(hidden_state)
        # transform from tf class to Pytorch tensors signature
        # return tf.transpose(output), tf.transpose(new_hidden)

    def init_hidden(self, size):
        return tf.zeros(name='hidden_state',shape=(size, self.hidden_size), dtype=tf.float64)

class neighborhood_stat_enc():
    # TODO code for later experiment
    #   define correlation static neighborhood and pedestrians relative distances to context
    #   run kernel over region around pedestrian to specify static context features and determine
    #   the static spot that influence pedestrian path
    #   we need context features (cnn ?)
    #   or grid of fixed lstm placements for top view.
    #   lstm grid network can be customized for on-board camera.
    #   consider the frame discretization in (Choi et al 2019)
    #   they used convolutional features for H-S and H-H
    #    install a grid checkpoints once, pedestrian enter a checkpoint share parameters from previous checkpoint
    #    between new checkpoints and its surrounding checkpoints
    #    each checkpoint forms potential neighborhood and it's a small region of the static scene
    #  how is this different from social lstm 2D grid mask?
    #  how is this different from sharing pattern in LSTMGrid?
    #  In order for us to know how neighborhood forms we need to
    #  understand pedestrians awareness state and know how they perceive
    #  environmental and social factors
    #  passing features between past neighborhood and new neighborhoods and
    #  their surrounding neighborhoods will enable the model of anticipating future area of anvigation and strict a band of future trajectory
    #  the idea here is to use grid ecnoder comprised of LSTM instead of cnn (convolutions)
    #  the passed and combined (summation) spatio-temporal features will be passed through soft-attention for weighing out the salient ROI
    #  and the random walker will take the pedestrian spatial features for predicting relations and completing the edges
    # Social-LSTM 2D grid is occupancy filter for pooling inside single neighborhood area
    # test stvp that pools future paths together and not only the present paths
    # after sharing parameters with new checkpoint run random walker to anticipate future region of interest that pedestrian is likely to navigate through
    # based on their social interactions and vfoa
    # our grid LSTM is responsible for controlling message passing about pedestrians H-S and H-H states between neighborhoods

    # TODO MX-LSTM deploys on Basic LSTM cell implementation of Social LSTM
    # try grid lstm cell in place of cnn to encode spatial features
    # to encode relative spatial interactions human-space combine both neighborhood networks,
    # the static neighborhood network encodes the poi in the scene first
    def __init__(self, hidden_size, num_layers,grid_size, dim, num_nodes,dropout=0):
        # super(neighborhood_stat_enc, self).__init__()

        self.hidden_size = hidden_size
        # in gridLSTM frequency blocks are the units of lstm stacking vertically
        # while time LSTM spans horizontally
        self.rnn = rnn.GridLSTMCell(num_units=num_layers,
                                    feature_size=grid_size,
                                    frequency_skip=grid_size,
                                    use_peepholes=False,
                                    num_frequency_blocks=[int(grid_size/2)],
                                    share_time_frequency_weights=True,
                                    state_is_tuple=False,
                                    couple_input_forget_gates=True,
                                    reuse=True)

        self.static_mask = tf.placeholder(name='static_mask', shape=(dim,num_nodes), dtype=tf.float64)
        self.social_frame = tf.placeholder(name='social_frame', shape=(num_nodes,dim), dtype=tf.float64)
        self.state_f00_b00_c = tf.placeholder(name='state_f00_b00_c', shape=(num_nodes,hidden_size), dtype=tf.float64)
        self.c_hidden_states = tf.placeholder(name='c_hidden_states',
                                              shape=(num_nodes, (grid_size * (grid_size/2))), dtype=tf.float64)

        self.state_f00_b00_m = tf.placeholder(name='state_f00_b00_m',
                                              shape=(num_nodes, (grid_size * (grid_size/2))), dtype=tf.float64)

        self.output = tf.placeholder(dtype=tf.float64, shape=[num_nodes, (grid_size * (grid_size / 2))],
                                     name="output")

        self.forward(self.static_mask , self.social_frame, self.state_f00_b00_c)
        # self.rnn = nn.GRUCell(input_size, hidden_size, num_layers)
        # GRUCell is stacked GRU model but it doesnt provide Grid scheme of sharing weights along multidimensional GRU
        # llua('/home/serene/PycharmProjects/multimodaltraj/grid-lstm-master/model/GridLSTM.lua')

    def forward(self, input, social_frame, hidden):
        # map locations []
        # use social frame in filling occupancy state of the static mask.
        # opposed to occupancy map in Social LSTM (2016) which used binary occupancy descriptors
        # our occupancy map has weighted descriptors.
        # In algebra multiplying two feature vector calculates distance
        # between two corresponding points in Euclidean geometry
        # such that this relative distance embeddings now leads us to weight the frame neighborhoods and
        # evaluate occupancy inside each local neighborhood

        input = tf.matmul(b=input, a=social_frame)# Soft-attention mechanism equipped with static grid
        # input = tf.matmul(input, tf.ones_like(tf.transpose(input)))
        # input = tf.transpose(input)
        self.output, self.c_hidden_states = self.rnn(inputs=input, state=hidden)

        # return output, new_hidden
