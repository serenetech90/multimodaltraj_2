# import torch
import torch.nn as nn
import tensorflow as tf
# import tensorflow.gra as grad
import nri_learned as nri
import networkx_graph as nx
from torch.nn import functional as F
import torch.cuda

class g2k_lstm_mcr(nn.Module):
    def __init__(self, in_features, out_size):
        super(g2k_lstm_mcr).__init__()
        # self.relu = tf.nn.relu_layer()
        self.out_size = out_size
        self.init_w = tf.initializers.random_normal(mean=0, stddev=1,seed=0,dtype=tf.float64)
        self.weight_k = tf.Variable(name='weight_k', initial_value= \
            self.init_w(shape=(in_features.shape[0].value, out_size)),
                                    # shape=tf.shape(1,in_features.shape[1].value),
                                    dtype=tf.float64)
        self.bias_k = tf.Variable(name='bias_k', initial_value= \
            self.init_w(shape=(out_size,)),
                                  # shape=(in_features.shape[1].value, 1),
                                  dtype=tf.float64)

        self.weight_v = tf.Variable(name='weight_v', initial_value= \
            self.init_w(shape=(out_size, 4)),
            # shape=tf.shape(1,in_features.shape[1].value),
            dtype=tf.float64)

        self.bias_v = tf.Variable(name='bias_v', initial_value= \
            self.init_w(shape=(4,)),
            # shape=tf.shape(1,in_features.shape[1].value),
            dtype=tf.float64)

        # def randomWalker(self, in_features, w, b):
        #
        #     # random walks with restarts to estimate pedestrians proximecs.
        #     # consider lars et al. 2011 Supervised
        #     # weighted random walker (use hard-attention mechanism that relies on VFOA)
        #     # kernel = random_walk ...
        #     return

    def forward(self, outputs,ngh, visual_path):
        # st_graph = nodes
        # pred_path_band = self.randomWalker(graph=st_graph, edges_mat=edges)

        embedded_spatial_vislet = tf.nn.relu(tf.nn.xw_plus_b(visual_path, self.weight_v, self.bias_v))
        ngh = tf.multiply(embedded_spatial_vislet, ngh)
        # ngh = tf.Variable(initial_value=tf.multiply(embedded_spatial_vislet, ngh),
        #                   trainable=True,
        #                   name='ngh')
        # self.mlp()
        # with tf.GradientTape() as t:
        #     t.watch(outputs)
        #     # out = f(outputs, ngh)
        # return t.gradient(outputs, ngh)

        # pred_path_band = grad.jacobian(output=outputs, inputs=ngh)
        # d_outputs need to be square matrix (positive-definite) NOT necessary
        # convex loss
        # Jacobian is the following m × n matrix
        # compute neighborhood as a function of the social and spatial features
        # determined by social embedded features.

        ys_temp = tf.zeros_like(ngh)
        d_outputs = tf.gradients(ys=ngh, xs=outputs, grad_ys=ys_temp,
                                 unconnected_gradients='zero')

        pred_path_band = tf.transpose(d_outputs) * self.weight_k + self.bias_k
        # pred_path_band = tf.nn.xw_plus_b(x=tf.transpose(d_outputs), weights=self.weight_k, biases=self.bias_k)
        # estimate gradient of every pedestrian function using Jacobian (matrix calculus).

        return pred_path_band, d_outputs

    def backward(self):
        return

    def register_backward_hook(self, hook):
        return





