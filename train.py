from models import g2k_lstm_mcr as mcr
import argparse
import nri_learned as nri
import networkx_graph as nx_g
import load_traj as load
import torch
import tensorflow as tf
import helper
import numpy as np
import os
import MX_LSTM.grid as grid

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--input_size', type=int, default=2,
                        help='size of input features vector')

    # RNN size parameter (dimension of the output/hidden state)
    parser.add_argument('--rnn_size', type=int, default=128,
                        help='size of RNN hidden state')
    # TODO: (improve) Number of layers not used. Only a single layer implemented
    # Number of layers parameter
    parser.add_argument('--num_layers', type=int, default=2,
                        help='number of layers in the RNN')
    # Model currently not used. Only LSTM implemented
    # Type of recurrent unit parameter
    parser.add_argument('--model', type=str, default='lstm',
                        help='rnn, gru, or lstm')
    # Size of each batch parameter
    parser.add_argument('--batch_size', type=int, default=16, # read 16 frames at once containing all related pedestrians and their trajectories
                        help='minibatch size')
    # Length of sequence to be considered parameter
    parser.add_argument('--seq_length', type=int, default=8,
                        help='RNN sequence length')
    parser.add_argument('--pred_len', type=int, default=8,
                        help='number of layers in the RNN')
    # Number of epochs parameter
    parser.add_argument('--num_epochs', type=int, default=100,
                        help='number of epochs')
    # Frequency at which the model should be saved parameter
    parser.add_argument('--save_every', type=int, default=10,
                        help='save frequency')
    # TODO: (resolve) Clipping gradients for now. No idea whether we should
    # Gradient value at which it should be clipped
    parser.add_argument('--grad_clip', type=float, default=10.,
                        help='clip gradients at this value')
    # Learning rate parameter
    parser.add_argument('--learning_rate', type=float, default=0.005,
                        help='learning rate')
    # Decay rate for the learning rate parameter
    parser.add_argument('--decay_rate', type=float, default=0.95,
                        help='decay rate for rmsprop')
    # Dropout not implemented.
    # Dropout probability parameter
    parser.add_argument('--dropout', type=float, default=0.8,
                        help='dropout keep probability')
    # Dimension of the embeddings parameter
    parser.add_argument('--embedding_size', type=int, default=64,
                        help='Embedding dimension for the spatial coordinates')
    # Size of neighborhood to be considered parameter
    parser.add_argument('--neighborhood_size', type=int, default=32,
                        help='Neighborhood size to be considered for social grid')
    # Size of the social grid parameter
    parser.add_argument('--grid_size', type=int, default=4,
                        help='Grid size of the social grid')

    parser.add_argument('--num_freq_blocks', type=int, default=10,
                        help='Grid size of the social grid')

    # Maximum number of pedestrians to be considered
    parser.add_argument('--maxNumPeds', type=int, default=20,
                        help='Maximum Number of Pedestrians')
    # The leave out dataset
    parser.add_argument('--leaveDataset', type=int, default=3,
                        help='The dataset index to be left out in training')
    # Lambda regularization parameter (L2)
    parser.add_argument('--lambda_param', type=float, default=0.0005,
                        help='L2 regularization parameter')
    args = parser.parse_args()
    train(args)

    return

def train(args):
    dataloader = load.DataLoader(args=args, datasets=[0,1,2,3,4,5], sel=1)
    target_traj = []
    true_path = []
    # Train the model
    e = 0
    euc_loss = 0
    frame = 1

    parent_dir = '/home/serene/PycharmProjects/multimodaltraj/kernel_models/MX-LSTM-master/data'
    traj = dataloader.load_trajectories(data_file=dataloader.sel_file)

    tf_graph = tf.Graph()
    dim = int(args.neighborhood_size / args.grid_size)

    # tf.disable_eager_execution()
    graph = nx_g.online_graph(args)
    with tf.Session(graph=tf_graph) as sess:
        with sess.as_default():
            batch, target_traj, _ = dataloader.next_step(targets=target_traj)
            # true_path.append(batch[args.seq_length + 1])
            # correct batch len
            graph_t = graph.ConstructGraph(current_batch=batch, future_traj=target_traj , framenum=1)
            batch_v = list(graph_t.get_node_attr(param='node_pos_list').values())
            batch_v = np.transpose(batch_v)
            num_nodes = batch_v.shape[1]

            # TODO augment vislets later

            vislet = np.zeros(shape=(1,args.num_freq_blocks))

            with tf.variable_scope('weight_input'):
                init_w = tf.initializers.random_normal(mean=0, stddev=1, seed=0, dtype=tf.float64)
                weight_i = tf.Variable(name='weight_i', initial_value=init_w(shape=(num_nodes, args.num_freq_blocks)),
                                       trainable=True, dtype=tf.float64)
                weight_ii = tf.Variable(name='weight_ii',
                                        initial_value=init_w(shape=(args.num_freq_blocks, args.input_size)),
                                        trainable=True, dtype=tf.float64)

            vislet = tf.expand_dims(batch_v[0], axis=0)
            vislet_emb = tf.matmul(vislet, weight_i)
            # salient social interaction spot
            # GNN component
            # cat = batch_v.shape[1] - batch_v.shape[0]
            # batch_v = tf.zeros(shape=(batch_v.shape[1], batch_v.shape[1])) + tf.convert_to_tensor(batch_v, dtype=tf.float64)
            # batch_v = np.concatenate((batch_v, np.zeros(shape=(cat, num_nodes))), axis=0)

            nghood_enc = helper.neighborhood_vis_loc_encoder(
                hidden_size=args.rnn_size,
                hidden_len=args.num_freq_blocks,
                num_layers=args.num_layers,
                grid_size=args.grid_size,
                embedding_size=args.embedding_size,
                dropout=args.dropout)

            # hidden_state = np.zeros(shape=(batch_v.shape[1], args.rnn_size))

            stat_ngh = helper.neighborhood_stat_enc(
                hidden_size=args.rnn_size,
                num_layers=args.num_layers,
                grid_size=args.grid_size,
                dim=dim,
                num_nodes=num_nodes,
                dropout=args.dropout)

            stat_mask = tf.zeros(shape=(dim, num_nodes), dtype=tf.float64)
            stat_mask += tf.expand_dims(tf.range(start=0, limit=1, delta=0.125, dtype=tf.float64), axis=1)
            static_mask_nd = stat_mask.eval()

            krnl_mdl = mcr.g2k_lstm_mcr(in_features=nghood_enc.input , out_size=batch_v.shape[1],
                                        num_nodes=num_nodes, obs_len=args.seq_length,
                                        lambda_reg=args.lambda_param)

            # sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver(tf.all_variables(), max_to_keep=None)
            print('session started')
            start = args.seq_length + 1
            end = int(len(batch)/(args.seq_length + 1))

            inputs = tf.convert_to_tensor(batch_v, dtype=tf.float64)
            inputs = tf.matmul(inputs, weight_i)
            inputs = tf.matmul(weight_ii, inputs)
            hidden_state = np.zeros(shape=(args.num_freq_blocks, args.rnn_size))

            # dim = [720, 576]
            # Train
            while e < args.num_epochs:
                for b in range(dataloader.num_batches):
                    with tf.variable_scope('nghood_init'):
                        out_init = tf.zeros(dtype=tf.float64,shape=(args.num_freq_blocks, (args.grid_size * (args.grid_size/2))))
                        c_hidden_init = tf.zeros(dtype=tf.float64,shape=(args.num_freq_blocks,(args.grid_size * (args.grid_size/2))))

                    tf.initialize_variables(var_list=[weight_i, weight_ii]).run()
                    # frame = list(batch.keys())[0]
                    for frame in batch:
                        # check if get_node_attr gets complete sequence for all nodes
                        # num_nodes x obs_length
                        try:
                            true_path.append(batch[frame])
                        except KeyError:
                            if frame == max(batch.keys()):
                                break
                            if frame+args.seq_length+1 > max(batch.keys()):
                                frame = max(batch.keys())
                            else:
                                frame += args.seq_length + 1
                            continue
                        # if len(batch_v) != nghood_enc.input.shape[0]:
                        #     nghood_enc.update_input_size(new_size=len(batch_v))
                        # hidden_state = nghood_enc.init_hidden(len(batch_v))

                        st_embeddings, hidden_state, output, c_hidden_state =\
                            sess.run([nghood_enc.input, nghood_enc.state_f00_b00_c,
                                      nghood_enc.output, nghood_enc.c_hidden_state],
                                    feed_dict={nghood_enc.input: inputs.eval(),
                                               nghood_enc.state_f00_b00_c: hidden_state,
                                               nghood_enc.output:out_init.eval(),
                                               nghood_enc.c_hidden_state: c_hidden_init.eval()})

                        # st_embeddings, hidden_state = nghood_enc.forward(batch_v, hidden_state)
                        # input_size=args.input_size,
                        # salient static spot
                        # generate weighted embeddings of spatial context in the scene
                        # input_size=args.input_size,
                        # tf.variables_initializer(var_list=[self.weight_k, self.bias_k])
                        # sess.run(tf.initialize_all_variables())
                        # stat_embed = grid.getSequenceGridMask(sequence= batch_v,
                        #                                       dimensions= dim,
                        #                                       neighborhood_size=args.neigborhood,
                        #                                       grid_size= args.grid_size)
                        # tf.initialize_all_variables().run()
                        # st_embeddings = nghood_enc.input
                        # st_embeddings = nghood_enc.output.eval()
                        # hidden_state = nghood_enc.c_hidden_state.eval()
                        # hidden_state = nghood_enc.c_hidden_state.eval()

                        combined_ngh, hidden_state = \
                            sess.run([stat_ngh.static_mask, stat_ngh.state_f00_b00_c],
                                     feed_dict={stat_ngh.static_mask: static_mask_nd,
                                                stat_ngh.social_frame: output,
                                                stat_ngh.state_f00_b00_c: hidden_state,
                                                stat_ngh.output: out_init.eval(),
                                                stat_ngh.c_hidden_states: c_hidden_init.eval()})

                        # to become weighted mask of densest regions (interactive regions / hot-spots )
                        # combined_ngh [8x4] and st_embeddings [8x2] , next use generate vislets features embeddings
                        # reach here, TODO: check if states and frequency blocks output is properly done.
                        # Pass Random Walker on this weighted features
                        # stat_ngh.forward(input=static_mask, social_frame=st_embeddings, hidden=hidden_state)
                        # GNN vs RW in terms of encoding graph structures into smaller pieces (atomic structures)
                        # make use of Jacobi matrix for achieving derivatives of (vector-valued function) f(x)
                        # where x_i is trajectory of pedestrian i. Jacobi matrix will be on random walk process
                        # make loss function results constrained to Jacobian matrix optimization
                        # and transform linearly or using MLP to generate
                        # derivatives of Jacobi matrix, then pass this matrix through an MLP to get the future (x,y) positions
                        # combine vislet with underlying grid embeddings
                        # the visual field will guide the kernel on how to share states between local neighborhoods
                        # accordingly, outputs from the underlying grid after sharing states, are taken as entries
                        # of kernel, forming Jacobian matrix. Transform matrix through mlp into vector of 2 entries
                        # corresponding to future locations.
                        # TODO embed vislet features with both static neighborhood and social neighborhood
                        #      using activations.
                        # edge_mat = tf.zeros(shape=(batch_v.shape[1], batch_v.shape[1]))
                        # sess.run()
                        # with tf.Session() as sess2:
                        # combined_ngh_nd = combined_ngh.eval()
                        # feed = {krnl_mdl.outputs: tf.make_ndarray(st_embeddings),
                        #         krnl_mdl.ngh: tf.make_ndarray(combined_ngh),
                        #         krnl_mdl.visual_path: tf.make_ndarray(vislet)}
                        # run tf session to get through the GridLSTM then continue with pyTorch
                        # krnl_mdl.cost is our relational loss (its loss related to having lower regression curve compared to the all-ones edge matrix)
                        # logistic regression over diagonal values in jacobian (identity) matrix
                        # the krnl_mdl.cost has eigen values where each eigen value is the derivative of each pedestrian with respect to its own trajectory and  affected by ngh * lambda
                        # take each eigen and verify how it can be transformed to sample future trajectory
                        # TODO: the eigen can be mean of future path distribution or can be parameter to set neighborhood boundaries?
                        # vislet = tf.expand_dims(batch_v[0], axis=0)
                        # vislet_emb = tf.matmul(vislet, weight_i)

                        pred_path, jacobian =\
                            sess.run([krnl_mdl.pred_path_band, krnl_mdl.cost],
                                     feed_dict={krnl_mdl.outputs: np.concatenate((st_embeddings,vislet_emb.eval()), axis=0),
                                     krnl_mdl.ngh: combined_ngh,
                                     krnl_mdl.pred_path_band: np.zeros(shape=(2, 8, num_nodes))})

                        # , krnl_mdl.visual_path: vislet.eval()
                        # pred_path, jacobian = sess.run(fetches=krnl_mdl)
                        # pred_path, jacobian = sess.run(krnl_mdl.forward,
                        #                         feed_dict={x:st_embeddings,
                        #                                     y:combined_ngh,
                        #                                     'visual_path':vislet})
                        # jacobian = torch.Tensor(jacobian)
                        # jacobian.backward()
                        # generate weighted embeddings of spatial/temporal motion features in the frame
                        # decode edge_mat embeddings into relations
                        # rlns = tf.Sigmoid(jacobian)
                        adj_mat = nri.eval_rln_ngh(jacobian, combined_ngh)

                        # relational_loss.backward()
                        # nx_g.online_graph.linkGraph(curr_graph=graph_t, new_edges=rlns, frame=frame)
                        # output = krnl_mdl(in_features)
                        # TODO loss type ??? suitable loss to measure
                        # correct tru path
                        # true_path = target_traj[0:num_nodes]
                        pred_path = np.transpose(pred_path, (2,1,0))
                        for i in range(1,num_nodes):
                            try:
                                if len(target_traj[i]) < args.pred_len:
                                    euc_loss = np.linalg.norm((pred_path[i][0:len(target_traj[i])] - target_traj[i]), ord=2)#/num_nodes
                                else:
                                    euc_loss = np.linalg.norm((pred_path[i][0:args.pred_len] - target_traj[i][0:args.pred_len]), ord=2)#/num_nodes
                                    # np.linalg.norm((pred_path[i][0:len(target_traj[i])] - target_traj[i]), ord=2)
                            except KeyError:
                                i += 1
                                continue
                                # print()
                        # euc_loss.backward()
                        # i += args.seq_length + 1
                        # dataloader.tick_frame_pointer(incr= args.seq_length)
                        print('Frame {3} Batch {0} of {1}, Loss = {2}, ADE={4}, num_ped={5}'
                              .format(b, dataloader.num_batches,krnl_mdl.cost, frame, euc_loss, num_nodes))
                        frame += args.seq_length + 1
                    # frame = list(batch.keys())[0]
                    # seed =  frame
                    batch, target_traj, _ = dataloader.next_step(targets=target_traj)
                    # if len(batch) == 0:
                    #     break
                    graph_t = graph.ConstructGraph(current_batch=batch, framenum=frame,future_traj=target_traj)
                    batch_v = list(graph_t.get_node_attr(param='node_pos_list').values())
                    batch_v = np.transpose(batch_v)
                    num_nodes = batch_v.shape[1]

                    with tf.variable_scope('weight_input'):
                        init_w = tf.initializers.random_normal(mean=0, stddev=1, seed=0, dtype=tf.float64)
                        weight_i = tf.Variable(name='weight_i',
                                               initial_value=init_w(shape=(num_nodes, args.num_freq_blocks)),
                                               trainable=True, dtype=tf.float64)
                        weight_ii = tf.Variable(name='weight_ii',
                                                initial_value=init_w(shape=(args.num_freq_blocks, args.input_size)),
                                                trainable=True, dtype=tf.float64)

                    inputs = tf.convert_to_tensor(batch_v, dtype=tf.float64)
                    inputs = tf.matmul(inputs, weight_i)
                    inputs = tf.matmul(weight_ii, inputs)

                    vislet = tf.expand_dims(batch_v[0], axis=0)
                    vislet_emb = tf.matmul(vislet, weight_i)

                    # salient social interaction spot
                    # GNN component
                    # cat = batch_v.shape[1] - batch_v.shape[0]
                    # batch_v = tf.zeros(shape=(batch_v.shape[1], batch_v.shape[1])) + tf.convert_to_tensor(batch_v, dtype=tf.float64)
                    # batch_v = np.concatenate((batch_v, np.zeros(shape=(cat, num_nodes))), axis=0)

                    nghood_enc = helper.neighborhood_vis_loc_encoder(
                        hidden_size=args.rnn_size,
                        hidden_len=args.num_freq_blocks,
                        num_layers=args.num_layers,
                        grid_size=args.grid_size,
                        embedding_size=args.embedding_size,
                        dropout=args.dropout)

                    stat_ngh = helper.neighborhood_stat_enc(
                        hidden_size=args.rnn_size,
                        num_layers=args.num_layers,
                        grid_size=args.grid_size,
                        dim=dim,
                        num_nodes=args.num_freq_blocks,
                        dropout=args.dropout)

                    stat_mask = tf.zeros(shape=(dim, args.num_freq_blocks), dtype=tf.float64)
                    stat_mask += tf.expand_dims(tf.range(start=0, limit=1, delta=0.125, dtype=tf.float64), axis=1)
                    static_mask_nd = stat_mask.eval()

                    krnl_mdl = mcr.g2k_lstm_mcr(in_features=nghood_enc.input, out_size=batch_v.shape[1],
                                                num_nodes=num_nodes, obs_len=args.seq_length,
                                                lambda_reg=args.lambda_param)
                # make another model file with attn
                if (e * dataloader.num_batches + b) % args.save_every == 0 and ((e * dataloader.num_batches + b) > 0):
                    checkpoint_path = os.path.join('/home/serene/PycharmProjects/multimodaltraj/save', 'g2k_mcr_model.ckpt')
                    saver.save(sess, checkpoint_path, global_step=e * dataloader.num_batches + b)
                    print("model saved to {}".format(checkpoint_path))

    # Validate
    # dataloader.reset_data_pointer(valid=True)
    # loss_epoch = 0
    # dataloader.valid_num_batches = dataloader.valid_num_batches + start_epoch
    # for batch in range(dataloader.valid_num_batches):
    #     # Get batch data
    #     x, _, d = dataloader.load_trajectories(data_file='')  ## stateless lstm without shuffling
    #
    #     # Loss for this batch
    #     loss_batch = 0
    #
    #     for sequence in range(dataloader.batch_size):
    #         stgraph.readGraph([x[sequence]], d, args.distance_thresh)
    #         # Convert to cuda variables
    #         nodes = Variable(torch.from_numpy(nodes).float()).cuda()
    #         edges = Variable(torch.from_numpy(edges).float()).cuda()
    #
    #         obsNodes = Variable(torch.from_numpy(obsNodes).float()).cuda()
    #         obsEdges = Variable(torch.from_numpy(obsEdges).float()).cuda()
    #
    #         # Define hidden states
    #         numNodes = nodes.size()[1]
    #         hidden_states_node_RNNs = Variable(torch.zeros(numNodes, args.human_node_rnn_size)).cuda()
    #         hidden_states_edge_RNNs = Variable(torch.zeros(numNodes * numNodes, args.human_human_edge_rnn_size)).cuda()
    #         cell_states_node_RNNs = Variable(torch.zeros(numNodes, args.human_node_rnn_size)).cuda()
    #         cell_states_edge_RNNs = Variable(torch.zeros(numNodes * numNodes, args.human_human_edge_rnn_size)).cuda()
    #
    #         numObsNodes = obsNodes.size()[1]
    #         hidden_states_obs_node_RNNs = Variable(torch.zeros(numObsNodes, args.obs_node_rnn_size)).cuda()
    #         hidden_states_obs_edge_RNNs = Variable(torch.zeros(numNodes * numNodes, args.human_obstacle_edge_rnn_size)).cuda()
    #
    #         cell_states_obs_node_RNNs = Variable(torch.zeros(numObsNodes, args.obs_node_rnn_size)).cuda()
    #         cell_states_obs_edge_RNNs = Variable(torch.zeros(numNodes * numNodes, args.human_obstacle_edge_rnn_size)).cuda()
    #
    #         outputs,  h_node_rnn, h_edge_rnn, cell_node_rnn, cell_edge_rnn,o_h_node_rnn ,o_h_edge_rnn, o_cell_node_rnn, o_cell_edge_rnn,  _= net(nodes[:args.seq_length], edges[:args.seq_length], nodesPresent[:-1],
    #                                      edgesPresent[:-1], hidden_states_node_RNNs, hidden_states_edge_RNNs,
    #                                      cell_states_node_RNNs, cell_states_edge_RNNs
    #                                      , obsNodes[:args.seq_length], obsEdges[:args.seq_length],
    #                                      obsNodesPresent[:-1], obsEdgesPresent[:-1]
    #                                      ,hidden_states_obs_node_RNNs, hidden_states_obs_edge_RNNs,
    #                                      cell_states_obs_node_RNNs, cell_states_obs_edge_RNNs)
    #
    #         # Compute loss
    #         # loss = Gaussian2DLikelihood(outputs, nodes[1:], nodesPresent[1:], args.pred_length)
    #
    #         loss_batch += loss.data[0]
    #
    #         # Reset the stgraph
    #         # stgraph.reset()
    #
    #     loss_batch = loss_batch / dataloader.batch_size
    #     loss_epoch += loss_batch
    #
    # loss_epoch = loss_epoch / dataloader.valid_num_batches
    #
    # # Update best validation loss until now
    # if loss_epoch < best_val_loss:
    #     best_val_loss = loss_epoch
    #     best_epoch = epoch
    sess.close()


if __name__ == '__main__':
    main()
