import time
from models import g2k_lstm_mcr as mcr
import argparse
import nri_learned as nri
import networkx_graph as nx_g
import load_traj as load
import tensorflow as tf
import helper
import numpy as np
import tensorflow.python.util.deprecation as deprecation
import os
import pandas as pd

# reduce tf messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
deprecation._PRINT_DEPRECATION_WARNINGS = False

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
    parser.add_argument('--seq_length', type=int, default=12,
                        help='RNN sequence length')
    parser.add_argument('--pred_len', type=int, default=12,
                        help='number of layers in the RNN')

    parser.add_argument('--obs_len', type=int, default=8,
                        help='RNN sequence length')

    # Number of epochs parameter
    parser.add_argument('--num_epochs', type=int, default=50,
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

# TODO use glstm to predict path along with neighborhood boundaries using inner estimated soft attention mechanism.
# take relative distance + orthogonality between people vislets (biased by visual span width)
# extract outputs from glstm as follows: neighborhood influence (make message passing between related pedestrians)
# then transform using mlp the new hidden states mixture into (x,y) euclidean locations.

def train(args):

    tf_graph = tf.Graph()
    with tf.Session(graph=tf_graph) as sess:
        with sess.as_default():
            # dataloader = load.DataLoader(args=args, datasets=[0, 1, 2, 3, 4, 5], start=2, sel=0)
            for d in {2,5}: #range(2,5):
                log_count = '/home/serene/PycharmProjects/multimodaltraj_2/log/g2k_lstm_mcr_counts_{0}.txt'.format(d)
                log_count_f = open(log_count, 'w')
                log_dir = '/home/serene/PycharmProjects/multimodaltraj_2/log/g2k_lstm_mcr_error_log_{0}.csv'.format(d)
                log_dir_fde = '/home/serene/PycharmProjects/multimodaltraj_2/log/g2k_lstm_mcr_fde_log_{0}.csv'.format(d)
                log_f = open(log_dir,'w')

                target_traj = []
                true_path = []
                # Train the model
                e = 0
                c_soft = 1
                euc_loss = []
                fde = []
                frame = 1
                num_targets = 0
                num_end_targets = 0

                dataloader = load.DataLoader(args=args, datasets=[0, 1, 2, 3, 4, 5], start=d, sel=0)
                traj = dataloader.load_trajectories(data_file=dataloader.sel_file)

                dim = int(args.neighborhood_size / args.grid_size)
                # TODO implement k-fold cross validation + check why pred_path is all zeros (bug in GridLSTMCell)
                graph = nx_g.online_graph(args)

                print(dataloader.sel_file)
                dataloader.reset_data_pointer()

                while e < args.num_epochs:
                    e_start = time.time()
                    batch, target_traj, _ = dataloader.next_step()

                    if len(batch) == 0:
                        break

                    if e == 0:
                        graph_t = graph.ConstructGraph(current_batch=batch, framenum=frame, future_traj=target_traj)
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
                        tf.initialize_variables(var_list=[weight_i, weight_ii]).run()

                        inputs = tf.convert_to_tensor(batch_v, dtype=tf.float64)
                        inputs = tf.matmul(inputs, weight_i)
                        inputs = tf.matmul(weight_ii, inputs)

                        hidden_state = np.zeros(shape=(args.num_freq_blocks, args.rnn_size))

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
                            dim=args.num_freq_blocks)

                        stat_mask = tf.zeros(shape=(dim, args.num_freq_blocks), dtype=tf.float64)
                        stat_mask += tf.expand_dims(tf.range(start=0, limit=1, delta=0.125, dtype=tf.float64), axis=1)
                        static_mask_nd = stat_mask.eval()

                        vislet = dataloader.vislet[:, frame:frame + num_nodes]
                        vislet_emb = tf.matmul(vislet, weight_i)

                        krnl_mdl = mcr.g2k_lstm_mcr(in_features=nghood_enc.input,
                                                    rel_features=tf.zeros_like(vislet_emb),
                                                    num_nodes=num_nodes, obs_len=args.obs_len,
                                                    hidden_states=hidden_state,
                                                    hidden_size=args.rnn_size,
                                                    lambda_reg=args.lambda_param)

                        # sess.run(fetches=tf.initialize_all_variables())
                        tf.initialize_variables(var_list=[krnl_mdl.weight_r, krnl_mdl.embed_vis, krnl_mdl.weight_v, krnl_mdl.bias_v]).run()
                        tf.initialize_variables(var_list=[krnl_mdl.cost, krnl_mdl.attn, krnl_mdl.weight_c, krnl_mdl.weight_o]).run()
                        # sess.run(fetches=tf.initialize_all_variables())

                        saver = tf.train.Saver(tf.all_variables(), max_to_keep=None)
                        print('session started')

                    for b in range(dataloader.num_batches):# range(2):
                        print('Batch {0} took '.format(b))
                        start_t = time.time()
                        vislet_past = vislet_emb
                        vislet_rel = vislet_past * vislet_emb

                        with tf.variable_scope('nghood_init'):
                            out_init = tf.zeros(dtype=tf.float64,shape=(args.num_freq_blocks, (args.grid_size * (args.grid_size/2))))
                            c_hidden_init = tf.zeros(dtype=tf.float64,shape=(args.num_freq_blocks,(args.grid_size * (args.grid_size/2))))

                        if b > 0 and b % 20 == 0:
                            sess.graph.clear_collection(name='variables')

                        for frame in batch:
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

                            with tf.variable_scope('ngh_stat'):
                                static_mask = tf.placeholder(name='static_mask',  # shape=(dim, static_frame_w),
                                                             dtype=tf.float64)

                                social_frame = tf.placeholder(name='social_frame',  # shape=(static_frame_w,dim),
                                                              dtype=tf.float64)
                                state_f00_b00_c = tf.placeholder(name='state_f00_b00_c',  # shape=(dim,hidden_size),
                                                                 dtype=tf.float64)
                                c_hidden_states = tf.placeholder(name='c_hidden_states',
                                                                 # shape=(dim, (grid_size * (grid_size/2))),
                                                                 dtype=tf.float64)
                                output = tf.placeholder(dtype=tf.float64,
                                                        # shape=[num_nodes, (grid_size * (grid_size / 2))],
                                                        name="output")

                            # compute relative locations and relative vislets
                            st_embeddings, hidden_state, ng_output, c_hidden_state =\
                                sess.run([nghood_enc.input, nghood_enc.state_f00_b00_c,
                                          nghood_enc.output, nghood_enc.c_hidden_state],
                                        feed_dict={nghood_enc.input: inputs.eval(),
                                                   nghood_enc.state_f00_b00_c: hidden_state,
                                                   nghood_enc.output:out_init.eval(),
                                                   nghood_enc.c_hidden_state: c_hidden_init.eval()})

                            # Soft-attention mechanism equipped with static grid
                            static_mask, social_frame =\
                                sess.run([static_mask, output],
                                        feed_dict={static_mask: static_mask_nd,
                                                   social_frame:ng_output,
                                                   state_f00_b00_c: c_soft*hidden_state,
                                                   output: out_init.eval(),
                                                   c_hidden_states: c_hidden_init.eval()
                                        })

                            input = tf.matmul(b=static_mask, a=social_frame).eval()
                            combined_ngh, hidden_state = sess.run([stat_ngh.input, stat_ngh.hidden_state],
                                                                  feed_dict={stat_ngh.input: input,
                                                                             stat_ngh.hidden_state: hidden_state})

                            reg_ng = np.transpose(args.lambda_param * np.transpose(ng_output))
                            pred_path, hidden_state, prob_mat = \
                                sess.run([krnl_mdl.pred_path_band, krnl_mdl.hidden_states, krnl_mdl.cost],
                                         feed_dict={
                                             krnl_mdl.outputs: np.concatenate((st_embeddings, vislet_emb.eval()), axis=0),
                                             krnl_mdl.ngh: reg_ng,
                                             krnl_mdl.rel_features: vislet_rel.eval(),
                                             krnl_mdl.hidden_states: hidden_state,
                                             # krnl_mdl.lambda_reg: args.lambda_reg,
                                             krnl_mdl.pred_path_band: tf.random_normal(shape=(2, 12, num_nodes)).eval()})

                            # for (i, idx) in zip(krnl_mdl.cost.eval(), range(len(hidden_state))):
                            attn = tf.exp(krnl_mdl.attn) / tf.cumsum(tf.exp(krnl_mdl.attn))

                            # update with weighted hidden states
                            krnl_mdl.hidden_states = tf.nn.softmax(krnl_mdl.hidden_states)  #weigh hidden states then send them back to glstm on next step
                            hidden_state = krnl_mdl.hidden_states.eval()
                            c_soft = np.sum(np.matmul(attn.eval(), hidden_state))
                            #tf.tanh(np.sum(np.matmul(attn.eval(), hidden_state)))
                            # pred_path = np.transpose(krnl_mdl.pred_path_band.eval(), (2,1,0))
                            pred_path = np.transpose(pred_path, (2, 1, 0))
                            num_targets += num_nodes
                            for i in range(1,num_nodes):
                                try:
                                    num_end_targets += 1
                                    if len(target_traj[i]) < args.pred_len:
                                        krnl_mdl.pred_path_band.eval()
                                        euc_loss.append(pred_path[i][0:len(target_traj[i])] - target_traj[i])  # , ord=2) / len(target_traj)
                                        fde.append(pred_path[i][len(target_traj[i])-1] - target_traj[i][len(target_traj[i])-1])
                                        # euc_loss = np.linalg.norm((pred_path[i][0:len(target_traj[i])] - target_traj[i]), ord=2)/len(target_traj)
                                    else:
                                        euc_loss.append(pred_path[i][0:args.pred_len] - target_traj[i][0:args.pred_len])
                                        fde.append(pred_path[i][args.pred_len-1]- target_traj[i][args.pred_len-1])
                                        # np.linalg.norm((pred_path[i][0:len(target_traj[i])] - target_traj[i]), ord=2)

                                    print('ADE= ',np.linalg.norm(euc_loss[len(euc_loss) - 1], ord=2) / (num_nodes * 12), \
                                          ' FDE= ', np.linalg.norm(fde[len(fde) - 1], ord=2) / num_nodes)

                                except KeyError:
                                    i += 1
                                    continue

                        batch, target_traj, _ = dataloader.next_step()

                        if len(batch) == 0:
                            break

                        graph_t = graph.ConstructGraph(current_batch=batch, framenum=frame, future_traj=target_traj)
                        batch_v = list(graph_t.get_node_attr(param='node_pos_list').values())
                        batch_v = np.transpose(batch_v)
                        num_nodes = batch_v.shape[1]

                        vislet = dataloader.vislet[:, frame:frame + num_nodes]
                        with tf.variable_scope('weight_input'):
                            init_w = tf.initializers.random_normal(mean=0, stddev=1, seed=0, dtype=tf.float64)
                            weight_i = tf.Variable(name='weight_i',
                                                   initial_value=init_w(shape=(vislet.shape[1], args.num_freq_blocks)),
                                                   trainable=True, dtype=tf.float64)

                            weight_ii = tf.Variable(name='weight_ii',
                                                    initial_value=init_w(shape=(args.num_freq_blocks, args.input_size)),
                                                    trainable=True, dtype=tf.float64)

                        tf.initialize_variables(var_list=[weight_i, weight_ii]).run()

                        inputs = tf.convert_to_tensor(batch_v, dtype=tf.float64)
                        inputs = tf.matmul(inputs, weight_i)
                        inputs = tf.matmul(weight_ii, inputs)

                        vislet_emb = tf.matmul(vislet, weight_i)
                        vislet_rel = vislet_past * vislet_emb

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
                            dim=args.num_freq_blocks)

                        stat_mask = tf.zeros(shape=(dim, args.num_freq_blocks), dtype=tf.float64)
                        stat_mask += tf.expand_dims(tf.range(start=0, limit=1, delta=0.125, dtype=tf.float64), axis=1)
                        static_mask_nd = stat_mask.eval()

                        krnl_mdl = mcr.g2k_lstm_mcr(in_features=nghood_enc.input,
                                                    rel_features=vislet_rel,
                                                    num_nodes=num_nodes, obs_len=args.obs_len,
                                                    hidden_states= hidden_state,
                                                    hidden_size=args.rnn_size,
                                                    lambda_reg=args.lambda_param)

                        tf.initialize_variables(
                            var_list=[krnl_mdl.weight_r, krnl_mdl.embed_vis, krnl_mdl.weight_v, krnl_mdl.bias_v]).run()
                        tf.initialize_variables(
                            var_list=[krnl_mdl.cost, krnl_mdl.attn, krnl_mdl.weight_c, krnl_mdl.weight_o]).run()

                        # sess.run(fetches=tf.initialize_all_variables())

                        end_t = time.time()
                        print('{0} seconds to complete'.format(end_t - start_t))
                        print('Frame {3} Batch {0} of {1}, Loss = {2}, num_ped={4}'
                              .format(b, dataloader.num_batches, krnl_mdl.cost, frame, len(target_traj)))

                    # make another model file with attn
                    fde = np.stack(fde)
                    euc_loss = np.ravel(euc_loss)
                    np.savetxt(fname=log_dir_fde, X=fde, delimiter=",")
                    np.savetxt(fname=log_dir, X=euc_loss, delimiter=",")

                    # log_f.write('{0}'.format(euc_loss))
                    if (e * dataloader.num_batches + b) % args.save_every == 0 and ((e * dataloader.num_batches + b) > 0):
                        checkpoint_path = os.path.join('/home/serene/PycharmProjects/multimodaltraj_2/save', 'g2k_mcr_model_val_{0}.ckpt'.format(b))
                        saver.save(sess, checkpoint_path, global_step=e * dataloader.num_batches + b)
                        print("model saved to {}".format(checkpoint_path))

                    e_end = time.time()
                    print('Epoch time taken: ', (e_end - e_start))
                log_count_f.write('Dataset {0}= ADE steps {1}\nFDE steps = {2}'.format(d,num_targets,num_end_targets))
                # log_f.close()
            log_count_f.close()

            #*************************************************************** VALIDATION *************************************
            # Validate
            # dataloader.reset_data_pointer(valid=True)
            loss_epoch = 0
            dataloader.valid_frame_pointer = frame + (frame * e)
            dataloader.valid_num_batches = int((dataloader.valid_frame_pointer / dataloader.seq_length) / dataloader.batch_size)
            for vb in range(dataloader.valid_num_batches):
                # Get batch data
                # x, _, d = dataloader.load_trajectories(data_file='')  ## stateless lstm without shuffling
                rang = range(frame, int(frame + (dataloader.batch_size * args.seq_length)), args.seq_length)
                val_traj = [{idx:traj[idx]} for idx in rang]
                # Loss for this batch
                loss_batch = 0
                # for batch in traj:
                print('Validation Batch {0} took '.format(vb))
                start_t = time.time()

                with tf.variable_scope('weight_input'):
                    init_w = tf.initializers.random_normal(mean=0, stddev=1, seed=0, dtype=tf.float64)
                    weight_i = tf.Variable(name='weight_i',
                                           initial_value=init_w(shape=(num_nodes, args.num_freq_blocks)),
                                           trainable=True, dtype=tf.float64)
                    weight_ii = tf.Variable(name='weight_ii',
                                            initial_value=init_w(shape=(args.num_freq_blocks, args.input_size)),
                                            trainable=True, dtype=tf.float64)

                with tf.variable_scope('nghood_init'):
                    out_init = tf.zeros(dtype=tf.float64,
                                        shape=(args.num_freq_blocks, (args.grid_size * (args.grid_size / 2))))
                    c_hidden_init = tf.zeros(dtype=tf.float64, shape=(
                    args.num_freq_blocks, (args.grid_size * (args.grid_size / 2))))

                tf.initialize_variables(var_list=[weight_i, weight_ii]).run()

                for frame_iter in iter(val_traj):
                    # check if get_node_attr gets complete sequence for all nodes
                    # num_nodes x obs_length
                    (frame_iter, _), = frame_iter.items()
                    try:
                        true_path.append(batch[frame_iter])
                    except KeyError:
                        if frame == max(batch.keys()):
                            break
                        if frame_iter + args.seq_length + 1 > max(batch.keys()):
                            frame_iter = max(batch.keys())
                        else:
                            frame_iter += args.seq_length + 1
                        continue

                    with tf.variable_scope('ngh_stat'):
                        static_mask = tf.placeholder(name='static_mask',  # shape=(dim, static_frame_w),
                                                     dtype=tf.float64)

                        social_frame = tf.placeholder(name='social_frame',  # shape=(static_frame_w,dim),
                                                      dtype=tf.float64)
                        state_f00_b00_c = tf.placeholder(name='state_f00_b00_c',  # shape=(dim,hidden_size),
                                                         dtype=tf.float64)
                        c_hidden_states = tf.placeholder(name='c_hidden_states',
                                                         # shape=(dim, (grid_size * (grid_size/2))),
                                                         dtype=tf.float64)

                        output = tf.placeholder(dtype=tf.float64,
                                                # shape=[num_nodes, (grid_size * (grid_size / 2))],
                                                name="output")

                    st_embeddings, hidden_state, ng_output, c_hidden_state = \
                        sess.run([nghood_enc.input, nghood_enc.state_f00_b00_c,
                                  nghood_enc.output, nghood_enc.c_hidden_state],
                                 feed_dict={nghood_enc.input: inputs.eval(),
                                            nghood_enc.state_f00_b00_c: hidden_state,
                                            nghood_enc.output: out_init.eval(),
                                            nghood_enc.c_hidden_state: c_hidden_init.eval()})

                    static_mask, social_frame =\
                        sess.run([static_mask, output],
                                 feed_dict={static_mask: static_mask_nd,
                                            social_frame: ng_output,
                                            state_f00_b00_c: hidden_state,
                                            output: out_init.eval(),
                                            c_hidden_states: c_hidden_init.eval()
                                            })

                    input = tf.matmul(b=static_mask,
                                      a=social_frame).eval()  # Soft-attention mechanism equipped with static grid
                    combined_ngh, hidden_state = sess.run([stat_ngh.input, stat_ngh.hidden_state],
                                                          feed_dict={stat_ngh.input: input,
                                                                     stat_ngh.hidden_state: hidden_state})
                    pred_path, jacobian = \
                        sess.run([krnl_mdl.pred_path_band, krnl_mdl.cost],
                                 feed_dict={
                                     krnl_mdl.outputs: np.concatenate((st_embeddings, vislet_emb.eval()),
                                                                      axis=0),
                                     krnl_mdl.ngh: np.transpose(args.lambda_param * np.transpose(ng_output)),
                                     # krnl_mdl.lambda_reg: args.lambda_reg,
                                     krnl_mdl.pred_path_band: tf.random_normal(shape=(2, 12, num_nodes)).eval()})

                    # adj_mat = nri.eval_rln_ngh(jacobian, combined_ngh)

                    pred_path = np.transpose(pred_path, (2, 1, 0))
                    for i in range(1, num_nodes):
                        try:
                            if len(target_traj[i]) < args.pred_len:
                                euc_loss = np.linalg.norm(
                                    (pred_path[i][0:len(target_traj[i])] - target_traj[i]), ord=2) / len(
                                    target_traj)
                            else:
                                euc_loss = np.linalg.norm(
                                    (pred_path[i][0:args.pred_len] - target_traj[i][0:args.pred_len]),
                                    ord=2) / len(target_traj)
                                # np.linalg.norm((pred_path[i][0:len(target_traj[i])] - target_traj[i]), ord=2)
                        except KeyError:
                            i += 1
                            continue

                end_t = time.time()
                print('{0} seconds to complete'.format(end_t - start_t))
                print('Frame {3} Batch {0} of {1}, Loss = {2}, num_ped={4}'
                      .format(b, dataloader.num_batches, krnl_mdl.cost, frame, len(target_traj)))

                batch, target_traj, _ = dataloader.next_step()

                graph_t = graph.ConstructGraph(current_batch=batch, framenum=frame, future_traj=target_traj)
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
                tf.initialize_variables(var_list=[weight_i, weight_ii]).run()

                inputs = tf.convert_to_tensor(batch_v, dtype=tf.float64)
                inputs = tf.matmul(inputs, weight_i)
                inputs = tf.matmul(weight_ii, inputs)

                vislet = dataloader.vislet[:, frame:frame + num_nodes]  # tf.expand_dims(batch_v[0], axis=0)
                vislet_emb = tf.matmul(vislet, weight_i)

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
                    dim=args.num_freq_blocks)
                # num_nodes=args.num_freq_blocks,
                # dropout=args.dropout)

                stat_mask = tf.zeros(shape=(dim, args.num_freq_blocks), dtype=tf.float64)
                stat_mask += tf.expand_dims(tf.range(start=0, limit=1, delta=0.125, dtype=tf.float64), axis=1)
                static_mask_nd = stat_mask.eval()

                krnl_mdl = mcr.g2k_lstm_mcr(in_features=nghood_enc.input, out_size=batch_v.shape[1],
                                            num_nodes=num_nodes, obs_len=args.obs_len,
                                            lambda_reg=args.lambda_param)

                checkpoint_path = os.path.join('/home/serene/PycharmProjects/multimodaltraj_2/log/save',
                                               'g2k_mc_model_val_{0}.ckpt'.format(vb))
                saver.save(sess, checkpoint_path, global_step=e * dataloader.num_batches + b)
                print("model saved to {}".format(checkpoint_path))

    sess.close()


if __name__ == '__main__':
    main()
