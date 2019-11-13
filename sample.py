import pickle
from models import g2k_lstm_mcr as mcr
import argparse
import nri_learned as nri
import networkx_graph as nx_g
import argParser as argsParser
import load_traj as load
import tensorflow as tf
import helper
import numpy as np
import tensorflow.python.util.deprecation as deprecation
import os
import scipy.io
import tensorflow.python.tools.inspect_checkpoint as ckpt_inspkt


# from social_train import getSocialGrid, getSocialTensor


def get_mean_error(predicted_traj, true_traj, observed_length, maxNumPeds):
    '''
    Function that computes the mean euclidean distance error between the
    predicted and the true trajectory
    params:
    predicted_traj : numpy matrix with the points of the predicted trajectory
    true_traj : numpy matrix with the points of the true trajectory
    observed_length : The length of trajectory observed
    '''
    # The data structure to store all errors
    error = np.zeros(len(true_traj) - observed_length)
    # For each point in the predicted part of the trajectory
    for i in range(observed_length, len(true_traj)):
        # The predicted position. This will be a maxNumPeds x 3 matrix
        pred_pos = predicted_traj[i, :]
        # The true position. This will be a maxNumPeds x 3 matrix
        true_pos = true_traj[i, :]
        timestep_error = 0
        counter = 0
        for j in range(maxNumPeds):
            if true_pos[j, 0] == 0:
                # Non-existent ped
                continue
            elif pred_pos[j, 0] == 0:
                # Ped comes in the prediction time. Not seen in observed part
                continue
            else:
                if true_pos[j, 1] > 1 or true_pos[j, 1] < 0:
                    continue
                elif true_pos[j, 2] > 1 or true_pos[j, 2] < 0:
                    continue

                timestep_error += np.linalg.norm(true_pos[j, [1, 2]] - pred_pos[j, [1, 2]])
                counter += 1

        if counter != 0:
            error[i - observed_length] = timestep_error / counter

        # The euclidean distance is the error
        # error[i-observed_length] = np.linalg.norm(true_pos - pred_pos)

    # Return the mean error
    return np.mean(error)


def main():
    # Set random seed
    np.random.seed(1)

    parser = argparse.ArgumentParser()
    args = argsParser.ArgsParser().parser.parse_args()
    # Observed length of the trajectory parameter
    parser.add_argument('--obs_length', type=int, default=8,
                        help='Observed length of the trajectory')
    # Predicted length of the trajectory parameter
    parser.add_argument('--pred_length', type=int, default=12,
                        help='Predicted length of the trajectory')
    # Test dataset
    parser.add_argument('--test_dataset', type=int, default=2,
                        help='Dataset to be tested on')

    # Model to be loaded
    parser.add_argument('--epoch', type=int, default=1,
                        help='Epoch of model to be loaded')

    # Parse the parameters
    sample_args = parser.parse_args()

    # Save directory
    save_directory = '/home/siri0005/Documents/multimodaltraj_2/save'

    # Define the path for the config file for saved args
    # with open(os.path.join(save_directory, 'social_config.pkl'), 'rb') as f:
    #     saved_args = pickle.load(f)
    # Initialize a TensorFlow session
    sess = tf.InteractiveSession()

    # Dataset to get data from
    dataset = [sample_args.test_dataset]

    # Create a SocialDataLoader object with batch_size 1 and seq_l8ength equal to observed_length + pred_length
    data_loader = load.DataLoader(args=args, datasets=[0, 1, 2, 3, 4, 5], start=sample_args.test_dataset, sel=0)
    traj = data_loader.load_trajectories(data_file=data_loader.sel_file)

    # Reset all pointers of the data_loadert
    data_loader.reset_data_pointer()

    c_soft = 1
    frame = 1
    results = []
    vislet_emb = None
    dim = int(args.neighborhood_size / args.grid_size)
    # Variable to maintain total error
    total_error = 0

    # For each batch
    for b in range(data_loader.num_batches):
        # Get the source, target and dataset data for the next batch
        x_batch, y_batch, _ = data_loader.next_step()
        if vislet_emb is not None:
            vislet_past = vislet_emb
        else:
            vislet_past = 1

        graph = nx_g.online_graph(args)
        graph_t = graph.ConstructGraph(current_batch=x_batch, framenum=frame, future_traj=y_batch)

        batch_v = list(graph_t.get_node_attr(param='node_pos_list').values())
        batch_v = np.transpose(batch_v)

        # Batch size is 1
        # x_batch, y_batch, d_batch, x_batch_grid, ped_fr_batch = x[0], y[0], d[0], x_grid[0], fr_ped[0]
        num_nodes = batch_v.shape[1]

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

        vislet = data_loader.vislet[:, frame:frame + num_nodes]
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

        # Create a SocialModel object with the saved_args and infer set to true
        hidden_state = np.zeros(shape=(args.num_freq_blocks, args.rnn_size))
        krnl_mdl = mcr.g2k_lstm_mcr(in_features=nghood_enc.input,
                                    rel_features=tf.zeros_like(vislet_emb),
                                    num_nodes=num_nodes, obs_len=args.obs_len,
                                    hidden_states=hidden_state,
                                    hidden_size=args.rnn_size,
                                    lambda_reg=args.lambda_param)

        if b == 0:
            # Initialize a saver
            # saver = tf.train.Saver(tf.all_variables())
            # Get the checkpoint state for the model
            ckpt = tf.train.get_checkpoint_state(checkpoint_dir=save_directory, latest_filename='checkpoint')
            print('Importing meta data of stored model in' + ckpt.model_checkpoint_path+'.meta')
            saver = tf.train.import_meta_graph(os.path.join(save_directory, ckpt.model_checkpoint_path+'.meta'))
            # print ('loading model: ', ckpt.model_checkpoint_path)

            print('loading model: ', ckpt.all_model_checkpoint_paths)

            # Restore the model at the checkpoint
            saver.restore(sess, ckpt.model_checkpoint_path)

        static_mask_nd = stat_mask.eval()

        with tf.variable_scope('nghood_init'):
            out_init = tf.zeros(dtype=tf.float64, shape=(args.num_freq_blocks, (args.grid_size * (args.grid_size / 2))))
            c_hidden_init = tf.zeros(dtype=tf.float64,
                                     shape=(args.num_freq_blocks, (args.grid_size * (args.grid_size / 2))))

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

        # dimensions = [720, 576]
        # grid_batch = getSequenceGridMask(x_batch_grid, dimensions, saved_args.neighborhood_size, saved_args.grid_size)
        # obs_traj = x_batch[:sample_args.obs_length]
        # obs_grid = grid_batch[:sample_args.obs_length]
        # obs_traj is an array of shape obs_length x maxNumPeds x 3

        print("********************** SAMPLING A NEW TRAJECTORY", b, "******************************")

        # compute relative locations and relative vislets
        st_embeddings, hidden_state, ng_output, c_hidden_state = \
            sess.run([nghood_enc.input, nghood_enc.state_f00_b00_c,
                      nghood_enc.output, nghood_enc.c_hidden_state],
                     feed_dict={nghood_enc.input: inputs.eval(),
                                nghood_enc.state_f00_b00_c: hidden_state,
                                nghood_enc.output: out_init.eval(),
                                nghood_enc.c_hidden_state: c_hidden_init.eval()})

        # Soft-attention mechanism equipped with static grid
        static_mask, social_frame = \
            sess.run([static_mask, output],
                     feed_dict={static_mask: static_mask_nd,
                                social_frame: ng_output,
                                state_f00_b00_c: c_soft * hidden_state,
                                output: out_init.eval(),
                                c_hidden_states: c_hidden_init.eval()
                                })

        input = tf.matmul(b=static_mask, a=social_frame).eval()
        combined_ngh, hidden_state = sess.run([stat_ngh.input, stat_ngh.hidden_state],
                                              feed_dict={stat_ngh.input: input,
                                                         stat_ngh.hidden_state: hidden_state})

        reg_ng = np.transpose(args.lambda_param * np.transpose(ng_output))

        tf.initialize_variables(var_list=[krnl_mdl.weight_r, krnl_mdl.embed_vis, krnl_mdl.weight_v, krnl_mdl.bias_v]).run()
        tf.initialize_variables(var_list=[krnl_mdl.cost, krnl_mdl.attn, krnl_mdl.weight_c, krnl_mdl.weight_o]).run()

        complete_traj, hidden_state, prob_mat = \
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
        krnl_mdl.hidden_states = tf.nn.softmax(
            krnl_mdl.hidden_states)  # weigh hidden states then send them back to glstm on next step
        hidden_state = krnl_mdl.hidden_states.eval()
        c_soft = np.sum(np.matmul(attn.eval(), hidden_state))

        complete_traj = np.transpose(complete_traj, (2, 1, 0))

        # complete_traj is an array of shape (obs_length+pred_length) x maxNumPeds x 3
        y_batch = np.stack(list(graph_t.get_node_attr(param='targets').values())).squeeze()
        total_error += get_mean_error(complete_traj, y_batch, sample_args.obs_length, maxNumPeds=num_nodes)

        print("Processed trajectory number : ", b, "out of ", data_loader.num_batches, " trajectories")

        # plot_trajectories(x[0], complete_traj, sample_args.obs_length)
        # return
        results.append((x_batch, complete_traj, sample_args.obs_length))

    # Print the mean error across all the batches
    print("Total mean error of the model is ", total_error / data_loader.num_batches)

    print("Saving results")
    with open(os.path.join(save_directory, 'social_results.pkl'), 'wb') as f:
        pickle.dump(results, f)
    path = 'save/'
    filename = path + 'social_results.pkl'
    f = open(filename, 'rb')
    results = pickle.load(f)
    print("Saving results as .mat")
    filesave = 'VisualizeUtils/' + 'tc_real_head_pose_z1_z2_ucy_social_lstm.mat'
    scipy.io.savemat(filesave, mdict={'data': results})


if __name__ == '__main__':
    main()
