import argParser as parser
import time
from models import g2k_lstm_mcr as mcr
from matplotlib.pyplot import imread
import networkx_graph as nx_g
import load_traj as load
import tensorflow as tf
import helper
import numpy as np
import os
import glob

# reduce tf messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# deprecation._PRINT_DEPRECATION_WARNINGS = False

def main():
    args = parser.ArgsParser()
    train(args.parser.parse_args())

    return

def train(args):
    tf_graph = tf.Graph()
    out_graph = tf.Graph()
    with tf.Session(graph=out_graph).as_default() as out_sess:

        for l in {args.leaveDataset}:
            e = 0
            true_path = []
            euc_loss = []
            fde = []

            frame = 1
            num_targets = 0
            num_end_targets = 0

            datasets = {2, 3, 4, 5}
            datasets.remove(l)  # leave dataset
            parent_dir = '/fakepath/multimodaltraj_2/'
            for d in datasets:  # range(2,5):
                log_count = parent_dir+'log/g2k_lstm_counts_{0}.txt'.format(
                    d)
                log_count_f = open(log_count, 'w')
                log_dir = parent_dir+'g2k_MPC_error_log_kfold_{0}.csv'.format(
                    d)
                log_dir_fde = parent_dir+'g2k_MPC_fde_log_kfold_{0}.csv'.format(
                    d)

                dataloader = load.DataLoader(args=args, datasets=[0, 1, 2, 3, 4, 5], start=d, sel=0)

                # TODO implement k-fold cross validation + check why pred_path is all zeros (bug in GridLSTMCell)
                graph = nx_g.online_graph(args)

                print(dataloader.sel_file)
                dataloader.reset_data_pointer()
                flag = True

                while e < args.num_epochs:
                    e_start = time.time()
                    batch, target_traj, _ = dataloader.next_step()

                    if len(batch) == 0:
                        if d == max(datasets):
                            dataloader.dataset_pointer = min(datasets)
                        dataloader.reset_data_pointer()
                        continue

                    print('session started')
                    # with tf_graph.as_default():
                    for b in range(dataloader.num_batches):  #
                        with tf.Session(graph=out_graph) as sess:
                        # if e == 0:
                            graph_t = graph.ConstructGraph(current_batch=batch, framenum=frame,
                                                           future_traj=target_traj)
                            batch_v = list(graph_t.get_node_attr(param='node_pos_list').values())
                            if len(np.array(batch_v).shape) > 1:
                                batch_v = np.array(batch_v)[frame:frame + args.obs_len]
                                batch_v = np.linalg.norm(batch_v, axis=2).squeeze()
                            else:
                                # batch_v = np.array(batch_v)[frame:frame + args.obs_len]
                                dataloader.reset_data_pointer()
                                break

                            batch_v = np.transpose(batch_v)
                            try:
                                num_nodes = batch_v.shape[1]
                            except IndexError:
                                dataloader.reset_data_pointer()
                                break

                            if e == 0 and flag:
                                dim = int(args.neighborhood_size / args.grid_size)
                                ctxt_img_path = glob.glob(dataloader.current_dir + 'ctxt.png')
                                ctxt_img = tf.convert_to_tensor(imread(ctxt_img_path[0]), dtype=tf.float64)

                                ctxt_img_pd = tf.convert_to_tensor(
                                    tf.pad(ctxt_img, paddings=tf.constant([[1, 1, ], [0, 1], [0, 0]])),
                                    dtype=tf.float64)
                                width = int(ctxt_img_pd.shape.dims[0])
                                height = int(ctxt_img_pd.shape.dims[1])

                                ctxt_img_pd = tf.expand_dims(ctxt_img_pd, axis=0)
                                _2dconv = tf.nn.conv2d(input=ctxt_img_pd,
                                                       filter=tf.random_normal(shape=[width-dim+1, height-dim+1, 3, 1],
                                                                               dtype=tf.float64),
                                                       padding='VALID', strides=[1, 1, 1, 1])

                                _2dconv = tf.squeeze(_2dconv).eval()
                                _2dconv = args.lambda_param * _2dconv
                                # _2dconv /= np.max(_2dconv) - 1
                                hidden_state = np.zeros(shape=(dim, args.rnn_size))
                                flag = False

                            if b == 0:
                                init_w = tf.initializers.random_normal(mean=0, stddev=1, seed=0,
                                                                       dtype=tf.float64)

                                nghood_enc = helper.neighborhood_vis_loc_encoder(
                                    hidden_size=args.rnn_size,
                                    hidden_len=dim,
                                    num_layers=args.num_layers,
                                    grid_size=args.grid_size,
                                    embedding_size=args.embedding_size,
                                    dropout=args.dropout)

                                stat_ngh = helper.neighborhood_stat_enc(
                                    hidden_size=args.rnn_size,
                                    num_layers=args.num_layers,
                                    grid_size=args.grid_size,
                                    dim=dim,
                                    ctxt_path=ctxt_img_path)

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

                                with tf.variable_scope('nghood_init', reuse=True):
                                    out_init = tf.zeros(dtype=tf.float64, shape=(
                                        dim, dim)) #(args.grid_size * (args.grid_size / 2))))
                                    c_hidden_init = tf.zeros(dtype=tf.float64, shape=(
                                        dim, dim)) # (args.grid_size * (args.grid_size / 2))))

                                stat_mask = tf.zeros(shape=(dim, args.obs_len), dtype=tf.float64)
                                stat_mask += tf.expand_dims(tf.range(start=0, limit=1, delta=(1/args.obs_len), dtype=tf.float64), axis=0)
                                static_mask_nd = stat_mask.eval()

                                _2dconv_in = tf.matmul(_2dconv, stat_mask)
                                # _2dconv_valin = _2dconv_in.eval()

                            krnl_mdl = mcr.g2k_lstm_mcr(in_features=nghood_enc.input,  # MC.g2k_lstm_mc
                                                        num_nodes=num_nodes, obs_len=args.obs_len,
                                                        hidden_size=args.rnn_size,
                                                        lambda_reg=args.lambda_param,
                                                        sess_g=out_graph)

                            with tf.variable_scope('weight_input',reuse=True):
                                weight_i = tf.Variable(name='weight_i',
                                                       initial_value=init_w(
                                                           shape=(num_nodes, dim)),
                                                       trainable=True, dtype=tf.float64)
                                weight_ii = tf.Variable(name='weight_ii',
                                                        initial_value=init_w(
                                                            shape=(dim, args.obs_len)),
                                                        trainable=True, dtype=tf.float64)
                            tf.initialize_variables(var_list=[weight_i, weight_ii]).run()

                            inputs = tf.convert_to_tensor(batch_v, dtype=tf.float64)
                            inputs = tf.matmul(inputs, weight_i)
                            inputs = tf.matmul(weight_ii, inputs)

                            vislet = dataloader.vislet[:, frame:frame + num_nodes]
                            vislet_emb = tf.matmul(vislet, weight_i)

                            tf.initialize_variables(
                                var_list=[krnl_mdl.weight_r, krnl_mdl.weight_v, krnl_mdl.bias_v]).run()  # , krnl_mdl.embed_vis
                            tf.initialize_variables(var_list=[krnl_mdl.cost, krnl_mdl.attn, krnl_mdl.weight_c,
                                                              krnl_mdl.weight_o]).run() # krnl_mdl.attn,
                            # sess.run(fetches=tf.initialize_all_variables())

                            print('Batch {0} took '.format(b))

                            start_t = time.time()
                            vislet_past = vislet_emb
                            vislet_rel = vislet_past * vislet_emb

                            for frame in batch:
                                true_path.append(batch[frame])

                                # compute relative locations and relative vislets
                                st_embeddings, hidden_state, ng_output, c_hidden_state = \
                                    sess.run([nghood_enc.input, nghood_enc.state_f00_b00_c,
                                              nghood_enc.output, nghood_enc.c_hidden_state],
                                             feed_dict={nghood_enc.input: inputs.eval(),
                                                        nghood_enc.state_f00_b00_c: hidden_state,
                                                        nghood_enc.output: out_init.eval(),
                                                        nghood_enc.c_hidden_state: c_hidden_init.eval()})

                                # Soft-attention mechanism equipped with static grid
                                static_mask_np, social_frame_np =\
                                    sess.run([static_mask, output],
                                            feed_dict={static_mask: static_mask_nd,
                                                       social_frame:ng_output,
                                                       state_f00_b00_c: hidden_state,# or multiply c_soft directly with static_frame
                                                       output: out_init.eval(),
                                                       c_hidden_states: c_hidden_init.eval()
                                            })

                                # input = tf.matmul(b=static_mask_np, a=social_frame_np).eval()
                                combined_ngh, hidden_state = sess.run([stat_ngh.input, stat_ngh.hidden_state],
                                                                      feed_dict={stat_ngh.input: _2dconv_in.eval(),
                                                                                 stat_ngh.hidden_state: hidden_state})

                                # reg_ng = np.transpose()
                                pred_path = tf.random_normal(shape=(2, 12, num_nodes))
                                pred_path, hidden_state, prob_mat = \
                                    sess.run([krnl_mdl.pred_path_band, krnl_mdl.hidden_states, krnl_mdl.cost],
                                             # krnl_mdl.hidden_states,
                                             feed_dict={
                                                 krnl_mdl.outputs: #st_embeddings,
                                                 np.concatenate((st_embeddings, vislet_emb.eval()), axis=0),
                                                 krnl_mdl.ngh: _2dconv_in.eval(),
                                                 krnl_mdl.rel_features: vislet_rel.eval(),
                                                 krnl_mdl.hidden_states: hidden_state,
                                                 krnl_mdl.out_size: num_nodes,
                                                 # krnl_mdl.lambda_reg: args.lambda_reg,
                                                 krnl_mdl.pred_path_band: pred_path.eval()
                                             })

                                attn = tf.nn.softmax(tf.exp(krnl_mdl.attn) / tf.cumsum(tf.exp(krnl_mdl.attn)))

                                # weigh hidden states then send them back to glstm on next step
                                krnl_mdl.hidden_states = tf.nn.softmax(krnl_mdl.hidden_states)
                                hidden_state = krnl_mdl.hidden_states.eval()
                                # # weightage of final hidden states resulting from chaining of hidden states through social (dynamic) neighborhood then weighted static neighborhood
                                # # then softmaxing hidden states
                                hidden_state = np.matmul(attn.eval(), hidden_state)
                                adj_mat = tf.matmul(tf.nn.softmax(hidden_state),
                                                    tf.ones(shape=(hidden_state.shape[1], 1), dtype=tf.float64))

                                # # GG-NN 2016 A_nx2n , we use A_nxn; n is |G_v| cardinality of node v in Graph G.
                                hidden_state = adj_mat.eval() * hidden_state

                                pred_path = np.transpose(pred_path, (2, 1, 0))
                                num_targets += num_nodes

                                for i, itr in zip(range(1, num_nodes), iter(target_traj)):
                                    try:
                                        num_end_targets += 1
                                        if len(target_traj[i]) < args.pred_len:
                                            krnl_mdl.pred_path_band.eval()
                                            euc_loss.append(pred_path[i][0:len(target_traj[itr])] - target_traj[
                                                itr])  # , ord=2) / len(target_traj)
                                            fde.append(pred_path[i][len(target_traj[itr]) - 1] - target_traj[itr][
                                                len(target_traj[i]) - 1])
                                            # euc_loss = np.linalg.norm((pred_path[i][0:len(target_traj[i])] - target_traj[i]), ord=2)/len(target_traj)
                                        else:
                                            euc_loss.append(
                                                pred_path[i][0:args.pred_len] - target_traj[itr][0:args.pred_len])
                                            fde.append(pred_path[i][args.pred_len - 1] - target_traj[itr][
                                                args.pred_len - 1])
                                            # np.linalg.norm((pred_path[i][0:len(target_traj[i])] - target_traj[i]), ord=2)

                                    except KeyError:
                                        i += 1
                                        continue

                            batch, target_traj, _ = dataloader.next_step()

                            graph_t = graph.ConstructGraph(current_batch=batch, framenum=frame,
                                                           future_traj=target_traj)
                            batch_v = list(graph_t.get_node_attr(param='node_pos_list').values())

                            if len(batch_v) == 0:
                                break
                            if len(np.array(batch_v).shape) > 1:
                                batch_v = np.array(batch_v)[frame:frame + args.obs_len]
                                batch_v = np.linalg.norm(batch_v, axis=2).squeeze()
                            else:
                                # batch_v = np.array(batch_v)[frame:frame + args.obs_len]
                                dataloader.reset_data_pointer()
                                break

                            batch_v = np.transpose(batch_v)
                            try:
                                num_nodes = batch_v.shape[1]
                            except IndexError:
                                dataloader.reset_data_pointer()
                                break

                            # vislet = dataloader.vislet[:, frame:frame + num_nodes]
                            with tf.variable_scope('weight_input', reuse=True):
                                # init_w = tf.initializers.random_normal(mean=0, stddev=1, seed=0, dtype=tf.float64)
                                weight_vi = tf.Variable(name='weight_vi',
                                                        initial_value=init_w(
                                                            shape=(vislet.shape[1], args.num_freq_blocks)),
                                                        trainable=True, dtype=tf.float64)
                                weight_i = tf.Variable(name='weight_i',
                                                       initial_value=init_w(
                                                           shape=(batch_v.shape[1], dim)),
                                                       trainable=True, dtype=tf.float64)

                                weight_ii = tf.Variable(name='weight_ii',
                                                        initial_value=init_w(
                                                            shape=(dim, args.obs_len)),
                                                        trainable=True, dtype=tf.float64)

                            tf.initialize_variables(var_list=[weight_vi, weight_i, weight_ii]).run()

                            tf.initialize_variables(
                                var_list=[krnl_mdl.weight_v,krnl_mdl.weight_r, krnl_mdl.bias_v]).run()  # , krnl_mdl.embed_vis
                            tf.initialize_variables(
                                var_list=[krnl_mdl.cost, krnl_mdl.attn, krnl_mdl.weight_c,
                                          krnl_mdl.weight_o]).run()

                            end_t = time.time()
                            print('{0} seconds to complete'.format(end_t - start_t))
                            print('Frame {3} Batch {0} of {1}, Loss = {2}, num_ped={4}'
                                  .format(b, dataloader.num_batches, krnl_mdl.cost, frame, len(target_traj)))
                            if (e * dataloader.num_batches + b) % args.save_every == 0: #e % args.save_every == 0 and
                                print('Saving model at batch {0}, epoch {1}'.format(b, e))
                                checkpoint_path = os.path.join(parent_dir+'save',
                                                               'g2k_MPC_model_kfold_train_{1}_{0}_{2}.ckpt'.format(e, d, b))

                                try:
                                    sess.run(fetches=tf.initialize_all_variables()) #critical when it initializes all variables stored in default graph each time
                                except tf.errors.FailedPreconditionError:
                                    sess.run(fetches=tf.initialize_all_variables())
                                saver = tf.train.Saver(tf.all_variables())
                                saver.save(sess, checkpoint_path,
                                           global_step=e * dataloader.num_batches + b)

                                print("model saved to {}".format(checkpoint_path))

                        sess.close()

                    # make another model file with attn
                    fde_np = np.stack(fde)
                    euc_loss_np = np.ravel(euc_loss)
                    np.savetxt(fname=log_dir_fde, X=fde_np, delimiter=",")
                    np.savetxt(fname=log_dir, X=euc_loss_np, delimiter=",")

                    e += 1
                    tf.reset_default_graph()
                    del out_graph
                    out_graph = tf.Graph()

                    e_end = time.time()

                    print('Epoch time taken: ', (e_end - e_start))
                log_count_f.write(
                    'Dataset {0}= ADE steps {1}\nFDE steps = {2}'.format(d, num_targets, num_end_targets))
                # log_f.close()
                log_count_f.close()
        out_sess.close()
        del (graph)
        # else:
    # *************************************************************** VALIDATION *************************************
    # Validate

    out_graph = tf.Graph()
    with out_graph.as_default():
        with tf.Session(graph=out_graph).as_default() as val_sess:
            graph = nx_g.online_graph(args)
            l = args.leaveDataset  # int(input("Select which dataset to validate: 2.Zara1, 3.Zara2, 4.UCY \n"))
            dataloader = load.DataLoader(args=args, datasets=[0, 1, 2, 3, 4, 5], start=l, sel=0)
            dataloader.reset_data_pointer(valid=True)
            # out_sess.close()
            # with tf.Session(graph=out_graph).as_default() as sess:

            parent_dir = '/fakepath/multimodaltraj_2/'
            checkpoint_path = parent_dir+'save'
            ckpt = tf.train.get_checkpoint_state(checkpoint_dir=checkpoint_path, latest_filename='checkpoint')

            # process epoch number *****************************************
            import re
            e = re.findall(r'_[0-9]+', ckpt.model_checkpoint_path)[1]
            e = int(e.replace('_', ''))
            # **************************************************************

            true_path = []
            frame = 1

            print('Importing meta data of stored model in' + ckpt.model_checkpoint_path + '.meta')
            saver = tf.train.import_meta_graph(
                os.path.join(checkpoint_path, ckpt.model_checkpoint_path + '.meta'))
            # print ('loading model: ', ckpt.model_checkpoint_path)

            print('loading model: ', ckpt.all_model_checkpoint_paths)

            # Restore the model at the checkpoint
            saver.restore(val_sess, ckpt.model_checkpoint_path)
            # def_graph = tf.get_default_graph()

            # dataloader = load.DataLoader(args=args, datasets=[0, 1, 2, 3, 4, 5], start=l, sel=0, infer=True)
            # traj = dataloader.load_trajectories(data_file=dataloader.sel_file)

            dataloader.valid_frame_pointer = int((dataloader.len - int(
                dataloader.max * .7)) / dataloader.val_max)  # + dataloader.seq_length #+ (frame * e)

            dataloader.valid_num_batches = int(dataloader.val_max / (dataloader.batch_size ))#* 20
            dim = int(args.neighborhood_size / args.grid_size)

            with tf.variable_scope('nghood_init', reuse=True):
                out_init = tf.zeros(dtype=tf.float64, shape=(
                    dim, dim))  # (args.grid_size * (args.grid_size / 2))))
                c_hidden_init = tf.zeros(dtype=tf.float64, shape=(
                    dim, dim))  # (args.grid_size * (args.grid_size / 2))))

            cv_ade_err = []
            cv_fde_err = []

            for vb in range(dataloader.valid_num_batches):
                num_targets = 0
                start_t = time.time()
                batch, target_traj, fp = dataloader.next_step()

                if len(batch) == 0:
                    break

                graph_t = graph.ConstructGraph(current_batch=batch, framenum=fp, future_traj=target_traj)
                batch_v = list(graph_t.get_node_attr(param='node_pos_list').values())

                if len(batch_v) == 0:
                    break

                if len(np.array(batch_v).shape) > 1:
                    batch_v = np.array(batch_v)[frame:frame + args.obs_len]
                    batch_v = np.linalg.norm(batch_v, axis=2).squeeze()
                else:
                    # batch_v = np.array(batch_v)[frame:frame + args.obs_len]
                    break

                batch_v = np.transpose(batch_v)
                num_nodes = batch_v.shape[1]

                if vb == 0:
                    init_w = tf.initializers.random_normal(mean=0, stddev=1, seed=0, dtype=tf.float64)
                    hidden_state = np.zeros(shape=(dim, args.rnn_size))
                    nghood_enc = helper.neighborhood_vis_loc_encoder(
                        hidden_size=args.rnn_size,
                        hidden_len=dim,
                        num_layers=args.num_layers,
                        grid_size=args.grid_size,
                        embedding_size=args.embedding_size,
                        dropout=args.dropout)

                    ctxt_img_path = glob.glob(dataloader.current_dir + 'ctxt.png')
                    ctxt_img = tf.convert_to_tensor(imread(ctxt_img_path[0]), dtype=tf.float64)

                    ctxt_img_pd = tf.convert_to_tensor(
                        tf.pad(ctxt_img, paddings=tf.constant([[1, 1, ], [0, 1], [0, 0]])),
                        dtype=tf.float64)
                    width = int(ctxt_img_pd.shape.dims[0])
                    height = int(ctxt_img_pd.shape.dims[1])

                    ctxt_img_pd = tf.expand_dims(ctxt_img_pd, axis=0)
                    _2dconv = tf.nn.conv2d(input=ctxt_img_pd,
                                           filter=tf.random_normal(shape=[width - dim + 1, height - dim + 1, 3, 1],
                                                                   dtype=tf.float64),
                                           padding='VALID', strides=[1, 1, 1, 1])

                    _2dconv = tf.squeeze(_2dconv)
                    _2dconv = args.lambda_param * _2dconv

                    # stat_ngh = helper.neighborhood_stat_enc(
                    #     hidden_size=args.rnn_size,
                    #     num_layers=args.num_layers,
                    #     grid_size=args.grid_size,
                    #     dim=args.num_freq_blocks)
                    #
                    # with tf.variable_scope('ngh_stat'):
                    #     static_mask = tf.placeholder(name='static_mask',  # shape=(dim, static_frame_w),
                    #                                  dtype=tf.float64)
                    #     social_frame = tf.placeholder(name='social_frame',  # shape=(static_frame_w,dim),
                    #                                   dtype=tf.float64)
                    #     state_f00_b00_c = tf.placeholder(name='state_f00_b00_c',  # shape=(dim,hidden_size),
                    #                                      dtype=tf.float64)
                    #     c_hidden_states = tf.placeholder(name='c_hidden_states',
                    #                                      # shape=(dim, (grid_size * (grid_size/2))),
                    #                                      dtype=tf.float64)
                    #     output = tf.placeholder(dtype=tf.float64,
                    #                             # shape=[num_nodes, (grid_size * (grid_size / 2))],
                    #                             name="output")
                # krnl_mdl = MC.g2k_lstm_mc(in_features=nghood_enc.input,
                #                             num_nodes=num_nodes, obs_len=args.obs_len,
                #                             lambda_reg=args.lambda_param,
                #                             sess_g=def_graph)

                # krnl_mdl = mcr.g2k_lstm_mcr(in_features=nghood_enc.input,
                #                             num_nodes=num_nodes, obs_len=args.obs_len,
                #                             hidden_size=args.rnn_size,
                #                             lambda_reg=args.lambda_param,
                #                             sess_g=def_graph)

                # num_targets += num_nodes

                krnl_mdl = mcr.g2k_lstm_mcr(in_features=nghood_enc.input,
                                            num_nodes=num_nodes, obs_len=args.obs_len,
                                            lambda_reg=args.lambda_param,
                                            hidden_size=args.rnn_size,
                                            sess_g=out_graph)

                with tf.variable_scope('weight_input', reuse=True):
                    weight_i = tf.Variable(name='weight_i',
                                           initial_value=init_w(shape=(num_nodes, dim)),
                                           trainable=True, dtype=tf.float64)
                    weight_ii = tf.Variable(name='weight_ii',
                                            initial_value=init_w(shape=(dim, args.obs_len)),
                                            trainable=True, dtype=tf.float64)
                tf.initialize_variables(var_list=[weight_i, weight_ii]).run()

                inputs = tf.convert_to_tensor(batch_v, dtype=tf.float64)
                inputs = tf.matmul(inputs, weight_i)
                inputs = tf.matmul(weight_ii, inputs)

                vislet = dataloader.vislet[:,
                         dataloader.valid_frame_pointer:dataloader.valid_frame_pointer + num_nodes]  # tf.expand_dims(batch_v[0], axis=0)
                vislet_emb = tf.matmul(vislet, weight_i)

                if vb == 0:
                    vislet_past = vislet_emb

                vislet_rel = vislet_past * vislet_emb

                stat_mask = tf.zeros(shape=(dim, args.obs_len), dtype=tf.float64)
                stat_mask += tf.expand_dims(tf.range(start=0, limit=1, delta=(1 / args.obs_len), dtype=tf.float64), axis=0)
                _2dconv_in = tf.matmul(_2dconv, stat_mask)
                # static_mask_nd = stat_mask.eval()

                # sess.run(fetches=tf.initialize_all_variables())
                # krnl_mdl.weight_r, krnl_mdl.attn,
                tf.initialize_variables(
                    var_list=[krnl_mdl.weight_v, krnl_mdl.weight_r, krnl_mdl.bias_v]).run()  # , krnl_mdl.embed_vis
                tf.initialize_variables(
                    var_list=[krnl_mdl.cost, krnl_mdl.attn, krnl_mdl.weight_c,
                              krnl_mdl.weight_o]).run()
                # for frame_iter in iter(val_traj):
                # check if get_node_attr gets complete sequence for all nodes
                # num_nodes x obs_length
                # if num_nodes > 0 :
                cv_err = []
                fde = []

                # (frame, _), = frame_iter.items()
                for frame in batch:
                    print('Frame {0}'.format(frame))

                    true_path.append(batch[frame])
                    st_embeddings, hidden_state, ng_output, c_hidden_state = \
                        val_sess.run([nghood_enc.input, nghood_enc.state_f00_b00_c,
                                  nghood_enc.output, nghood_enc.c_hidden_state],
                                 feed_dict={nghood_enc.input: inputs.eval(session=val_sess),
                                            nghood_enc.state_f00_b00_c: hidden_state,
                                            nghood_enc.output: out_init.eval(session=val_sess),
                                            nghood_enc.c_hidden_state: c_hidden_init.eval(session=val_sess)})

                    # static_mask_np, social_frame_np =\
                    #     val_sess.run([static_mask, output],
                    #              feed_dict={static_mask: static_mask_nd,
                    #                         social_frame: ng_output,
                    #                         state_f00_b00_c: hidden_state,
                    #                         output: out_init.eval(),
                    #                         c_hidden_states: c_hidden_init.eval()
                    #                         })
                    #
                    # input = tf.matmul(b=static_mask_np,
                    #                   a=social_frame_np).eval()  # Soft-attention mechanism equipped with static grid
                    # combined_ngh, hidden_state = val_sess.run([stat_ngh.input, stat_ngh.hidden_state],
                    #                                       feed_dict={stat_ngh.input: input,
                    #                                                  stat_ngh.hidden_state: hidden_state})
                    # reg_ng = np.transpose(args.lambda_param * np.transpose(ng_output))
                    # hidden_state,

                    pred_path = tf.random_normal(shape=(2, 12, num_nodes))
                    pred_path, hidden_state, prob_mat = \
                        val_sess.run([krnl_mdl.pred_path_band, krnl_mdl.hidden_states, krnl_mdl.cost],
                                 # krnl_mdl.hidden_states,
                                 feed_dict={
                                     krnl_mdl.outputs:  # st_embeddings,
                                        np.concatenate((st_embeddings, vislet_emb.eval(session=val_sess)), axis=0),
                                     krnl_mdl.ngh: _2dconv_in.eval(session=val_sess),
                                     krnl_mdl.rel_features: vislet_rel.eval(session=val_sess),
                                     krnl_mdl.hidden_states: hidden_state,
                                     krnl_mdl.out_size: num_nodes,
                                     # krnl_mdl.lambda_reg: args.lambda_reg,
                                     krnl_mdl.pred_path_band: pred_path.eval(session=val_sess)
                                 })

                    # krnl_mdl.ngh = reg_ng
                    # krnl_mdl.out_size = tf.Variable(num_nodes)
                    # krnl_mdl.hidden_states = hidden_state
                    # krnl_mdl.outputs = np.concatenate((st_embeddings, vislet_emb.eval()), axis=0)
                    # krnl_mdl.pred_path_band = tf.random_normal(shape=(2, 12, num_nodes))
                    # krnl_mdl.rel_features = vislet_rel.eval(session=val_sess)
                    # krnl_mdl.forward()
                    # pred_path = krnl_mdl.pred_path_band

                    # attn = tf.nn.softmax(tf.exp(krnl_mdl.attn) / tf.cumsum(tf.exp(krnl_mdl.attn)))

                    # weigh hidden states then send them back to glstm on next step
                    # krnl_mdl.hidden_states = tf.nn.softmax(krnl_mdl.hidden_states)
                    # hidden_state = krnl_mdl.hidden_states.eval()
                    # weightage of final hidden states resulting from chaining of hidden states through social (dynamic) neighborhood then weighted static neighborhood
                    # then softmaxing hidden states
                    # hidden_state = np.matmul(attn.eval(), hidden_state)
                    # adj_mat = tf.matmul(tf.nn.softmax(hidden_state),
                    #                     tf.ones(shape=(hidden_state.shape[1], 1), dtype=tf.float64))

                    # GG-NN 2016 A_nx2n , we use A_nxn; n is |G_v| cardinality of node v in Graph G.
                    # hidden_state = adj_mat.eval() * hidden_state
                    attn = tf.nn.softmax(tf.exp(krnl_mdl.attn) / tf.cumsum(tf.exp(krnl_mdl.attn)))

                    # weigh hidden states then send them back to glstm on next step
                    krnl_mdl.hidden_states = tf.nn.softmax(krnl_mdl.hidden_states)
                    hidden_state = krnl_mdl.hidden_states.eval(session=val_sess)
                    # # weightage of final hidden states resulting from chaining of hidden states through social (dynamic) neighborhood then weighted static neighborhood
                    # # then softmaxing hidden states
                    hidden_state = np.matmul(attn.eval(session=val_sess), hidden_state)
                    adj_mat = tf.matmul(tf.nn.softmax(hidden_state),
                                        tf.ones(shape=(hidden_state.shape[1], 1), dtype=tf.float64))

                    # # GG-NN 2016 A_nx2n , we use A_nxn; n is |G_v| cardinality of node v in Graph G.
                    hidden_state = adj_mat.eval(session=val_sess) * hidden_state

                    pred_path = np.transpose(pred_path, (2, 1, 0))
                    # ped_ids = list(graph_t.getNodes().keys())
                    num_targets += num_nodes
                    if num_nodes > 0:
                        for i, itr in zip(range(num_nodes), iter(target_traj)):
                            try:
                                if len(target_traj[itr]) < args.pred_len:
                                    euc_loss = np.linalg.norm(
                                        (pred_path[i][0:len(target_traj[itr])] - target_traj[itr]), ord=2) / len(
                                        target_traj) / 12
                                    err = (pred_path[i][len(target_traj[itr]) - 1] - target_traj[itr][len(target_traj[itr]) - 1])
                                else:
                                    euc_loss = np.linalg.norm(
                                        (pred_path[i][0:args.pred_len] - target_traj[itr][0:args.pred_len]),
                                        ord=2) / 12  # / len(target_traj)
                                    err = (pred_path[i][args.pred_len - 1] - target_traj[itr][args.pred_len - 1])
                                        # np.linalg.norm(
                                        # (pred_path[i][args.pred_len - 1] - target_traj[itr][args.pred_len - 1]),
                                        # ord=2)

                                fde.append(err)
                                cv_err.append(euc_loss)
                                print('euc_loss = ', euc_loss)
                                print('fde_err = ', err)
                            except KeyError:
                                i += 1
                                continue

                # cv_fde_err.append(fde)
                end_t = time.time()

                # next(frame_iter)
                if len(cv_err) > 0:
                    cv_ade_err.append(np.mean(cv_err))
                if len(fde) > 0:
                    if l == 5:
                        cv_fde_err.append(np.linalg.norm(fde) / num_nodes)  # np.mean(fde)
                    else:
                        cv_fde_err.append(np.linalg.norm(fde)/len(batch)) #np.mean(fde)

                print('{0} seconds to complete'.format(end_t - start_t))
                print('Batch {0} of {1}, Loss = {2}, num_ped={3}'
                      .format(vb, dataloader.valid_num_batches, krnl_mdl.cost, len(target_traj)))

                # dataloader.tick_frame_pointer()
                dataloader.frame_pointer = frame

            # traj = dataloader.load_trajectories(data_file=dataloader.sel_file)
            # ped_ids = len(list(graph_t.getNodes().keys()))
            # cv_ade_err = np.divide(cv_err, ped_ids)
            # cv_fde_err = np.divide(fde, ped_ids)

            print('Cross-Validation total mean error (ADE) for dataset {0} = '.format(l), np.mean(cv_ade_err))
            print('Cross-Validation total final error (FDE) for dataset {0} = '.format(l), np.mean(cv_fde_err))
            checkpoint_path = os.path.join(parent_dir+'save',
                                           'g2k_MPC_model_kfold_val_{0}.ckpt'.format(l))

            saver.save(val_sess, checkpoint_path, global_step = e * dataloader.valid_num_batches + vb)
            print("model saved to {}".format(checkpoint_path))
        val_sess.close()

if __name__ == '__main__':
    main()
