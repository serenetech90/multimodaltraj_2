import argparse

class ArgsParser():
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
    parser.add_argument('--num_epochs', type=int, default=10,
                        help='number of epochs')
    # Frequency at which the model should be saved parameter
    parser.add_argument('--save_every', type=int, default=50,
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
    parser.add_argument('--neighborhood_size', type=int, default=64,
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
    parser.add_argument('--leaveDataset', type=int, default=2,
                        help='The dataset index to be left out in training')
    # Lambda regularization parameter (L2)
    parser.add_argument('--lambda_param', type=float, default=0.0005,
                        help='L2 regularization parameter')
