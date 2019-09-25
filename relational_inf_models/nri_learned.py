import torch.nn as nn

def graph_to_kernel():
    # TODO construct kernel from random walk theory
    # transform graph to kernel to parameterize
    # fNRI factorization of edges using the softmax
    edges = gumbel_softmax(logits, tau=args.temp, hard=args.hard)
    prob = my_softmax(logits, -1)
    loss_kl = kl_categorical_uniform(prob, args.num_atoms, edge_types)

    return

# Through gated neighborhood network (neighborhood encoders & random walker)
def infer_rlns(adj_mat):
    # infer relationships from the kernel (kernel output by random walker algorithm)
    sig = nn.Sigmoid()
    prob_mat = sig(adj_mat)

    return prob_mat

def eval_rln_ngh(ngh, static_ngh):
    # evaluate importance of relations to form the hybrid neighborhood(social(temporal) + static(spatial))
    # prob_mat = nn.Sigmoid(adj_mat)

    return

