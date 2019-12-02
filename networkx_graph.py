''' class online graph is to construct and adapt underlying graph as soon as new step is take by pedestrian '''

import networkx as nx
import numpy as np
import torch
import torch.cuda
# import criterion as cr

batch_size = 4

class online_graph():

    def reset_graph(self,framenum):
        del self.nodes
        del self.edges
        self.onlineGraph.delGraph(framenum)

        self.nodes = [{}]
        self.edges = [[]]

    def __init__(self, args):

        self.diff = args.seq_length
        self.batch_size = args.batch_size  # 1
        # self.seq_length = args.seq_length # 1
        self.nodes = [{}]
        self.edges = [{}]
        self.onlineGraph = Graph()

    def ConstructGraph(self, current_batch, future_traj, framenum, stateful=True, valid=False):
        self.onlineGraph.step = framenum

        if valid:
            for pedID, pos in current_batch.items():
                node_id = pedID
                node_pos_list = {}
                node_pos_list[framenum] = [pos]

                node = Node(node_id, node_pos_list)
                if framenum > 0 and framenum % 8:
                    node.setTargets(seq=future_traj[pedID])
                self.onlineGraph.setNodes(framenum, node)
        else:
            self.pos_list_len = len(current_batch)
            for idx,itr in zip(current_batch, range(len(current_batch))):
                try:
                    frame = current_batch[idx]  # * self.diff
                except KeyError:
                    key = list(current_batch.keys())
                    frame = current_batch[key[0]]

                for item in frame:
                        (pedID, pos), = item.items()
                        pedID = int(pedID)
                        node_id = pedID
                        node_pos_list = pos
                        # if self.node:
                        node = Node(node_id, node_pos_list)
                        # if framenum > 8 and framenum % 8 == 1:
                        try:
                            if len(future_traj[pedID]) < framenum:
                                node.setTargets(seq=future_traj[pedID][0:12])
                            elif len(future_traj[pedID][framenum:framenum+12]) < 12:
                                node.setTargets(future_traj[pedID])
                            else:
                                node.setTargets(seq=future_traj[pedID][framenum:framenum+12])
                        except KeyError:
                            continue
                        self.onlineGraph.setNodes(itr, node)

        self.onlineGraph.dist_mat = torch.zeros(len(self.nodes), len(self.nodes))
        # check online graph nodes are correct
        return self.onlineGraph

    def linkGraph(self, curr_graph, new_edges, frame):
        # common_nodes, n1, _ = cr.get_common_nodes(curr_graph, frame)
        # bring nodes from previous frame
        # if len(common_nodes):
        #     for item in common_nodes:
        n1 = curr_graph.getNodes()
        for item in n1:
            edge_id = (item, item)
            dist = torch.norm(
                torch.from_numpy(np.subtract(n1[frame][item.id].pos[frame],
                                             n1[frame - 1][item.id].pos[frame - 1])), p=2)

            edge_weight = dist #{frame:dist}
            e = Edge(edge_id, edge_weight)
            curr_graph.setEdges(u=item.id, framenum=frame)
            curr_graph.setEdges(u= item.id , v=item.id, obj=e, mode='s')
        # else:
        #     self.onlineGraph.edges.append([])

class Graph(nx.MultiDiGraph):
    def __init__(self):
        super(Graph, self).__init__()
        self.adj_mat = []
        self.dist_mat = []

        # self.nodes = [{}]
        # self.edges = [[]]  # dict
        self.Stateful = True
        self.step = 0

        # by default the graph is stateful and each graph segment is connected to the previous temporal segment
        # unless nodes in a graph no longer exist in the scene, then we need to disconnect and destroy variables

    def getNodes(self):
        return self.nodes

    def getEdges(self):
        return self.edges

    def setNodes(self, framenum, node, pos_list_len=8):
        # if len(self.nodes) <= framenum:
        if node.id in self.nodes.keys():
            try:
                self.nodes[node.id]['node_pos_list'][framenum] = node.pos
            except IndexError:
                pass
                # self.nodes[node.id]['node_pos_list'] = \
                #     np.concatenate((self.nodes[node.id]['node_pos_list'], np.reshape(node.pos, newshape=(1, 2))))
        else:
            self.add_node(node.id,seq= node.seq,
                    node_pos_list= np.zeros(shape=(pos_list_len, 2)),
                    state= node.state,
                    cell=node.cell,
                    targets= node.targets,
                    vel= node.vel)
        # self.node_attr_dict_factory()
        # self.nodes.append({})
        # self.nodes[framenum][node.id] = node
        # else:
        #     self.nodes[framenum][node.id] = node

    def setEdges(self, framenum , obj ,u,v=None, mode='t'):
        if mode == 't':
            nx.add_cycle(self.graph, self.nodes[u])
        else:
            if len(self.edges) <= framenum - 1:
                self.add_edge(u_for_edge=u, v_for_edge=v, key=str((u, v)), attr=obj.edge_weight)

        # print("appended new empty array")
        #     self.edges.append([])
        #     self.edges[framenum - 1].append(edge)
        # else:
        #     new_edge = Edge(edge.id , )

    def get_node_attr(self, param):
        return nx.get_node_attributes(G=self, name=param)

    def delGraph(g, framenum):
        g.clear()
        # del self.nodes
        # del self.edges
        # self.nodes = []
        # self.edges = []
        # for i in range(framenum):
        #     self.nodes.append({})
        #     self.edges.append([])

class Node():
    def __init__(self, node_id, node_pos_list):
        self.id = node_id
        # if len(self.pos):
        self.pos = node_pos_list
        self.state = torch.zeros(batch_size, 256)  # 256 self.human_ebedding_size
        self.cell = torch.zeros(batch_size, 256)
        self.seq = []
        self.targets = []
        self.vel = 0

    def setState(self, state, cell):
        self.state = state
        self.cell = cell

    def getState(self):
        return self.state, self.cell

    def setPrediction(self, seq):
        self.seq = seq

    def getPrediction(self):
        return self.seq

    def setTargets(self,seq):
        self.targets.append(seq)

    def getTargets(self):
        return self.targets

    def get_node_attr_dict(self):
        return {
                'seq': self.seq,
                'node_pos_list': self.pos,
                'state': self.state,
                'cell': self.cell,
                'targets': self.targets,
                'vel': self.vel
                }


class Edge():
    def __init__(self, edge_id, edge_pos_list):
        self.id = edge_id
        self.dist = edge_pos_list


