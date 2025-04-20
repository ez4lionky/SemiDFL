import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


def plot_topology(edge_matrix, num_worker, l_client_ids, m_client_ids, u_client_ids):
    G = nx.DiGraph(edge_matrix)
    node_colors = np.array(['#98CF34' for _ in range(num_worker)])
    node_colors[l_client_ids] = '#98CF34'
    node_colors[m_client_ids] = '#338AE9'
    node_colors[u_client_ids] = '#C77150'
    nx.draw(G, node_size=500, with_labels=True, node_color=node_colors)
    plt.show()
    return


def get_topology(nid, fully_supervised=False):
    num_worker = 10
    l_client_ids = [0]
    m_client_ids = [1]
    u_client_ids = [2]
    if nid == 1:
        dist_wm = np.array(
            [[1, 0, 1, 0, 0, 0, 1, 1, 1, 0],
             [0, 1, 0, 1, 1, 1, 0, 0, 0, 1],
             [1, 0, 1, 0, 1, 0, 0, 1, 1, 0],
             [0, 1, 0, 1, 1, 1, 0, 1, 0, 0],
             [0, 1, 1, 1, 1, 0, 0, 0, 0, 1],
             [0, 1, 0, 1, 0, 1, 1, 0, 1, 0],
             [1, 0, 0, 0, 0, 1, 1, 1, 0, 1],
             [1, 0, 1, 1, 0, 0, 1, 1, 0, 0],
             [1, 0, 1, 0, 0, 1, 0, 0, 1, 1],
             [0, 1, 0, 0, 1, 0, 1, 0, 1, 1]
             ])
        l_client_ids = [0, 1]
        m_client_ids = [2, 3]
        u_client_ids = [4, 5, 6, 7, 8, 9]
        # plot_topology(dist_wm, num_worker, l_client_ids, m_client_ids, u_client_ids)
    elif nid == 2:
        # duplication of fully supervised, same topology as nid=1
        dist_wm = np.array(
            [[1, 0, 1, 0, 0, 0, 1, 1, 1, 0],
             [0, 1, 0, 1, 1, 1, 0, 0, 0, 1],
             [1, 0, 1, 0, 1, 0, 0, 1, 1, 0],
             [0, 1, 0, 1, 1, 1, 0, 1, 0, 0],
             [0, 1, 1, 1, 1, 0, 0, 0, 0, 1],
             [0, 1, 0, 1, 0, 1, 1, 0, 1, 0],
             [1, 0, 0, 0, 0, 1, 1, 1, 0, 1],
             [1, 0, 1, 1, 0, 0, 1, 1, 0, 0],
             [1, 0, 1, 0, 0, 1, 0, 0, 1, 1],
             [0, 1, 0, 0, 1, 0, 1, 0, 1, 1]
             ])
        l_client_ids = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        m_client_ids = []
        u_client_ids = []
    elif nid == 3:
        # 1 l client, 3 mixed client
        dist_wm = np.array(
            [[1, 0, 1, 0, 0, 0, 1, 1, 1, 0],
             [0, 1, 0, 1, 1, 1, 0, 0, 0, 1],
             [1, 0, 1, 0, 1, 0, 0, 1, 1, 0],
             [0, 1, 0, 1, 1, 1, 0, 1, 0, 0],
             [0, 1, 1, 1, 1, 0, 0, 0, 0, 1],
             [0, 1, 0, 1, 0, 1, 1, 0, 1, 0],
             [1, 0, 0, 0, 0, 1, 1, 1, 0, 1],
             [1, 0, 1, 1, 0, 0, 1, 1, 0, 0],
             [1, 0, 1, 0, 0, 1, 0, 0, 1, 1],
             [0, 1, 0, 0, 1, 0, 1, 0, 1, 1]
             ])
        l_client_ids = [0]
        m_client_ids = [1, 2, 3]
        u_client_ids = [4, 5, 6, 7, 8, 9]
        # plot_topology(dist_wm, num_worker, l_client_ids, m_client_ids, u_client_ids)
    elif nid == 4:
        # symmetric topology
        dist_wm = np.array(
            [[1, 1, 0, 1, 1, 1, 1, 0, 0, 0],
             [1, 1, 1, 0, 0, 0, 0, 1, 1, 1],
             [0, 1, 1, 0, 0, 0, 0, 0, 0, 0],
             [1, 0, 0, 1, 0, 0, 0, 0, 0, 0],
             [1, 0, 0, 0, 1, 0, 0, 0, 0, 0],
             [1, 0, 0, 0, 0, 1, 0, 0, 0, 0],
             [1, 0, 0, 0, 0, 0, 1, 0, 0, 0],
             [0, 1, 0, 0, 0, 0, 0, 1, 0, 0],
             [0, 1, 0, 0, 0, 0, 0, 0, 1, 0],
             [0, 1, 0, 0, 0, 0, 0, 0, 0, 1],
             ])
        l_client_ids = [0, 1]
        m_client_ids = [2, 3]
        u_client_ids = [4, 5, 6, 7, 8, 9]
        # plot_topology(dist_wm, num_worker, l_client_ids, m_client_ids, u_client_ids)
    elif nid == 5:
        # ring topology
        dist_wm = np.array(
            [[1, 0, 0, 1, 0, 0, 0, 0, 0, 1],
             [0, 1, 0, 0, 1, 0, 1, 0, 0, 0],
             [0, 0, 1, 0, 0, 1, 0, 1, 0, 0],
             [1, 0, 0, 1, 0, 0, 1, 0, 0, 0],
             [0, 1, 0, 0, 1, 0, 0, 1, 0, 0],
             [0, 0, 1, 0, 0, 1, 0, 0, 1, 0],
             [0, 1, 0, 1, 0, 0, 1, 0, 0, 0],
             [0, 0, 1, 0, 1, 0, 0, 1, 0, 0],
             [0, 0, 0, 0, 0, 1, 0, 0, 1, 1],
             [1, 0, 0, 0, 0, 0, 0, 0, 1, 1],
             ])
        l_client_ids = [0, 1, 2]
        m_client_ids = [3, 4, 5]
        u_client_ids = [6, 7, 8, 9]
        # plot_topology(dist_wm, num_worker, l_client_ids, m_client_ids, u_client_ids)
    elif nid == 6:
        num_worker = 20
        dist_wm = np.array(
            [[1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0],
             [0, 1, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1],
             [0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 0, 0, 1],
             [1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0],
             [1, 0, 1, 1, 1, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 0, 0, 0, 1, 1],
             [1, 1, 0, 1, 0, 1, 1, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1],
             [0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1],
             [1, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 1, 1, 1, 0, 1, 1, 0, 0],
             [1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 1, 0, 1, 0],
             [1, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 0, 1, 0, 0, 1],
             [1, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0],
             [1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 0, 0],
             [1, 1, 1, 1, 0, 1, 0, 0, 0, 1, 1, 0, 1, 1, 0, 1, 0, 1, 0, 0],
             [1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1],
             [0, 1, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 0, 1],
             [1, 1, 0, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 0],
             [0, 1, 0, 1, 0, 1, 1, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 1],
             [1, 1, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 1, 0, 1, 0, 1],
             [1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1],
             [1, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 1]
             ])
        l_client_ids = [0, 1, 2, 3]
        m_client_ids = [4, 5, 6, 7]
        u_client_ids = [_ for _ in range(8, 20)]
        # plot_topology(dist_wm, num_worker, l_client_ids, m_client_ids, u_client_ids)
    elif nid == 0:
        # same as nid=1, but fully connected
        dist_wm = np.array(
            [[1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
             [1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
             [1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
             [1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
             [1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
             [1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
             [1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
             [1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
             [1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
             [1., 1., 1., 1., 1., 1., 1., 1., 1., 1.]]
        )
        l_client_ids = [0, 1]
        m_client_ids = [2, 3]
        u_client_ids = [4, 5, 6, 7, 8, 9]
    else:
        assert 'Wrong dist_wm_id!'
    if fully_supervised:
        l_client_ids = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        m_client_ids = []
        u_client_ids = []
    return num_worker, l_client_ids, m_client_ids, u_client_ids, dist_wm
