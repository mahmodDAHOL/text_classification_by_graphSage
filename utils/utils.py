from pyvis.network import Network
from dataset import DataItem


def show_graph(dataitem: DataItem, output_filename: str) -> Network:
    graph = dataitem.graph.nodes[:][1]._graph
    nodes = graph.nodes

    edges = dataitem.graph.edges()
    tokens = []
    for i, _ in enumerate(nodes):
        tokens.append(nodes[i][1].getitem('token'))
        if i == len(nodes) - 1:
            break
    nodes_ids = [id for id in range(graph.get_node_num())]
    nodes_titls = [token for token in tokens]

    g = Network()
    g.add_nodes(nodes_ids, title=nodes_titls)
    g.add_edges(edges)
    g.show(output_filename)
    return g
