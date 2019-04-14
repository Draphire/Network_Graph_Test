from multiprocessing.spawn import freeze_support

import networkx as nx
import matplotlib.pyplot as plt
import os
import plotly.plotly as py
import plotly.graph_objs as go
from plotly.offline.offline import matplotlib
import matplotlib
import matplotlib.pyplot as plt
import graph
import pickle

import plotly as p
import plotly.graph_objs as go

from randomnetwork.random_network_analyzer import calculate_degree_prob_distribution
from realnetwork import distance_analyzer, path
# from realnetwork.analyzer2 import GraphAnalyzer
from realnetwork.clustering_coefficient import plot_clustering_coefficient_dist_log_binning
from realnetwork.clustering_coefficient_analyzer import calculate_local_clustering_coefficients
from realnetwork.convert_txt_to_csv import get_data_frame
from realnetwork.degree_analyzer import plot_and_store_degree_cc_prob_distribution
from realnetwork.degree_correlation import analyze_degree_correlation, plot_degree_correlations
from realnetwork.distance_analyzer import get_distance_distribution, calculate_distance_prob_distribution
# from realnetwork.real_network import _analyze_real_network_properties, _load_graph_csv_from_file_system, \
#     _compute_real_network_properties


from multiprocessing.managers import BaseManager

from multiprocessing import Process, Queue

from realnetwork.real_network import plot_graph, plot_graph_big
from realnetwork.real_network_analyzer import GraphAnalyzer


def compute_properties(graph, analyzer, save_dir):
    analyzer.compute_average_degree()
    print("Average degree: " + str(round(analyzer.avg_degree, 5)))
    print("largest k: " + str(analyzer.comptue_max_degree()))
    print("|V|: " + str(graph.get_node_count()))
    print("|E|: " + str(graph.get_edge_count()))
    print("Neighbor of vertex 0: " + str(graph.neighbor_of_node(1)))
    analyzer.compute_sssp_related_properties([])
    print("Betweenness: " + str(len(analyzer.bc_values)))
    print("Avg path length: " + str(round(analyzer.avg_path_length, 5)))
    print("Closeness: " + str(len(analyzer.close_values)))
    analyzer.compute_degree_correlation()
    print("Degree correlation: " + str(list(analyzer.knn.values())[:10]))
    analyzer.compute_degree_based_clustering_coef()
    analyzer.compute_avg_clustering_coef()
    print("Avg clustering coef: " + str(analyzer.avg_clustering_coef))
    analyzer.compute_degree_prob_distribution()
    print("Degree prob distribution: " + str(analyzer.degree_prob_distribution))

    property_info_dict = {"avg_degree" : analyzer.avg_degree,
                        "degree_distribution" : analyzer.degree_prob_distribution,
                        "degrees" : graph.get_each_node_degree(),
                        "bc_values" : analyzer.bc_values,
                        "avg_path_len" : analyzer.avg_path_length,
                        "closeness" : analyzer.close_values,
                        "degree_correlation" : analyzer.knn,
                        "avg_clustering_coef" : analyzer.avg_clustering_coef,
                        "clustering_coef" : analyzer.degree_based_clustering_coef,
                        "1st_moment" : analyzer.compute_nth_moment(1),
                        "2nd_moment" : analyzer.compute_nth_moment(2),
                        "3rd_moment" : analyzer.compute_nth_moment(3)}
    # store the calculated properties for the graph
    with open(os.path.join(save_dir,"properties.pkl"), "wb") as f:
        pickle.dump(property_info_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
# plt.figure(figsize=(18,18))
# network = random_network_generator.generate_random_network(100, 0.5)
# network = _analyze_real_network_properties("amazon")
"""
 Draw a random graph with 2**i nodes,
 and p=i/(2**i)
 """
# g_random = nx.gnp_random_graph(2 ** i, 2 * i / (2 ** i))

#

# # G = _load_graph_csv_from_file_system("amazon");
#
# # _compute_random_network_properties
#
# nx.draw(G, node_size=1)
# graph_pos = nx.spring_layout(G)
# nx.draw_networkx_nodes(G, graph_pos, node_size=10, node_color='blue', alpha=0.3)
# nx.draw_networkx_edges(G, graph_pos)
# nx.draw_networkx_labels(G, graph_pos, font_size=8, font_family='sans-serif')
#
# plt.show()
from scalefreenetwork.degree_analyzer import plot_and_store_degree_prob_distribution, count_degree, \
    calculate_degree_moment
from scalefreenetwork.scale_free_network import _compute_scale_free_properties

# network = _load_graph_csv_from_file_system("amazon")

# for diameter
# G = nx.gnp_random_graph(50,0.5)
# network = nx.gnp_random_graph(50,0.5)
# # diameter = distance_analyzer.find_network_diameter(network)
# dist_prod = get_distance_distribution(network)
#
# dist_prob_dist = calculate_distance_prob_distribution(dist_prod);
#
# print(dist_prob_dist)


# print(len(network.nodes()))
# print(len(network.edges()))
# scale_free_property = _compute_scale_free_properties(network)
#
# print(scale_free_property["expected_degree_exponent"])

# distance_analyzer.get_distance_distribution(network)
# analyze_degree_correlation(network)
# plot_clustering_coefficient_dist_log_binning(network);

# G = nx.gnp_random_graph(100,0.5)
# cc = calculate_local_clustering_coefficients(network)
# cc_list=list(cc.values())
# cc[] = cc
# cc = sorted(cc)
#graph2 = graph.__init__("csv/amazon.csv")

# get_data_frame("p2network")

# graph = graph.Graph("p2network")
# graph = graph.Graph("p2networktestneg")
# graph = graph.Graph("p2networktrain")
graph = graph.Graph("facebook_train")
# graph = graph.Graph("csv/graph.txt")

print(graph.get_nodes())
# print(graph.neighbor_of_node(1))

analyzer = GraphAnalyzer(graph)
save_dir = "csv"

k = 1
plt.rcParams["figure.figsize"] = (11, 7)
nx_graph = nx.Graph()
# own_graph = graph.Graph("csv/amazon.txt")
# own_graph = graph.Graph("csv/amazon")
# print(own_graph.get_vertices())
# own_graph = graph.Graph(sys.argv[1])
degrees = graph.get_each_node_degree()
# print(own_graph.get_degrees())
hubs = []
matplotlib.rcParams.update({'font.size': 20})
for v in graph.get_nodes():
    # print(v)
    if degrees[v] > k:
        hubs.append(v)
        # print(v)
        for w in graph.neighbor_of_node(v):
            nx_graph.add_edge(v, w)
result_dir = path.DB_PLOT_DIR_PATH


plot_graph_big(nx_graph,save_dir)
# compute_properties(graph, analyzer, save_dir)


# G=nx.random_geometric_graph(50,0.125)
# closeness = nx.closeness_centrality(G)
# x = list(closeness.keys())
# y = list(closeness.values())
# print(x)
# print(y)
#
# p.offline.plot({
#     "data": [go.Scatter(x=x, y=y)],
#     "layout": go.Layout(title="closeness")
# }, auto_open=False, )
#
# p.show
# plot_closeness(G)



# graph.get_each_node_degree()
# graph2 = new graph("csv/amazon.csv")

# graph = graph(path.CSV_NETWORK_DIR_PATH + "\\" + "amazon" + ".txt")

# print(cc)
# print(sorted(cc_list))
# plot_and_store_degree_cc_prob_distribution("amazon", cc)

# dc = analyze_degree_correlation(network)
# plot_degree_correlations(dc)

# dc = count_degree(network)
#
# plot_and_store_degree_prob_distribution("test", dc)

# plot_clustering_coefficient_dist_log_binning(network)
# _compute_real_network_properties("amazon", network)



# def main():
#     BaseManager.register('get_queue', callable=lambda:  Queue.Queue())
#
#     manager = BaseManager(address=('', 5000), authkey='abc')
#     manager.start()
#     manager.shutdown()

# if __name__ == '__main__':
    # freeze_support() here if program needs to be frozen
    # freeze_support()
    # main()  # execute this only when run directly, not when imported!
# degreecount = count_degree(network)
#
# print(calculate_degree_moment(degreecount))
# plot_and_store_degree_prob_distribution("test", degreecount)


# G=nx.random_geometric_graph(200,0.125)
# pos=nx.get_node_attributes(G,'pos')
#
# dmin=1
# ncenter=0
# for n in pos:
#     x,y=pos[n]
#     d=(x-0.5)**2+(y-0.5)**2
#     if d<dmin:
#         ncenter=n
#         dmin=d
#
# p=nx.single_source_shortest_path_length(G,ncenter)
#
#
# edge_trace = go.Scatter(
#     x=[],
#     y=[],
#     line=dict(width=0.5,color='#888'),
#     hoverinfo='none',
#     mode='lines')
#
# for edge in G.edges():
#     x0, y0 = G.node[edge[0]]['pos']
#     x1, y1 = G.node[edge[1]]['pos']
#     edge_trace['x'] += tuple([x0, x1, None])
#     edge_trace['y'] += tuple([y0, y1, None])
#
# node_trace = go.Scatter(
#     x=[],
#     y=[],
#     text=[],
#     mode='markers',
#     hoverinfo='text',
#     marker=dict(
#         showscale=True,
#         # colorscale options
#         #'Greys' | 'YlGnBu' | 'Greens' | 'YlOrRd' | 'Bluered' | 'RdBu' |
#         #'Reds' | 'Blues' | 'Picnic' | 'Rainbow' | 'Portland' | 'Jet' |
#         #'Hot' | 'Blackbody' | 'Earth' | 'Electric' | 'Viridis' |
#         colorscale='YlGnBu',
#         reversescale=True,
#         color=[],
#         size=10,
#         colorbar=dict(
#             thickness=15,
#             title='Node Connections',
#             xanchor='left',
#             titleside='right'
#         ),
#         line=dict(width=2)))
#
# for node in G.nodes():
#     x, y = G.node[node]['pos']
#     node_trace['x'] += tuple([x])
#     node_trace['y'] += tuple([y])
#
#     for node, adjacencies in enumerate(G.adjacency()):
#         node_trace['marker']['color'] += tuple([len(adjacencies[1])])
#         node_info = '# of connections: ' + str(len(adjacencies[1]))
#         node_trace['text'] += tuple([node_info])
#
#
#
# fig = go.Figure(data=[edge_trace, node_trace],
#              layout=go.Layout(
#                 title='<br>Network graph made with Python',
#                 titlefont=dict(size=16),
#                 showlegend=False,
#                 hovermode='closest',
#                 margin=dict(b=20,l=5,r=5,t=40),
#                 annotations=[ dict(
#                     text="Python code: <a href='https://plot.ly/ipython-notebooks/network-graphs/'> https://plot.ly/ipython-notebooks/network-graphs/</a>",
#                     showarrow=False,
#                     xref="paper", yref="paper",
#                     x=0.005, y=-0.002 ) ],
#                 xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
#                 yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)))
#
#
# py.iplot(fig, filename='networkx')

def plot_closeness(network):
   closeness = nx.closeness_centrality(network)
   x = list(closeness.keys())
   y = list(closeness.values())
   print(x)
   print(y)

   p.offline.plot({
       "data": [go.Scatter(x=x, y=y)],
       "layout": go.Layout(title="closeness")
   }, auto_open=True,  )

# calculate_degree_prob_distribution("test degree prob Distribution",3000,0.5)


