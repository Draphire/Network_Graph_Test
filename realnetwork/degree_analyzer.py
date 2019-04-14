
import plot_util
import math
import os
import time

import matplotlib.pyplot as plt
import networkx as nx

import path as path
from scalefreenetwork import plot_util

def count_degree(network):
    dic_list = {}
    for (node, val) in network.degree():
        dic_list[node] = val
    return dic_list

# def count_degree(network):
#     start_time = int(round(time.time() * 1000))
#     node_degree = {}
#     for key, value in network:
#         node_degree[key] = len(value)
#     end_time = int(round(time.time() * 1000))
#     compute_time = int(end_time) - int(start_time)
#     print('({}ms) Count the number of degree for each node'.format(compute_time))
#     print(node_degree)
#     return node_degree

'''
def plot_and_store_degree_prob_distribution(network_name, degree_count):
    file_name = network_name + '_degree_distribution_log_binning.png'
    file_path = os.path.join(PATH.DB_PLOT_DIR_PATH, file_name)

    n, bins = plot_util.log_binning(degree_count, n_bins=50, plot=False)
    bin_centers = list((bins[1:] + bins[:-1]) / 2)
    n = list(n)

    x_log, y_log = plot_util.get_log_log_points(bin_centers, n)
    plt.scatter(x_log, y_log, s=2, c='r')
    plt.title('Log-Log Degree Distribution with Log Binning')
    plt.xlabel('k')
    plt.ylabel('P(k)')
    plt.savefig(file_path)
    plt.close()

    return file_name
'''

def plot_and_store_degree_prob_distribution(network_name, degree_count):

    file_name = network_name + '_degree_distribution_log_binning.png'
    file_path = os.path.join(path.DB_PLOT_DIR_PATH, file_name)

    n, bins = plot_util.log_binning(degree_count, n_bins=50, plot=False)
    print(bins)
    print(n)
    bin_centers = list((bins[1:] + bins[:-1]) / 2)

    n = list(n)

    x_log, y_log = plot_util.get_log_log_points(bin_centers, n)
    plt.scatter(x_log, y_log, s=2, c='r')
    plt.title('Log-Log Degree Distribution with Log Binning')
    plt.xlabel('k')
    plt.ylabel('P(k)')
    plt.savefig(file_path)
    plt.show()
    plt.close()

    return file_name


def plot_and_store_degree_cc_prob_distribution(network_name, degree_count):

    file_name = network_name + '_degree_distribution_log_binning.png'
    file_path = os.path.join(path.DB_PLOT_DIR_PATH, file_name)

    n, bins = plot_util.log_binning(degree_count, n_bins=50, plot=False)
    bin_centers = list((bins[1:] + bins[:-1]) / 2)
    n = list(n)

    x_log, y_log = plot_util.get_log_log_points(bin_centers, n)
    plt.scatter(x_log, y_log, s=2, c='r')
    plt.title('Log-Log Degree Distribution with Log Binning')
    plt.xlabel('k')
    plt.ylabel('P(k)')
    plt.savefig(file_path)
    plt.show()
    plt.close()

    return file_name


def calculate_degree_distribution(network):
    return nx.degree_histogram(network)


def calculate_degree_prob_distribution(no_of_nodes, degree_distribution):
    for key in range(len(degree_distribution)):
        degree_distribution[key] = float(degree_distribution[key]) / no_of_nodes

    return degree_distribution

def calculate_degree_moment(degree_count, n=1):
    return sum([
        math.pow(c, n) for c in degree_count.values()
    ]) / len(degree_count.values())


def find_largest_degree(degree_distribution):
    return (len(degree_distribution)-1)


def find_smallest_degree(degree_distribution):
    j = 0
    for i in degree_distribution:
        if i != 0:
            return j
        else:
            j += 1

