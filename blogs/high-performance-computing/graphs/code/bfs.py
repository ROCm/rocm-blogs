from scipy.io import mmread
import matplotlib.pyplot as plt
import networkx as nx
import time
import argparse

parser = argparse.ArgumentParser(description='Run bfs on graph datasets.')

parser.add_argument("n", type=int, help="number of iterations")
parser.add_argument("s", type=int, help="starting node")
parser.add_argument('file_path', type=str, help="Path to mtx file")

args = parser.parse_args()

data = mmread(args.file_path)
graph = nx.Graph(data)
total_time = 0

for _ in range(args.n):
    start = time.time()
    bfs_tree = nx.bfs_tree(graph, source=args.s)
    end = time.time()
    elapsed = (end - start)
    total_time += elapsed

print("File path: %s" % args.file_path)
print("Starting node: %s" % args.s)
print("Number of iterations: %s" % args.n)
print("BFS CPU avg time: %s milliseconds" % ((total_time/args.n)*1000))