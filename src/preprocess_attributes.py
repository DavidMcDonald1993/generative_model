import numpy as np
import pandas as pd
import networkx as nx
import os

def read_graph(dir, edgelist_file):

	G = nx.read_edgelist(os.path.join(dir, edgelist_file))

	return G

def read_communities(dir, community_files, G):

	communities = {}
	for community_file in community_files:
		c = community_file.split(".")[0]
		with open(os.path.join(dir, community_file), "r") as f:
			for line in f:
				line_split = line.rstrip().split("\t")
				if c + line_split[0] in communities.keys():
					communities[c+line_split[0]] += line_split[1:]
				else:
					communities.update({c+line_split[0] : line_split[1:]})

	true_communities = {n : [] for n in G.nodes()}

	for i, (c, l) in enumerate(communities.items()):
		for n in set(l):
			if n in G.nodes():
				true_communities[n].append(i)

	max_len=0
	max_n = ""
	for n, l in true_communities.items():
		if len(l) > max_len:
			max_len = len(l)
			max_n = n

	for n, c in true_communities.items():
	
		while len(c) < max_len:
			c.append(-1)

	return pd.DataFrame.from_dict(true_communities, dtype=int).T

def read_attributes(dir, feature_files, feature_name_files):

	feature_name_maps = [{} for _ in range(len(feature_files))]
	
	for i, feature_name_file in enumerate(feature_name_files):
		
		with open(os.path.join(dir, feature_name_file), "r") as f:
			
			for line in f:
				line = line.rstrip()
				line_split = line.split(" ")
				feature_name_maps[i].update({line_split[0] : " ".join(line_split[1:])})
	
	dfs = []

	for i, feature_file in enumerate(feature_files):
		
		df = pd.read_csv(os.path.join(dir, feature_file), sep=" ", 
			index_col=0, header=None)

		df.columns = [feature_name_maps[i][str(int(f) - 1)] for f in df.columns]
		
		dfs.append(df)

	return pd.concat(dfs,).fillna(value=0)


def main():

	dir = "facebook"
	edgelist_file = "0.edges"
	community_files = [f for f in os.listdir(dir) if re.match("[A-Za-z0-9]*.circles", f)]
	feature_files = sorted([f for f in os.listdir('./facebook') if re.match("[0-9]+.feat$", f)])
	feature_name_files = sorted([f for f in os.listdir('./facebook') if re.match("[0-9]+.featnames$", f)])
	
	

	return

is __name__ == "__main__":
	main()