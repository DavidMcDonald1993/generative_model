import numpy as np
import pandas as pd
import networkx as nx

import powerlaw

def estimate_T():
	'''
	TODO
	'''
	return 0.1

def estimate_gamma(degrees):
	result = powerlaw.Fit(degrees)
	return result.power_law.alpha

def preprocess_G(gml_file):

	G = nx.read_gml(gml_file)
	# G = max(nx.connected_component_subgraphs(G), key=len)
# 
	N = nx.number_of_nodes(G)

	degree_dict = nx.degree(G)
	
	# node id
	order_of_appearance = np.array(sorted(degree_dict, key=degree_dict.get, reverse=True))
	nodes = np.array(G.nodes())
	# node index
	order_of_appearance = np.concatenate([np.where(nodes==n)[0] for n in order_of_appearance])

	degrees = np.array(degree_dict.values())

	# PS model parameters -- to estimate in real world network
	m = degrees.mean() / 2
	T = estimate_T()
	gamma = estimate_gamma(degrees)
	beta = 1 / (gamma - 1)

	# determine radial coordinates of nodes
	R = 2 * beta * np.log(range(1, N + 1)) + 2 * (1 - beta) * np.log(N) 
	R = R[order_of_appearance]

	# observed adjacency matrix
	A = np.array(nx.adjacency_matrix(G).todense())

	return G, N, R, A, order_of_appearance

def generate_X(G, N, K, attribute_file, order_of_appearance):

	attribute_df = pd.read_csv(attribute_file, sep=" ", header=None, index_col=0, dtype=np.float)

	# attribs = attribute_df.loc[sorted(order_of_appearance),:].iloc[:,0]
	attribs = attribute_df.iloc[:,0].values
	# print attribs

	# artificially create binary attributes from radial coordinates
	X = np.zeros((N, K))

	for i, att in enumerate(attribs):
		
		id = att * K / attribs.max()
		# print att, attribs.max(), id
		
		p = np.array([np.exp(-(id-x)**2 / (2 * (5.0)**2) ) for x in range(K)])
		# print p
		# print
		
		r = np.random.rand(K)
		# print r
		# print r<p
		
		X[i][r < p] = 1
		
	return X

G, N, _, _, order_of_appearance = preprocess_G("email_graph.gml")

# print G.nodes()
K = 50

X = generate_X(G, N, K, "email-Eu-core-department-labels.txt.gz", order_of_appearance)

print X.shape

X = pd.DataFrame(X)

X.to_csv("email_graph_attributes.csv")

X = pd.read_csv("email_graph_attributes.csv", index_col=0).values

print X[0]