import os

import matplotlib
matplotlib.use("Agg")

import argparse

import numpy as np
import scipy as sp
import networkx as nx
import pandas as pd

from sys import stdout

from functools import partial
# from multiprocessing.pool import ThreadPool as Pool
# from multiprocessing import Pool

from sklearn.metrics import normalized_mutual_info_score as NMI

import powerlaw

import matplotlib.pyplot as plt

from keras import backend as K
from keras.engine.topology import Layer
from keras.initializers import uniform
from keras.models import Model
from keras.layers import Input, Dense
from keras import regularizers
from keras import constraints
from keras.regularizers import l1
from keras.constraints import NonNeg
from keras.optimizers import Adam

class ThetaLookupLayer(Layer):

	def __init__(self, **kwargs):
		self.output_dim=1
		super(ThetaLookupLayer, self).__init__(**kwargs)

	def build(self, input_shape):
		# Create a trainable weight variable for this layer.
		self.kernel = self.add_weight(name='theta_lookup_matrix', 
									  shape=(input_shape[1], self.output_dim),
									  initializer=uniform(minval=0, maxval=6),
									  trainable=True)
		super(ThetaLookupLayer, self).build(input_shape)  # Be sure to call this somewhere!

	def call(self, x):
		return K.dot(x, self.kernel)

	def compute_output_shape(self, input_shape):
		return (input_shape[0], self.output_dim)
	
class FLayer(Layer):

	def __init__(self, output_dim, activity_regularizer=None, kernel_constraint=None, **kwargs):
		self.output_dim = output_dim
		self.activity_regularizer = activity_regularizer
		self.kernel_constraint = kernel_constraint
		super(FLayer, self).__init__(**kwargs)

	def build(self, input_shape):
		# Create a trainable weight variable for this layer.
		self.kernel = self.add_weight(name='M', 
									  shape=(3, self.output_dim),
									  initializer=uniform(minval=0, maxval=6), 
									  constraint=self.kernel_constraint,
									  trainable=True)
		super(FLayer, self).build(input_shape)  # Be sure to call this somewhere!

	def call(self, (thetas, R)):
		delta_theta = np.pi - K.abs(np.pi - K.abs(thetas - self.kernel[1]))
		delta_theta = K.maximum(1e-8, delta_theta)
		H =  R + self.kernel[0] + 2 * K.log(delta_theta / 2)
		# H =  R + self.kernel[0] + delta_theta
		F = K.exp(- 0.5 * K.square(H) / K.square(self.kernel[2]))
		return F

	def compute_output_shape(self, input_shape):
		return (None, self.output_dim)

	def get_config(self):
		config = {
			'activity_regularizer': regularizers.serialize(self.activity_regularizer),
			'kernel_constraint': constraints.serialize(self.kernel_constraint),
		}
		base_config = super(FLayer, self).get_config()
		return dict(list(base_config.items()) + list(config.items()))
	
class PLayer(Layer):

	def __init__(self, **kwargs):
		self.output_dim = 1
		super(PLayer, self).__init__(**kwargs)

	def build(self, input_shape):
		super(PLayer, self).build(input_shape)  # Be sure to call this somewhere!

	def call(self, (F_u, F_v)):
		P = 1 - K.exp( - K.sum( F_u * F_v, axis=1 ))
		return K.reshape(P, (-1, 1))

	def compute_output_shape(self, input_shape):
		return (None, self.output_dim)

def build_model(N, K, C, lamb_F=1e-2, lamb_W=1e-2, alpha=0.5, attribute_type="binary"):

	
	u = Input(shape=(N,))
	v = Input(shape=(N,))
	
	r_u = Input(shape=(1,))
	r_v = Input(shape=(1,))

	theta_lookup = ThetaLookupLayer()
	
	theta_u = theta_lookup(u)
	theta_v = theta_lookup(v)
	
	F = FLayer(C, activity_regularizer=l1(lamb_F), kernel_constraint=NonNeg())
	
	F_u = F([theta_u, r_u])
	F_v = F([theta_v, r_v])
	
	P_uv = PLayer()([F_u, F_v])
	
	if attribute_type == "binary":
		activation = "sigmoid"
	else:
		activation = "linear"
	
	Q = Dense(K, name="Q", activation=activation, kernel_regularizer=l1(lamb_W), bias_regularizer=l1(lamb_W))
	
	Q_u = Q(F_u)
	Q_v = Q(F_v)

	loss = ["binary_crossentropy"]
	if attribute_type == "binary":
		loss += ["binary_crossentropy"] * 2
	else:
		loss += ["mse"] * 2

	trainable_model = Model([u, v, r_u, r_v], [P_uv, Q_u, Q_v], name="trainable_model")


	adam = Adam(clipnorm=1.0)
	trainable_model.compile(optimizer=adam, loss=loss, 
		loss_weights=[1-alpha, alpha, alpha], )

	community_assignment_model = Model([u, r_u], F_u, name="community_assignment_model")
	
	return trainable_model, community_assignment_model

def save_trained_models(models):

	for model in models:
		model_json = model.to_json()

		with open("models/{}.json".format(model.name), "w") as f:
			f.write(model_json)

		model.save_weights("models/{}_weights.h5".format(model.name))

def input_pattern_generator(N, R, A, X, batch_size=100):
	
	I = np.identity(N)
	
	while True:
		
		U = np.random.choice(N, replace=True, size=(batch_size,))
		V = np.random.choice(N, replace=True, size=(batch_size,))
		
		yield [I[U], I[V], R[U], R[V]], [A[U, V].T, X[U].todense(), X[V].todense()]

def train_model(N, R, A, X, trainable_model, community_assignment_model, 
	num_epochs=10000, batch_size=100, true_communities=None):
	
	generator = input_pattern_generator(N, R, A, X, batch_size)

	for epoch in range(num_epochs):
		trainable_model.fit_generator(generator, steps_per_epoch=1000, epochs=1, verbose=1)
		if true_communities is not None:
			community_predictions = community_assignment_model.predict([np.identity(N), R])
			community_membership_predictions = np.argmax(community_predictions, axis=1)
			stdout.write("NMI: {}\n".format(NMI(true_communities, community_membership_predictions)))
			stdout.flush()

def estimate_T():
	'''
	TODO
	'''
	return 0.1

def estimate_gamma(degrees):
	'''
	TODO
	'''
	result = powerlaw.Fit(degrees)
	return result.power_law.alpha

def preprocess_G(gml_file, gamma, T):

	G = nx.read_gml(gml_file)
	G = max(nx.connected_component_subgraphs(G), key=len)

	N = nx.number_of_nodes(G)

	degree_dict = nx.degree(G)
	
	# node id
	order_of_appearance = np.array(sorted(degree_dict, key=degree_dict.get, reverse=True))
	nodes = np.array(G.nodes())
	# node index
	order_of_appearance = np.concatenate([np.where(order_of_appearance==n)[0]
		for n in nodes])

	degrees = np.array(degree_dict.values())

	# PS model parameters -- to estimate in real world network
	m = degrees.mean() / 2
	if T == None:
		stdout.write("T is not given, estimating it\n")
		T = estimate_T()
	if gamma == None:
		stdout.write("gamma is not given, estimating it\n")
		gamma = estimate_gamma(degrees)
	beta = 1 / (gamma - 1)

	stdout.write("m={}, T={}, gamma={}, beta={}\n".format(m, T, gamma, beta))

	# determine radial coordinates of nodes
	R = 2 * beta * np.log(range(1, N + 1)) + 2 * (1 - beta) * np.log(N) 
	R = np.matrix(R[order_of_appearance]).T

	# observed adjacency matrix
	A = nx.adjacency_matrix(G)

	L = nx.laplacian_matrix(G).asfptype()

	return nodes, N, R, A, L

def preprocess_X(nodes, attribute_file):
	attr_df = pd.read_csv(attribute_file, index_col=0,)
	return sp.sparse.csr_matrix(attr_df.iloc[nodes, :].values)

def preprocess_true_communities(nodes, true_community_file):
	if true_community_file == None:
		return None
	community_df = pd.read_csv(true_community_file, header=None, index_col=0, sep=" ")
	return community_df.iloc[nodes, 0].values

def draw_network(N, C, R, thetas, M, e, L_G, L_X, plot_directory):

	_, h = hyperbolic_distance(R, thetas, M)
	F = compute_F(h, M)
	assignments = F.argmax(axis=1).A1
	assignment_strength = F[np.arange(N), assignments].A1

	node_cartesian = np.column_stack([np.multiply(R, np.cos(thetas)),
	np.multiply(R, np.sin(thetas))])
	community_cartesian = np.column_stack([np.multiply(M[0], np.cos(M[1])).T, 
		np.multiply(M[0], np.sin(M[1])).T])

	plt.figure(figsize=(15, 15))
	plt.title("Epoch={}, L_G={}, L_X={}".format(e, L_G, L_X))
	plt.scatter(node_cartesian[:,0].A1, node_cartesian[:,1].A1, 
		c=assignments, s=100*assignment_strength)
	plt.scatter(community_cartesian[:,0].A1, community_cartesian[:,1].A1, 
		c=np.arange(C), s=100)
	plt.scatter(community_cartesian[:,0].A1, community_cartesian[:,1].A1, 
		c="k", s=25)
	# plt.show()
	plt.savefig(os.path.join(plot_directory, "epoch_{}.png".format(e)))
	plt.close()

def parse_args():

	parser = argparse.ArgumentParser(description="Embed complex network to hyperbolic space.")
	parser.add_argument("gml_file", metavar="gml_file_path",
						help="path of gml file of network")
	parser.add_argument("attribute_file", metavar="attribute_file", 
						help="path of attribute file")
	parser.add_argument("num_communities", metavar="C", type=int,
					help="number of communities")
	parser.add_argument("--T", dest="T", type=np.float,
					help="network temperature (if this is not given, it is estimated)", default=None)
	parser.add_argument("--gamma", dest="gamma", type=np.float,
					help="network scaling exponent (if this is not given, it is estimated)",
					default=None)
	parser.add_argument("--attribute_type", dest="attribute_type", 
					help="type of attribute (default is binary)", default="binary")
	parser.add_argument("-e", dest="num_epochs", type=int,
					help="number of epochs to train for (default is 10000)", default=10000)
	parser.add_argument("-b", dest="batch_size", type=int,
					help="minibatch_size (default is 100)", default=100)
	parser.add_argument("--lamb_F", dest="lamb_F", type=np.float,
					help="l1 penalty on F (default is 0.01) (default is 1e-2)", default=1e-2)
	parser.add_argument("--lamb_W", dest="lamb_W", type=np.float,
					help="l1 penalty on W (default is 0.01) (default is 1e-2)", default=1e-2)
	parser.add_argument("--alpha", dest="alpha", type=np.float,
					help="weighting of likelihoods (default is 0.5)", default=0.5)
	parser.add_argument("-c", dest="true_communities", 
				help="path of csv file containing ground truth community memberships", default=None)
	parser.add_argument("-p", dest="num_processes", type=np.int,
				help="number of parallel processes", default=None)
	parser.add_argument("--thetas", dest="thetas_filepath",
			help="filepath of trained thetas vector (default is \"thetas.csv\")", default="thetas.csv")
	parser.add_argument("--M", dest="M_filepath",
			help="filepath of trained M matrix (default is \"M.csv\")", default="M.csv")
	parser.add_argument("--W", dest="W_filepath",
			help="filepath of trained W matrix (default is \"W.csv\")", default="W.csv")
	parser.add_argument("--F", dest="F_filepath",
			help="filepath of trained F matrix (default is \"F.csv\")", default="F.csv")
	parser.add_argument("--plot", dest="plot_directory",
				help="path of directory to save plots")


	args = parser.parse_args()

	return args

# @profile
def main():

	args = parse_args()

	gml_file = args.gml_file
	gamma = args.gamma
	T = args.T
	stdout.write("Reading G from {} with scaling exponent={} and T={}\n".format(gml_file, gamma, T))
	nodes, N, R, A, L = preprocess_G(gml_file, gamma, T)
	stdout.write("Preprocessed G\n")

	# X, true_communities = generate_X(N, K, attribute_file, order_of_appearance)
	attribute_file = args.attribute_file
	stdout.write("Reading attributes from {}\n".format(attribute_file))
	X = preprocess_X(nodes, attribute_file)
	stdout.write("Preprocessed X\n")

	K = X.shape[1]
	stdout.write("K={}\n".format(K))

	C = args.num_communities
	stdout.write("C={}\n".format(C))
	
	alpha = args.alpha
	lamb_F = args.lamb_F
	lamb_W = args.lamb_W
	attribute_type = args.attribute_type

	trainable_model, community_assignment_model = build_model(N, K, C, lamb_F, lamb_W, alpha, attribute_type)
	stdout.write("Built model\n")
	
	num_epochs = args.num_epochs
	batch_size = args.batch_size

	
	true_communities = preprocess_true_communities(nodes, args.true_communities)

	plot_directory = args.plot_directory

	stdout.write("num_epochs={}, lamb_F={}, lamb_W={}, alpha={}, attribute_type={}".format(num_epochs, 
		lamb_F, lamb_W, alpha, attribute_type))
	# stdout.write("saving plots to {}\n".format(plot_directory))
	stdout.flush()

	train_model(N, R, A, X, trainable_model, community_assignment_model, num_epochs, batch_size, true_communities)

	stdout.write("Trained matrices\n")

	thetas = trainable_model.layers[2].get_weights()[0]
	M = trainable_model.layers[5].get_weights()[0]
	W = np.vstack(trainable_model.layers[7].get_weights())
	F = community_assignment_model.predict([np.identity(N), R])

	thetas = pd.DataFrame(thetas)
	M = pd.DataFrame(M)
	W = pd.DataFrame(W)
	F = pd.DataFrame(F)

	thetas_filepath = args.thetas_filepath
	M_filepath = args.M_filepath
	W_filepath = args.W_filepath
	F_filepath = args.F_filepath

	thetas.to_csv(thetas_filepath, sep=",", index=False, header=False)
	M.to_csv(M_filepath, sep=",", index=False, header=False)
	W.to_csv(W_filepath, sep=",", index=False, header=False)
	F.to_csv(F_filepath, sep=",", index=False, header=False)

	stdout.write("Written trained matrices to file\n")

	save_trained_models([trainable_model, community_assignment_model])

	stdout.write("Saved model\n")

	return

if __name__ == "__main__":
	main()