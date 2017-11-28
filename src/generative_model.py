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
from multiprocessing import Pool

from sklearn.metrics import normalized_mutual_info_score as NMI

import powerlaw

from sklearn.cluster import KMeans

import matplotlib.pyplot as plt

# clip value to avoid taking log of 0 
clip_value = 1e-8

def sigmoid(x):
	'''
	Compute the elementwise sigmoid activation of matrix x
	'''
	return 1 / (1 + np.exp(-x))

def compute_L_G(A, P):
	
	'''
	compute likelihood of observing adjacacy matrix A
	'''
	
	return -(A.multiply(np.log(P)) + 
		np.log(1 - P) - A.multiply(np.log(1 - P))).mean()

def compute_L_X(X, Q, attribute_type):

	'''
	compute likelihood of observing attribute matrix X
	'''

	if attribute_type != "binary":
		return (0.5 * np.square(X - Q)).mean()  # for real valued attributes 
	
	# binary attributes
	return -(X.multiply(np.log(Q)) + 
		np.log(1 - Q) - X.multiply(np.log(1 - Q))).mean()

def compute_likelihood(A, X, N, K, R, thetas, M, W, lamb_F, lamb_W, alpha, attribute_type):
	
	'''
	compute overall likelood of observing A and X, weighted by alpha
	also includes l1 penalty on F and W
	'''

	_, h = hyperbolic_distance(R, thetas, M)
	F = compute_F(h, M)
	P = compute_P(F)

	# likelihood of G
	L_G = compute_L_G(A, P)

	# l1 penalty term for matrix F
	l1_F = lamb_F * np.linalg.norm(F, axis=0, ord=1).sum()

	# append columns of ones as C+1th column of W is the bias term
	F = np.column_stack([F, np.ones(N)])
	Q = compute_Q(F, W, attribute_type)
	
	# likelihood of X
	L_X = compute_L_X(X, Q, attribute_type)
	
	# l1 penalty on W
	l1_W = lamb_W * np.linalg.norm(W, axis=0, ord=1).sum()
	
	# overall likelihood
	likelihood = (1 - alpha) * L_G + alpha * L_X + l1_F + l1_W
	
	return L_G, L_X, l1_F, l1_W, likelihood

def hyperbolic_distance(R, thetas, M):
	'''
	computes hyperbolic distance between nodes gievn as (R, thetas)
	and communities given as (M[0], M[1])
	'''
	delta_theta = np.pi - abs(np.pi - abs(thetas - M[1]))
	if (delta_theta / 2 < 0).any():
		print "2nfgoewrngoe"
		i, j = np.where(delta_theta / 2 < 0)
		print i
		print j
		print thetas[i]
		print delta_theta[i]
		return
	# relaxation of hyperbolic law of cosines
	h = R + M[0] + 2 * np.log(delta_theta / 2)
	return delta_theta, h

def compute_F(h, M):
	'''
	compute F matrix from hyperbolic distances h and M
	'''
	F = np.exp(- np.square(h) / (2 * np.square(M[2])))
	# F = 1 / np.sqrt(2 * np.pi * M[:, 2] ** 2) * F
	return F

def compute_P(F):
	'''
	compute probabilities of connections bwteeen all nodes
	'''
	P = 1 - np.exp(-F.dot(F.T))
	return np.clip(P, a_min=clip_value, a_max=1-clip_value)

def compute_P_u(u, F):
	'''
	only compute uth row of P
	'''
	P_u = 1 - np.exp(-F[u].dot(F.T))
	return np.clip(P_u, a_min=clip_value, a_max=1-clip_value)

def compute_Q(F, W, attribute_type):
	'''
	compute probability of nodes possessing attributes
	'''
	Q = F.dot(W.T)
	if attribute_type != "binary":
		return Q
	# if we have binary attributes, then we should apply sigmoid transformation 
	Q = sigmoid(Q)
	return np.clip(Q, a_min=clip_value, a_max=1-clip_value)

def compute_Q_u(F_u, W, attribute_type):
	'''
	uth row of Q
	'''
	Q_u = F_u.dot(W.T)
	if attribute_type != "binary":
		return Q_u
	Q_u = sigmoid(Q_u)
	return np.clip(Q_u, a_min=clip_value, a_max=1-clip_value)

def compute_Q__k(F, W__k, attribute_type):
	'''
	kth column of Q
	'''
	Q__k = F.dot(W__k)
	if attribute_type != "binary":
		return Q__k
	Q__k = sigmoid(Q__k)
	return np.clip(Q__k, a_min=clip_value, a_max=1-clip_value)

def update_theta_u(u, N, K, C, A, X, R, thetas, M, W, alpha, lamb_F, attribute_type):

	'''
	compute update for angular coo-ordinate of node u

	'''

	# prelimaries to compute uth row of P matrix	
	delta_theta, h = hyperbolic_distance(R, thetas, M)
	F = compute_F(h, M)
	P_u = compute_P_u(u, F)

	
	# compute derivative of delta theta with respect to theta
	partial_delta_theta_u_partial_theta_u = np.multiply(-np.sign(np.pi - abs(thetas[u] - M[1])),
	 -np.sign(thetas[u] - M[1])).T

	# norm = np.linalg.norm(partial_delta_theta_u_partial_theta_u)
	# if norm > 1:
	# 	partial_delta_theta_u_partial_theta_u /= norm

	'''
	derivative of uth row of F with repect to delta theta
	multiply dF_uc/dh_uc with dh_uc/delta_theta_uc and ocnvert to 
	sparse diagonal matrix 
	'''

	partial_F_u_partial_delta_theta_u = np.multiply(np.multiply(-h[u] / np.square(M[2]), 
		F[u, :]), 4 / delta_theta[u])
	# norm = np.linalg.norm(partial_F_u_partial_delta_theta_u)
	# if norm > 1:
	# 	partial_F_u_partial_delta_theta_u /= norm
	partial_F_u_partial_delta_theta_u = sp.sparse.spdiags(partial_F_u_partial_delta_theta_u, 0,
		partial_F_u_partial_delta_theta_u.shape[1], partial_F_u_partial_delta_theta_u.shape[1])

	'''
	derivative of uth row of P with respect to uth row of F

	dP_uv/dF_uc = exp(-F_u F_v) * F_vc

	'''
	partial_P_u_partial_F_u = np.multiply(np.exp(-F[u].dot(F.T)).T, F)
	# norm = np.linalg.norm(partial_P_u_partial_F_u)
	# if norm > 1:
	# 	partial_P_u_partial_F_u /= norm


	# partial L_G 
	A_u = A[u]
	partial_L_G_u_partial_P_u = - 1.0 / N * \
	(A_u / P_u - 1 / (1 - P_u) + A_u / (1 - P_u))
	# norm = np.linalg.norm(partial_L_G_u_partial_P_u)
	# if norm > 1:
	# 	partial_L_G_u_partial_P_u /= norm

	# dot products to combine partial derivatives

	partial_L_G_u_partial_F_u = partial_L_G_u_partial_P_u\
	.dot(partial_P_u_partial_F_u)

	partial_F_u_partial_theta_u = partial_F_u_partial_delta_theta_u\
	.dot(partial_delta_theta_u_partial_theta_u)

	partial_L_G_u_partial_theta_u = partial_L_G_u_partial_F_u\
	.dot(partial_F_u_partial_theta_u)

	# derivative of abs(F_u) is sign(F_u)
	partial_l1_F_u_partial_F_u = np.sign(F[u])
	# norm = np.linalg.norm(partial_l1_F_u_partial_F_u)
	# if norm > 1:
	# 	partial_l1_F_u_partial_F_u /= norm

	# partial L_X
	F = np.append(F[u], np.matrix(1), axis=1)
	Q_u = compute_Q_u(F, W, attribute_type)

	'''
	dL_X / d_F_uc = (X_u - Q_u).dot(W__c)
	'''

	partial_L_X_u_partial_F_u = - 1.0 / N *\
	(X[u] - Q_u).dot(W[:,:-1])

	# norm = np.linalg.norm(partial_L_X_u_partial_F_u)
	# if norm > 1:
	# 	partial_L_X_u_partial_F_u /= norm

	partial_L_X_u_partial_theta_u = partial_L_X_u_partial_F_u.dot(partial_F_u_partial_theta_u)


	grad = ((1 - alpha) * partial_L_G_u_partial_theta_u + alpha * partial_L_X_u_partial_theta_u\
		+ lamb_F * partial_l1_F_u_partial_F_u.dot(partial_F_u_partial_theta_u))

	norm = np.linalg.norm(grad)

	if norm > 1:
		grad /= norm

	# if abs(-0.01*((1 - alpha) * partial_L_G_u_partial_theta_u + alpha * partial_L_X_u_partial_theta_u\
	# 		+ lamb_F * partial_l1_F_u_partial_F_u.dot(partial_F_u_partial_theta_u)).A1[0]) > np.pi/2:
	# 	stdout.write("TOO BIG\n")
	# 	print u, -0.01*((1 - alpha) * partial_L_G_u_partial_theta_u + alpha * partial_L_X_u_partial_theta_u\
	# 		+ lamb_F * partial_l1_F_u_partial_F_u.dot(partial_F_u_partial_theta_u)).A1
	# 	print
	# 	print delta_theta[u]
	# 	print 4/delta_theta[u]
	# 	# print np.sign(np.pi - abs(thetas[u] - M[:,1]))
	# 	# print np.sign(thetas[u] - M[:,1])
	# 	# print thetas[u] - M[:,1]
	# 	# print M
	# 	# print thetas[u]
	# 	print 
	# 	print P_u
	# 	print 1 / P_u

	# 	print 
	# 	print partial_delta_theta_u_partial_theta_u
	# 	print partial_F_u_partial_delta_theta_u
	# 	print partial_F_u_partial_theta_u
	# 	# print partial_L_G_u_partial_P_u.min()
	# 	# # print A_u[0,0]
	# 	# # print P_u[0]
	# 	# # print 1 - A_u[0,0]
	# 	# # print 1 - P_u[0]
	# 	# print partial_P_u_partial_F_u.min()
	# 	print partial_L_G_u_partial_F_u

	# 	print partial_L_G_u_partial_theta_u
	# 	print 
	# 	print partial_L_X_u_partial_theta_u
	# 	# return np.sign(-0.01*((1 - alpha) * partial_L_G_u_partial_theta_u + alpha * partial_L_X_u_partial_theta_u\
	# 	# 	+ lamb_F * partial_l1_F_u_partial_F_u.dot(partial_F_u_partial_theta_u)).A1[0]) * 0.01
	# 	return 

	# print -1 * ((1 - alpha) * partial_L_G_u_partial_theta_u + alpha * partial_L_X_u_partial_theta_u\
	# 		+ lamb_F * partial_l1_F_u_partial_F_u.dot(partial_F_u_partial_theta_u)).A1

	return grad


def compute_partial_P_partial_F_c(c, N, A, P, F):

	'''
	helper function to compute 3-dimension tensor representing dP/d_F_c
	(here represented as 2d sparse matrix of size (N**2, N))

	
					 exp(-F_u F_v) * F_uc if u' == v	
	dp_uv/df_u'c = { exp(-F_u F_v) * F_vc if u' == u
					 0 otherwise

	'''

	partial_P_partial_F_c = sp.sparse.lil_matrix((N, N**2))
	'''
	matrices U, and V represent all the us and vs repspectively 
	'''
	U = np.repeat(np.arange(N), 2*N)
	V = np.tile(np.repeat(np.arange(N), 2), N)
	U_prime = np.column_stack([np.repeat(np.arange(N), N), np.tile(np.arange(N), N)])

	# U_even = U[np.arange(U.shape[0]) % 2 == 0]
	# V_even = V[np.arange(V.shape[0]) % 2 == 0]
	# store exp(-F_u F_v) for all u, v
	exp = np.exp(-F.dot(F.T))

	partial_P_partial_F_c[U_prime.reshape(-1,), U * N + V] = \
	np.multiply(F[U_prime[:,::-1].reshape(1, -1), c], exp[U, V])

	# norm = np.linalg.norm(partial_P_partial_F_c, axis=1)[:,None]
	# partial_P_partial_F_c /= norm

	return sp.sparse.csr_matrix(partial_P_partial_F_c)

def update_community_r_c(c, N, K, A, X, R, thetas, M, W, alpha, lamb_F, attribute_type):
	
	'''
	compute update for radial coordinate of community c
	'''
	# compute F
	delta_theta, h = hyperbolic_distance(R, thetas, M)
	F = compute_F(h, M)
	P = compute_P(F)

	'''
	dh_uc / d_rc = 1
	'''

	partial_h_c_partial_r_c = np.matrix(np.ones((N, 1)))

	'''
	dF_uc / dh_u'c = -h_uc / sd_c ** 2 * F_uc if u' == u 

	'''
	partial_F_c_partial_h_c = np.multiply(-h[:, c] / np.square(M[2, c]), F[:,c]).T

	partial_F_c_partial_h_c = sp.sparse.spdiags(partial_F_c_partial_h_c, 0,
		partial_F_c_partial_h_c.shape[1], partial_F_c_partial_h_c.shape[1])

	'''
	dL_G / d_P_uv = A_uv / P_uv - (1 - A_uv) / (1 - P_uv)
	reshape into row matrix (1, N**2) for easy dot product with shape (N**2, N)
	'''
	
	partial_L_G_partial_P = - 1.0 / N * \
	(A / P - 1 / (1 - P) + A / (1 - P)).reshape(1, -1)

	partial_P_partial_F_c = compute_partial_P_partial_F_c(c, N, A, P, F)
	partial_L_G_c_partial_F_c = partial_P_partial_F_c.multiply(partial_L_G_partial_P).sum(axis=1).T
	
	partial_F_c_partial_r_c = partial_F_c_partial_h_c.dot(partial_h_c_partial_r_c)
	partial_L_G_c_partial_r_c = partial_L_G_c_partial_F_c.dot(partial_F_c_partial_r_c)

	# derivative of abs is sign
	partial_l1_F_c_partial_F_c = np.sign(F[:,c]).T
	
	# partial L_x 
	F = np.column_stack([F, np.ones(N)])
	Q = compute_Q(F, W, attribute_type)

	partial_L_X_c_partial_F_c = - 1.0 / N * \
	(X - Q).dot(W[:,c]).T
	# np.array([(X[u] - Q[u]).dot(W[:,c]) for u in range(N)])

	partial_L_X_c_partial_r_c = partial_L_X_c_partial_F_c.dot(partial_F_c_partial_r_c)

	# print M[c, 0]
	# print -0.01 * ((1 - alpha) * partial_L_G_c_partial_r_c + alpha * partial_L_X_c_partial_r_c\
	#  + lamb_F * partial_l1_F_c_partial_F_c.dot(partial_F_c_partial_r_c)).A1

	grad = ((1 - alpha) * partial_L_G_c_partial_r_c + alpha * partial_L_X_c_partial_r_c\
	 + lamb_F * partial_l1_F_c_partial_F_c.dot(partial_F_c_partial_r_c))

	norm = np.linalg.norm(grad)

	if norm > 1:
		grad /= norm

	return grad

def update_community_theta_c(c, N, K, A, X, R, thetas, M, W, alpha, lamb_F, attribute_type):
	
	# compute F
	delta_theta, h = hyperbolic_distance(R, thetas, M)
	F = compute_F(h, M)
	P = compute_P(F)
	
	# partial delta theta
	partial_delta_theta_c_partial_theta_c = - np.multiply(-np.sign(np.pi - abs(thetas - M[1, c])), 
		-np.sign(thetas - M[1, c]))
	

	partial_F_c_partial_delta_theta_c = np.multiply(np.multiply(-h[:, c] / np.square(M[2, c]), 
		F[:,c]), 4 / delta_theta[:,c]).T

	# norm = np.linalg.norm(partial_F_c_partial_delta_theta_c)
	# if norm > 1:
	# 	partial_F_c_partial_delta_theta_c /= norm

	partial_F_c_partial_delta_theta_c = sp.sparse.spdiags(partial_F_c_partial_delta_theta_c, 0,
		partial_F_c_partial_delta_theta_c.shape[1], partial_F_c_partial_delta_theta_c.shape[1])
	
	partial_F_c_partial_theta_c = partial_F_c_partial_delta_theta_c.dot(partial_delta_theta_c_partial_theta_c)

	partial_L_G_partial_P = - 1.0 / N * \
	(A / P - 1 / (1 - P) + A / (1 - P)).reshape(1, -1)

	partial_P_partial_F_c = compute_partial_P_partial_F_c(c, N, A, P, F)
	partial_L_G_c_partial_F_c = partial_P_partial_F_c.multiply(partial_L_G_partial_P).sum(axis=1).T

	partial_L_G_c_partial_theta_c = partial_L_G_c_partial_F_c.dot(partial_F_c_partial_theta_c)

	partial_l1_F_c_partial_F_c = np.sign(F[:,c]).T
	
	# LX
	F = np.column_stack([F, np.ones(N)])
	Q = compute_Q(F, W, attribute_type)

	# partial_L_X_c_partial_F_c = 1.0 / (N * K) *\
	partial_L_X_c_partial_F_c = - 1.0 / N * \
	(X - Q).dot(W[:,c]).T
	# np.array([(X[u] - Q[u]).dot(W[:,c]) for u in range(N)])

	partial_L_X_c_partial_theta_c = partial_L_X_c_partial_F_c.dot(partial_F_c_partial_theta_c)

	# print -0.01 * ((1 - alpha) * partial_L_G_c_partial_theta_c + alpha * partial_L_X_c_partial_theta_c\
	# 	+ lamb_F * partial_l1_F_c_partial_F_c.dot(partial_F_c_partial_theta_c)).A1

	# if 0.01 * ((1 - alpha) * partial_L_G_c_partial_theta_c + alpha * partial_L_X_c_partial_theta_c\
	# 	+ lamb_F * partial_l1_F_c_partial_F_c.dot(partial_F_c_partial_theta_c)) > np.pi / 2:
	# 	stdout.write("TOOBIG COMMUNITY\n")
	# 	return

	grad = ((1 - alpha) * partial_L_G_c_partial_theta_c + alpha * partial_L_X_c_partial_theta_c\
		+ lamb_F * partial_l1_F_c_partial_F_c.dot(partial_F_c_partial_theta_c))
	norm = np.linalg.norm(grad)

	if norm > 1:
		grad /= norm

	return grad

def update_community_sd_c(c, N, K, A, X, R, thetas, M, W, alpha, lamb_F, attribute_type):
	
	# compute F

	delta_theta, h = hyperbolic_distance(R, thetas, M)
	F = compute_F(h, M)
	P = compute_P(F)
	# partial F
	partial_F_c_partial_sd_c = np.multiply(np.square(h[:, c]) / np.power(M[2, c], 3), F[:,c])
	# partial_F_c_partial_sd_c = F[:,c] * (h[:,c] ** 2 / M[c,2] ** 3 - 1 / M[c,2])


	partial_L_G_partial_P = - 1.0 / N * \
	(A / P - 1 / (1 - P) + A / (1 - P)).reshape(1, -1)

	partial_P_partial_F_c = compute_partial_P_partial_F_c(c, N, A, P, F)
	partial_L_G_c_partial_F_c = partial_P_partial_F_c.multiply(partial_L_G_partial_P).sum(axis=1).T

	partial_L_G_c_partial_sd_c = partial_L_G_c_partial_F_c.dot(partial_F_c_partial_sd_c)

	partial_l1_F_c_partial_F_c = np.sign(F[:,c]).T
	
	# partial L_x 
	F = np.column_stack([F, np.ones(N)])
	Q = compute_Q(F, W, attribute_type)

	# partial_L_X_c_partial_F_c = 1.0 / (N * K) *\
	partial_L_X_c_partial_F_c = - 1.0 / N * \
	(X - Q).dot(W[:,c]).T

	partial_L_X_c_partial_sd_c = partial_L_X_c_partial_F_c.dot(partial_F_c_partial_sd_c)

	# print -0.01 * ((1 - alpha) * partial_L_G_c_partial_sd_c + alpha * partial_L_X_c_partial_sd_c\
	# 	+ lamb_F * partial_l1_F_c_partial_F_c.dot(partial_F_c_partial_sd_c)).A1

	grad = ((1 - alpha) * partial_L_G_c_partial_sd_c + alpha * partial_L_X_c_partial_sd_c\
		+ lamb_F * partial_l1_F_c_partial_F_c.dot(partial_F_c_partial_sd_c))
	norm = np.linalg.norm(grad)
	if norm > 1:
		grad /= norm
	return grad


def update_W_k(k, N, K, C, X, W, F, lamb_W, alpha, attribute_type):
	
	# kth row of W
	W_k = W[k]
	# print k
	# print W_k
	
	# kth column of Q
	Q__k = compute_Q__k(F, W_k.T, attribute_type)
	# Q = compute_Q(F, W, attribute_type)

	# partial_L_X_k_partial_Q_k = 1.0 / N * np.array([X[u, k] / Q[u, k] - (1 - X[u, k]) / (1 - Q[u, k])
	# 	for u in range(N)])
	# partial_Q_k_partial_W_k = np.array([[Q[u, k] * (1 - Q[u, k]) * F[u, c] 
	# 	for c in range(C+1)] for u in range(N)])
	# partial_L_X_k_partial_W_k = partial_L_X_k_partial_Q_k.dot(partial_Q_k_partial_W_k)

	# partial_L_X_k_partial_W_k = 1.0 / N *\
	partial_L_X_k_partial_W_k = - 1.0 / N * \
	(X.T[k] - Q__k.T).dot(F)
	# print partial_L_X_k_partial_W_k.shape
	# np.array([(X[:, k] - Q__k).dot(F[:, c]) for c in range(C + 1)])

	# print (X[:,k] - Q__k)
	# print partial_L_X_k_partial_W_k 

	partial_l1_W_k_partial_W_k = np.sign(W_k)
	# print X.T[k].T.shape
	# print Q__k.shape
	# print (X.T[k] - Q__k).shape
	# print partial_L_X_k_partial_W_k.shape
	# print partial_l1_W_k_partial_W_k.shape
	# print (alpha * partial_L_X_k_partial_W_k - lamb_W * partial_l1_W_k_partial_W_k).shape

	grad = alpha * partial_L_X_k_partial_W_k + lamb_W * partial_l1_W_k_partial_W_k
	norm = np.linalg.norm(grad)
	if norm > 1:
		grad /= norm

	return grad

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

def initialize_matrices(L, N, C, K, R):

	sigma = 10
	community_radii = R.mean()
	noise = 1e-2
	# community matrix M
	M = np.matrix(np.zeros((3, C)))
	# centre radii
	M[0] = np.random.normal(size=(1, C), loc=community_radii, scale=noise)
	# center angular coordinate
	M[1] = np.random.rand(1, C) * 2 * np.pi
	# M[:,1] = np.random.normal(size=(C,), scale=1e-2)
	# M[:,1] = np.random.normal(loc = np.arange(C) * 2 * np.pi / C, scale=noise)
	# M[:,1] -= M[:,1].mean()
	# community standard deviations
	M[2] = np.random.normal(size=(1, C), loc=sigma, scale=noise)


	# initialise logistic weights
	W = np.matrix(np.random.normal(size=(K, C + 1), scale=noise))
	# W = np.random.uniform(low=-1/np.sqrt(C+1), high=1/np.sqrt(C+1), size=(K, C+1))

	# h = np.sqrt(-2 * sigma ** 2 * np.log( np.sqrt( -np.log(0.5) / C)))
	# theta_targets = (np.cosh(R) * np.cosh(community_radii) - np.cosh(h)) / \
	# 	(np.sinh(R) * np.sinh(community_radii))

	# theta_targets[theta_targets<-1] = -1
	# theta_targets[theta_targets>1] = 1

	# theta_targets = np.arccos(theta_targets)
	# theta_targets = 2 * np.exp((h - R - community_radii) / 2)
	# theta_targets *= np.sign(np.random.uniform(high=1, low=-1, size=(N,)))

	u, V = sp.sparse.linalg.eigsh(L, k=3, which="LM", sigma=0)
	thetas = np.arctan2(V[:,2], V[:,1])
	thetas = np.matrix(thetas.argsort() * 2 * np.pi / N).T

	# kmeans = KMeans(n_clusters=C)
	# kmeans.fit(thetas.reshape(-1,1))

	# M[:,1] = kmeans.cluster_centers_.flatten()

	# print theta_targets
	# theta_targets = 0

	# angular co-ordinates of nodes
	# thetas = np.random.normal(size=(N,), loc=theta_targets, scale=noise)
	# thetas = np.random.rand(N) * 2 * np.pi - np.pi
	# thetas = theta_targets

	stdout.write("Initialized thetas to:\n")
	stdout.write("{}\n".format(thetas[:10]))


	stdout.write("Initialized M to:\n")
	stdout.write("{}\n".format(M)) 

	_, h = hyperbolic_distance(R, thetas, M)
	F = compute_F(h, M)
	P = compute_P(F)

	print "F="
	print F 

	stdout.write("P_mean={}\n".format(P.mean()))

	return thetas, M, W

def train(A, X, N, K, C, R, thetas, M, W, 
	eta=1e-2, alpha=0.5, lamb_F=1e-2, lamb_W=1e-2, 
	num_processes=None, num_epochs=0, true_communities=None, 
	attribute_type="binary", plot_directory=None):

	L_G, L_X, l1_F, l1_W, loss = compute_likelihood(A, X, N, K, R, thetas, M, W, 
		lamb_F=lamb_F, lamb_W=lamb_W, alpha=alpha, attribute_type=attribute_type)
	# alpha = L_X / (L_G + L_X)
	stdout.write("alpha={}, L_G={}, L_X={}, l1_F={}, l1_W={}, total_loss={}\n".format(alpha, L_G, L_X, l1_F, l1_W, loss))

	if num_processes is not None:
		pool = Pool(num_processes)

		# delta_thetas = np.zeros(thetas.shape)
		# delta_M = np.zeros(M[:,0].shape)
		# delta_W = np.zeros(W.shape)

	for e in range(num_epochs):

		if num_processes is None:
			
			# print "thetas"
			for u in np.random.permutation(N):
				thetas[u] -= eta * update_theta_u(u, N, K, C, A, X, R,
					thetas, M, W, alpha, lamb_F, attribute_type)
				thetas[u] = thetas[u] % (2 * np.pi)

			# print "r"
			for c in np.random.permutation(C):
				M[0, c] -= eta * update_community_r_c(c, N, K, A, X, R,
					thetas, M, W, alpha, lamb_F, attribute_type)

			# # # # print "community thetas"
			for c in np.random.permutation(C):
				M[1, c] -= eta * update_community_theta_c(c, N, K, A, X, R,
					thetas, M, W, alpha, lamb_F, attribute_type)
				M[1, c] = M[1, c] % (2 * np.pi)

			# # # # print "sd"
			for c in np.random.permutation(C):
				M[2, c] -= eta * update_community_sd_c(c, N, K, A, X, R,
					thetas, M, W, alpha, lamb_F, attribute_type)

		else:
			delta_thetas = np.concatenate(pool.map(partial(update_theta_u, 
				N=N, K=K, C=C, A=A, X=X, R=R, thetas=thetas, M=M, W=W, 
				alpha=alpha, lamb_F=lamb_F, attribute_type=attribute_type),
				range(N)))

			thetas -= eta * delta_thetas
			thetas = thetas % (2 * np.pi)

			delta_M = np.concatenate(pool.map(partial(update_community_r_c, 
				N=N, K=K, A=A, X=X, R=R, thetas=thetas, M=M, W=W, 
				alpha=alpha, lamb_F=lamb_F, attribute_type=attribute_type), range(C)), axis=1)

			M[0] -= eta * delta_M

			delta_M = np.concatenate(pool.map(partial(update_community_theta_c, 
				N=N, K=K, A=A, X=X, R=R, thetas=thetas, M=M, W=W, 
				alpha=alpha, lamb_F=lamb_F, attribute_type=attribute_type), range(C)), axis=1)

			M[1] -= eta * delta_M
			M[1] = M[1] % (2 * np.pi)

			delta_M = np.concatenate(pool.map(partial(update_community_sd_c, 
				N=N, K=K, A=A, X=X, R=R, thetas=thetas, M=M, W=W, 
				alpha=alpha, lamb_F=lamb_F, attribute_type=attribute_type), range(C)), axis=1)

			M[2] -= eta * delta_M
		
		_, h = hyperbolic_distance(R, thetas, M)
		F = compute_F(h, M)
		community_predictions = F.argmax(axis=1).A1
		F = np.column_stack([F, np.ones(N)])


		if  num_processes is None:

			for k in np.random.permutation(K):
				W[k] -= eta * update_W_k(k, N, K, C, X, W, F, lamb_W, alpha, attribute_type)

		else:

			delta_W = np.concatenate(pool.map(partial(update_W_k, 
				N=N, K=K, C=C, X=X, F=F, W=W, 
				alpha=alpha, lamb_W=lamb_W, attribute_type=attribute_type), range(K)), axis=0)

			W -= eta * delta_W

		L_G, L_X, l1_F, l1_W, loss = compute_likelihood(A, X, N, K, R, thetas, M, W, 
			lamb_F=lamb_F, lamb_W=lamb_W, alpha=alpha, attribute_type=attribute_type)
		# alpha = L_X / (L_G + L_X)
		stdout.write("epoch={}, alpha={}, L_G={}, L_X={}, l1_F={}, l1_W={}, total_loss={}\n".format(e, alpha, L_G, L_X, l1_F, l1_W, loss) )

		if true_communities is not None:
			# NMI
			stdout.write("NMI: {}\n".format(NMI(true_communities, community_predictions)))

		draw_network(N, C, R, thetas, M, e, L_G, L_X, plot_directory)

		stdout.flush()

	if num_processes is not None:
		pool.close()
		pool.join()

	return thetas, M, W

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
	parser.add_argument("--eta", dest="eta", type=np.float,
                    help="learning rate (default is 0.01)", default=1e-2)
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

	thetas, M, W = initialize_matrices(L, N, C, K, R)
	stdout.write("Initialized matrices\n")

	eta = args.eta
	alpha = args.alpha
	lamb_F = args.lamb_F
	lamb_W = args.lamb_W
	num_epochs = args.num_epochs
	true_communities = preprocess_true_communities(nodes, args.true_communities)
	attribute_type = args.attribute_type
	num_processes = args.num_processes
	plot_directory = args.plot_directory

	stdout.write("Training with eta={}, num_epochs={}, lamb_F={}, lamb_W={}, alpha={}, attribute_type={}, num_processes={}\n".format(eta,	
		num_epochs, lamb_F, lamb_W, alpha, attribute_type, num_processes))
	stdout.write("saving plots to {}\n".format(plot_directory))
	thetas, M, W = train(A, X, N, K, C, R, thetas, M, W, 
		eta=eta, lamb_F=lamb_F, lamb_W=lamb_W, alpha=alpha, 
		num_epochs=num_epochs, true_communities=true_communities, 
		attribute_type=attribute_type, num_processes=num_processes,
		plot_directory=plot_directory)

	stdout.write("Trained matrices\n") 

	thetas = pd.DataFrame(thetas)
	M = pd.DataFrame(M)
	W = pd.DataFrame(W)

	thetas_filepath = args.thetas_filepath
	M_filepath = args.M_filepath
	W_filepath = args.W_filepath

	thetas.to_csv(thetas_filepath, sep=",", index=False, header=False)
	M.to_csv(M_filepath, sep=",", index=False, header=False)
	W.to_csv(W_filepath, sep=",", index=False, header=False)

	stdout.write("Written trained matrices to file\n")

	return

if __name__ == "__main__":
	main()
