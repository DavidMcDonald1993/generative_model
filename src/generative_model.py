import matplotlib
matplotlib.use("Agg")

import argparse

import numpy as np
import scipy as sp
# from scipy.sparse import csr_matrix
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

# clip value 
clip_value = 1e-8

def sigmoid(x):
	return 1 / (1 + np.exp(-x))

def compute_L_G(A, P):
	
	# clip to avoid error
	A_array = A.toarray()
	
	return -(A_array *  np.log(P) + (1 - A_array) * np.log(1 - P)).mean()
	# return -(A_array *  np.log(P) + (1 - A_array) * np.log(1 - P)).sum()

def compute_L_X(X, Q, attribute_type):

	if attribute_type != "binary":
		return (0.5 * (X - Q) ** 2).mean()
		# return (0.5 * (X - Q) ** 2).sum(axis=1).mean()
		# return (0.5 * (X - Q) ** 2).sum()
	
	# clip to avoid error
	# X_clip = np.clip(X, a_min=clip_value, a_max=1-clip_value)
	X_array = X.toarray()
	
	return -(X_array * np.log(Q) + (1 - X_array) * np.log(1 - Q)).mean()
	# return -((X * np.log(Q) + (1 - X) * np.log(1 - Q))).sum(axis=1).mean()
	# return -(X_array * np.log(Q) + (1 - X_array) * np.log(1 - Q)).sum()

def compute_likelihood(A, X, N, K, R, thetas, M, W, lamb_F, lamb_W, alpha, attribute_type):
	
	_, h = hyperbolic_distance(R, thetas, M)
	F = compute_F(h, M)
	P = compute_P(F)

	# likelihood of G
	L_G = compute_L_G(A, P)

	# l1 penalty term
	l1_F = lamb_F * np.linalg.norm(F, axis=0, ord=1).sum()

	F = np.column_stack([F, np.ones(N)])

	Q = compute_Q(F, W, attribute_type)
	
	# likelihood of X
	L_X = compute_L_X(X, Q, attribute_type)
	
	# l1
	l1_W = lamb_W * np.linalg.norm(W, axis=0, ord=1).sum()
	
	# overall likelihood
	likelihood = (1 - alpha) * L_G + alpha * L_X + l1_F + l1_W
	
	return L_G, L_X, l1_F, l1_W, likelihood

def hyperbolic_distance(R, thetas, M):
	delta_theta = np.pi - abs(np.pi - abs(thetas[:,None] - M[:,1]))
	# x = np.cosh(R[:,None]) * np.cosh(M[:,0]) - np.sinh(R[:,None]) * np.sinh(M[:,0]) * np.cos(delta_theta)
	# h = np.arccosh(x)
	if (delta_theta / 2 < 0).any():
		print "2nfgoewrngoe"
		i, j = np.where(delta_theta / 2 < 0)
		print i
		print j
		print thetas[i]
		print delta_theta[i]
		return
	h = R[:,None] + M[:,0] + 2 * np.log(delta_theta / 2)
	return delta_theta, h

# def hyperbolic_distance_u(u, R, thetas, M):
# 	delta_theta_u = np.pi - abs(np.pi - abs(thetas[u] - M[:,1]))
# 	x_u = np.cosh(R[u]) * np.cosh(M[:,0]) - np.sinh(R[u]) * np.sinh(M[:,0]) * np.cos(delta_theta_u)
# 	h_u = np.arccosh(x_u)
# 	return delta_theta_u, x_u, h_u

# def hyperbolic_distance_c(c, R, thetas, M):
# 	delta_theta_c = np.pi - abs(np.pi - abs(thetas - M[c,1]))
# 	x_c = np.cosh(R) * np.cosh(M[c,0]) - np.sinh(R) * np.sinh(M[c,0]) * np.cos(delta_theta_c)
# 	h_c = np.arccosh(x_c)
# 	return delta_theta_c, x_c, h_c

def compute_F(h, M):
	F = np.exp(- h ** 2 / (2 * M[:,2] ** 2))
	# F = 1 / np.sqrt(2 * np.pi * M[:, 2] ** 2) * F
	return F

# def compute_F_u(h_u, M):
# 	F_u = np.exp(- h_u ** 2 / (2 * M[:,2] ** 2))
# 	return F_u

# def compute_F_c(c, h_c, M):
# 	F_c = np.exp(- h_c ** 2 / (2 * M[c,2] ** 2))
# 	return F_c

def compute_P(F):
	P = 1 - np.exp(-F.dot(F.T))
	return np.clip(P, a_min=clip_value, a_max=1-clip_value)

def compute_P_u(u, F):
	P_u = 1 - np.exp(-F[u].dot(F.T))
	return np.clip(P_u, a_min=clip_value, a_max=1-clip_value)

def compute_Q(F, W, attribute_type):
	Q = F.dot(W.T)
	if attribute_type != "binary":
		return Q
	Q = sigmoid(Q)
	return np.clip(Q, a_min=clip_value, a_max=1-clip_value)

def compute_Q_u(F_u, W, attribute_type):
	Q_u = F_u.dot(W.T)
	if attribute_type != "binary":
		return Q_u
	Q_u = sigmoid(Q_u)
	return np.clip(Q_u, a_min=clip_value, a_max=1-clip_value)

def compute_Q__k(F, W__k, attribute_type):
	Q__k = F.dot(W__k)
	if attribute_type != "binary":
		return Q__k
	Q__k = sigmoid(Q__k)
	return np.clip(Q__k, a_min=clip_value, a_max=1-clip_value)

def update_theta_u(u, N, K, C, A, X, R, thetas, M, W, alpha, lamb_F, attribute_type):
	
	# compute F
	delta_theta, h = hyperbolic_distance(R, thetas, M)
	F = compute_F(h, M)
	P_u = compute_P_u(u, F)
	
	# partial delta theta partial theta
	# partial_delta_theta_u_partial_theta_u = np.array([-np.sign(np.pi - abs(thetas[u] - M[c,1])) *\
	#  -np.sign(thetas[u] - M[c,1]) * 1 for c in range(C)])
	partial_delta_theta_u_partial_theta_u = -np.sign(np.pi - abs(thetas[u] - M[:,1])) *\
	 -np.sign(thetas[u] - M[:,1]) * 1 
	
	# parital h partial theta
	# partial_x_u_partial_delta_theta_u = np.diag([np.sinh(R[u]) * np.sinh(M[c,0]) * np.sin(delta_theta[u,c])
	# 	for c in range(C)])
	# partial_x_u_partial_delta_theta_u = np.diag(nap.sinh(R[u]) * np.sinh(M[:,0]) * np.sin(delta_theta[u,:]))

	# partial_h_u_partial_x_u = np.diag([1 / np.sqrt(x[u, c] ** 2 - 1) for c in range(C)])
	# partial_h_u_partial_x_u = np.diag(1 / np.sqrt(x[u, :] ** 2 - 1))
	 
	# partial F partial theta
	# partial_F_u_partial_h_u = np.diag([-h[u, c] / M[c,2] ** 2 * F[u, c]
	# 	for c in range(C)])
	# partial_F_u_partial_h_u = np.diag(-h[u, :] / M[:, 2] ** 2 * F[u, :])

	# partial_F_u_partial_delta_theta_u = sp.sparse.diags(-h[u, :] / M[:, 2] ** 2 * F[u, :] *
	# 	1 / np.sqrt(x[u, :] ** 2 - 1) * 
	# 	np.sinh(R[u]) * np.sinh(M[:,0]) * np.sin(delta_theta[u,:]))

	partial_F_u_partial_delta_theta_u = sp.sparse.diags(-h[u, :] / M[:, 2] ** 2 * F[u, :] *
		4 / delta_theta[u])

	# partial_P_u_partial_F_u = np.array([[F[v, c] * np.exp(-F[u].dot(F[v])) for c in range(C)]
	# 	for v in range(N)])
	partial_P_u_partial_F_u = np.exp(-F[u].dot(F.T))[:,None] * F
	
	# partial L_G 
	# A_u = A[u].toarray().flatten()
	A_u = A[u]
	# partial_L_G_u_partial_P_u = 1.0 / N *\
	partial_L_G_u_partial_P_u = - 1.0 / N * \
	(A_u / P_u - 1 / (1 - P_u) + A_u / (1 - P_u))
	# np.array([A[u, v] / P[u, v] - (1 - A[u, v]) / (1 - P[u, v])
	# 	for v in range(N)])

	partial_L_G_u_partial_F_u = partial_L_G_u_partial_P_u\
	.dot(partial_P_u_partial_F_u)
	# partial_L_G_u_partial_F_u = partial_L_G_u_partial_P_u *\
	# partial_P_u_partial_F_u

	partial_F_u_partial_theta_u = partial_F_u_partial_delta_theta_u\
	.dot(partial_delta_theta_u_partial_theta_u)
	norm = sp.linalg.norm(partial_F_u_partial_theta_u)
	# print norm 
	if norm > 1:
		partial_F_u_partial_theta_u /= norm

	# partial_F_u_partial_theta_u = partial_F_u_partial_h_u\
	# .dot(partial_h_u_partial_x_u)\
	# .dot(partial_x_u_partial_delta_theta_u)\
	# .dot(partial_delta_theta_u_partial_theta_u)

	partial_L_G_u_partial_theta_u = partial_L_G_u_partial_F_u\
	.dot(partial_F_u_partial_theta_u)

	partial_l1_F_u_partial_F_u = np.sign(F[u])

	# partial L_X
	F_u = np.append(F[u], 1)
	Q_u = compute_Q_u(F_u, W, attribute_type)
	# partial_L_X_u_partial_Q_u = 1.0 / K * np.array([X[u,k] / Q_u[k] -\
	# 	(1 - X[u, k]) / (1 - Q_u[k]) for k in range(K)])
	# partial_Q_u_partial_F_u = np.array([[Q_u[k] * (1 - Q_u[k]) * W[k, c] 
	# 	for c in range(C)] for k in range(K)])
	# partial_L_X_u_partial_F_u = partial_L_X_u_partial_Q_u.dot(partial_Q_u_partial_F_u)

	# partial_L_X_u_partial_F_u  = 1.0 / K *\
	partial_L_X_u_partial_F_u = - 1.0 / N * \
	(X[u] - Q_u).dot(W[:,:-1])
	# np.array([(X[u] - Q_u).dot(W[:,c]) for c in range(C)])

	partial_L_X_u_partial_theta_u = partial_L_X_u_partial_F_u.dot(partial_F_u_partial_theta_u)

	if abs(-0.01*((1 - alpha) * partial_L_G_u_partial_theta_u + alpha * partial_L_X_u_partial_theta_u\
			+ lamb_F * partial_l1_F_u_partial_F_u.dot(partial_F_u_partial_theta_u)).A1[0]) > np.pi/2:
		print "TOO BIG"
		print u, -0.01*((1 - alpha) * partial_L_G_u_partial_theta_u + alpha * partial_L_X_u_partial_theta_u\
			+ lamb_F * partial_l1_F_u_partial_F_u.dot(partial_F_u_partial_theta_u)).A1
		print
		print delta_theta[u]
		print 4/delta_theta[u]
		# print np.sign(np.pi - abs(thetas[u] - M[:,1]))
		# print np.sign(thetas[u] - M[:,1])
		# print thetas[u] - M[:,1]
		# print M
		# print thetas[u]
		print 
		print P_u
		print 1 / P_u

		print 
		print partial_delta_theta_u_partial_theta_u
		print partial_F_u_partial_delta_theta_u
		print partial_F_u_partial_theta_u
		# print partial_L_G_u_partial_P_u.min()
		# # print A_u[0,0]
		# # print P_u[0]
		# # print 1 - A_u[0,0]
		# # print 1 - P_u[0]
		# print partial_P_u_partial_F_u.min()
		print partial_L_G_u_partial_F_u

		print partial_L_G_u_partial_theta_u
		print 
		print partial_L_X_u_partial_theta_u
		# return np.sign(-0.01*((1 - alpha) * partial_L_G_u_partial_theta_u + alpha * partial_L_X_u_partial_theta_u\
		# 	+ lamb_F * partial_l1_F_u_partial_F_u.dot(partial_F_u_partial_theta_u)).A1[0]) * 0.01
		return 

	# print -0.01 * ((1 - alpha) * partial_L_G_u_partial_theta_u + alpha * partial_L_X_u_partial_theta_u\
	# 		+ lamb_F * partial_l1_F_u_partial_F_u.dot(partial_F_u_partial_theta_u)).A1

	return ((1 - alpha) * partial_L_G_u_partial_theta_u + alpha * partial_L_X_u_partial_theta_u\
		+ lamb_F * partial_l1_F_u_partial_F_u.dot(partial_F_u_partial_theta_u)).A1

# def compute_delta_theta(N, C, K, A, X, R, thetas, M, W, alpha, lamb_F):
	
# 	# compute F
# 	delta_theta = np.pi - abs(np.pi - abs(thetas[:,None] - M[:,1]))
# 	x = np.cosh(R[:,None]) * np.cosh(M[:,0]) - np.sinh(R[:,None]) * np.sinh(M[:,0]) * np.cos(delta_theta)
# 	h = np.arccosh(x)
# 	F = np.exp(- h ** 2 / (2 * M[:,2] ** 2))
# 	P = 1 - np.exp(-F.dot(F.T))
	
# 	# partial delta theta partial theta
# 	partial_delta_theta_partial_theta = np.array([[[-np.sign(np.pi - abs(thetas[u] - M[c,1])) *\
# 	 -np.sign(thetas[u] - M[c,1]) * 1 if u_prime == u else 0\
# 	 for u_prime in range(N)] for c in range(C)] for u in range(N)])
	
# 	# partial h partial theta
# 	partial_x_partial_delta_theta = np.array([[[[np.sinh(R[u]) * np.sinh(M[c,0]) * np.sin(delta_theta[u,c])\
# 		if u_prime==u and c_prime==c else 0\
# 		for c_prime in range(C)] for u_prime in range(N)] for c in range(C)] for u in range(N)])

# 	partial_h_partial_x = np.array([[[[1 / np.sqrt(x[u, c] ** 2 - 1) if u_prime==u and c_prime==c else 0\
# 		for c_prime in range(C)] for u_prime in range(N)] for c in range(C)] for u in range(N)])
	 
# 	# partial F partial theta
# 	partial_F_partial_h = np.array([[[[-h[u, c] / M[c,2] ** 2 * F[u, c] if u_prime==u and c_prime==c else 0\
# 		for c_prime in range(C)] for u_prime in range(N)] for c in range(C)] for u in range(N)])

# 	partial_P_partial_F = np.array([[[[F[v, c] * np.exp(-F[u].dot(F[v]))\
# 	 if u_prime==u else F[u, c] * np.exp(-F[u].dot(F[v])) if u_prime==v else 0\
# 		for c in range(C)] for u_prime in range(N)] for v in range(N)] for u in range(N)])
	
# 	# partial L_G
# 	partial_L_G_partial_P = 1 / N**2 *np.array([[A[u,v] / P[u,v] - (1 - A[u,v]) / (1 - P[u,v])\
# 		for v in range(N)] for u in range(N)])

# 	partial_L_G_partial_F = np.tensordot(partial_L_G_partial_P,\
# 	partial_P_partial_F)

# 	partial_F_partial_theta = np.tensordot(np.tensordot(np.tensordot(partial_F_partial_h,\
# 	partial_h_partial_x),\
# 	partial_x_partial_delta_theta),\
# 	partial_delta_theta_partial_theta)

# 	partial_L_G_u_partial_theta_u = np.tensordot(partial_L_G_partial_F,\
# 	partial_F_partial_theta)

# 	partial_l1_F_partial_F = np.sign(F)

# 	# partial L_X
# 	F = np.column_stack([F, np.ones(N)])
# 	Q = compute_Q(N, F, W)
# 	partial_L_X_partial_Q = 1 / (N*K) * np.array([[X[u, k] / Q[u, k] - (1 - X[u,k]) / (1 - Q[u, k])
# 		for k in range(K)] for u in range(N)])
# 	partial_Q_partial_F = np.array([[[[Q[u,k] * (1 - Q[u,k]) * W[k, c] if u_prime==u else 0
# 		for c in range(C)] for u_prime in range(N)] for k in range(K)] for u in range(N)])
# 	partial_L_X_partial_F = np.tensordot(partial_L_X_partial_Q, partial_Q_partial_F)

# 	partial_L_X_partial_theta = np.tensordot(partial_L_X_partial_F, partial_F_partial_theta)

# 	return (1 - alpha) * partial_L_G_partial_theta + alpha * partial_L_X_partial_theta\
# 		- lamb_F * np.tensordot(partial_l1_F_partial_F, partial_F_partial_theta)

def compute_partial_P_partial_F_c(c, N, A, P, F):

	partial_P_partial_F_c = sp.sparse.lil_matrix((N**2, N))

	U = np.repeat(np.arange(N), 2*N)
	V = np.tile(np.repeat(np.arange(N), 2), N)
	U_even = U[np.arange(U.shape[0]) % 2 == 0]
	V_even = V[np.arange(V.shape[0]) % 2 == 0]
	exp = np.exp(-F.dot(F.T))

	partial_P_partial_F_c[U * N + V, np.column_stack([U_even, V_even]).flatten()] = \
	F[np.column_stack([V_even, U_even]).flatten(), c] * exp[U, V]

	return sp.sparse.csr_matrix(partial_P_partial_F_c)

def update_community_r_c(c, N, K, A, X, R, thetas, M, W, alpha, lamb_F, attribute_type):
	
	# compute F
	delta_theta, h = hyperbolic_distance(R, thetas, M)
	F = compute_F(h, M)
	P = compute_P(F)

	# partial h
	# partial_x_c_partial_r_c = np.array([np.cosh(R[u]) * np.sinh(M[c, 0]) -\
	# 				  np.sinh(R[u]) * np.cosh(M[c, 0]) * np.cos(delta_theta[u, c]) for u in range(N)])
	# partial_x_c_partial_r_c = np.cosh(R) * np.sinh(M[c, 0]) -\
	# 				  np.sinh(R) * np.cosh(M[c, 0]) * np.cos(delta_theta[:, c])

	partial_h_c_partial_r_c = np.ones(N)

	# partial_h_c_partial_x_c = np.diag([1 / np.sqrt(x[u, c] ** 2 - 1) for u in range(N)])
	# partial_h_c_partial_x_c = np.diag(1 / np.sqrt(x[:, c] ** 2 - 1))
	
	# partial F
	# partial_F_c_partial_h_c = np.diag([ - h[u, c] / M[c, 2] ** 2 * F[u, c] for u in range(N)]) 
	# partial_F_c_partial_h_c = np.diag(-h[:, c] / M[c, 2] ** 2 * F[:, c]) 

	# partial_F_c_partial_x_c = sp.sparse.diags(-h[:, c] / M[c, 2] ** 2 * F[:, c] * 
	# 	1 / np.sqrt(x[:, c] ** 2 - 1))

	partial_F_c_partial_h_c = sp.sparse.diags(-h[:, c] / M[c, 2] ** 2 * F[:, c])
	
	# partial L_G TODO
	# partial_P_partial_F_c = np.array([[[F[v, c] * np.exp( - F[u].dot(F[v])) if u_prime == u\
	# else F[u, c] * np.exp( - F[u].dot(F[v])) if u_prime == v else 0\
	# for u_prime in range(N)] for v in range(N)] for u in range(N)])

	# partial_P_partial_F_c = np.zeros((N, N, N))
	# U = np.repeat(np.arange(N), 2*N)
	# V = np.tile(np.repeat(np.arange(N), 2), N)
	# U_even = U[np.arange(U.shape[0]) % 2 == 0]
	# V_even = V[np.arange(V.shape[0]) % 2 == 0]
	# exp = np.exp(-F.dot(F.T))
	# partial_P_partial_F_c[U, V, np.column_stack([U_even, V_even]).flatten()] =\
	# F[np.column_stack([V_even, U_even]).flatten(), c] * exp[U, V]

	# partial_L_G_partial_P = 1.0 / N**2 * \
	# partial_L_G_partial_P = 1.0 / N *\
	# (A / P - (1 - A.todense()) / (1 - P))
	# np.array([[A[u,v] / P[u,v] - (1 - A[u,v]) / (1 - P[u,v])\
	# 	for v in range(N)] for u in range(N)])

	# partial_L_G_c_partial_F_c = np.tensordot(partial_L_G_partial_P, partial_P_partial_F_c)
	
	partial_L_G_partial_P = - 1.0 / N * \
	(A / P - 1 / (1 - P) + A / (1 - P)).reshape(1, -1)

	partial_P_partial_F_c = compute_partial_P_partial_F_c(c, N, A, P, F)

	partial_L_G_c_partial_F_c = partial_L_G_partial_P * partial_P_partial_F_c

	# partial_F_c_partial_r_c = partial_F_c_partial_h_c\
	# .dot(partial_h_c_partial_x_c)\
	# .dot(partial_x_c_partial_r_c)
	# partial_F_c_partial_r_c = partial_F_c_partial_x_c.dot(partial_x_c_partial_r_c)
	partial_F_c_partial_r_c = partial_F_c_partial_h_c.dot(partial_h_c_partial_r_c)

	partial_L_G_c_partial_r_c = partial_L_G_c_partial_F_c.dot(partial_F_c_partial_r_c)

	partial_l1_F_c_partial_F_c = np.sign(F[:,c])
	
	# partial L_x 
	F = np.column_stack([F, np.ones(N)])
	Q = compute_Q(F, W, attribute_type)

	# partial_L_X_c_partial_Q = 1.0 / (N*K) * np.array([[X[u, k] / Q[u, k] - (1 - X[u, k]) / (1 - Q[u, k])\
	# 	for k in range(K)] for u in range(N)])
	# partial_Q_partial_F_c = np.array([[[Q[u,k] * (1 - Q[u,k]) * W[k,c] if u_prime == u else 0\
	# 	for u_prime in range(N)] for k in range(K)] for u in range(N)])
	# partial_L_X_c_partial_F_c = np.tensordot(partial_L_X_c_partial_Q, partial_Q_partial_F_c)

	# partial_L_X_c_partial_F_c = 1.0 / (N * K) *\ 
	partial_L_X_c_partial_F_c = - 1.0 / N * \
	(X - Q).dot(W[:,c])
	# np.array([(X[u] - Q[u]).dot(W[:,c]) for u in range(N)])

	partial_L_X_c_partial_r_c = partial_L_X_c_partial_F_c.dot(partial_F_c_partial_r_c)

	# print M[c, 0]
	# print -0.01 * ((1 - alpha) * partial_L_G_c_partial_r_c + alpha * partial_L_X_c_partial_r_c\
	#  + lamb_F * partial_l1_F_c_partial_F_c.dot(partial_F_c_partial_r_c)).A1

	return ((1 - alpha) * partial_L_G_c_partial_r_c + alpha * partial_L_X_c_partial_r_c\
	 + lamb_F * partial_l1_F_c_partial_F_c.dot(partial_F_c_partial_r_c)).A1

def update_community_theta_c(c, N, K, A, X, R, thetas, M, W, alpha, lamb_F, attribute_type):
	
	# compute F
	delta_theta, h = hyperbolic_distance(R, thetas, M)
	F = compute_F(h, M)
	P = compute_P(F)
	
	# partial delta theta
	# partial_delta_theta_c_partial_theta_c = np.array([-np.sign(np.pi - abs(thetas[u] - M[c,1]))\
	#  * -np.sign(thetas[u] - M[c,1]) * -1 for u in range(N)])
	partial_delta_theta_c_partial_theta_c = -np.sign(np.pi - abs(thetas - M[c,1]))\
	 * -np.sign(thetas - M[c,1]) * -1
	
	# partial h
	# partial_x_c_partial_delta_theta_c = np.diag([np.sinh(R[u]) * np.sinh(M[c,0]) * np.sin(delta_theta[u,c])\
	# 	for u in range(N)])
	# partial_x_c_partial_delta_theta_c = np.diag(np.sinh(R) * np.sinh(M[c,0]) * np.sin(delta_theta[:,c]))

	# partial_h_c_partial_x_c = np.diag([1 / np.sqrt(x[u, c] ** 2 - 1) for u in range(N)])
	# partial_h_c_partial_x_c = np.diag(1 / np.sqrt(x[:, c] ** 2 - 1))
	
	# partial F
	# partial_F_c_partial_h_c = np.diag([ - h[u, c] / M[c, 2] ** 2 * F[u, c] for u in range(N)]) 
	# partial_F_c_partial_h_c = np.diag(-h[:, c] / M[c, 2] ** 2 * F[:, c]) 

	# partial_F_c_partial_delta_theta_c = sp.sparse.diags(-h[:, c] / M[c, 2] ** 2 * F[:, c] * 
	# 	1 / np.sqrt(x[:, c] ** 2 - 1) * np.sinh(R) * np.sinh(M[c,0]) * np.sin(delta_theta[:,c]))

	partial_F_c_partial_delta_theta_c = sp.sparse.diags(-h[:, c] / M[c, 2] ** 2 * F[:, c] * 
		4 / delta_theta[:,c])
	
	partial_F_c_partial_theta_c = partial_F_c_partial_delta_theta_c.dot(partial_delta_theta_c_partial_theta_c)

	# print partial_F_c_partial_theta_c

	norm = sp.linalg.norm(partial_F_c_partial_theta_c)
	# print norm
	# if norm > 1:
	# 	partial_F_c_partial_theta_c /= norm

	# partial L_G 
	# partial_P_partial_F_c = np.array([[[F[v, c] * np.exp( - F[u].dot(F[v])) if u_prime == u\
	# else F[u, c] * np.exp( - F[u].dot(F[v])) if u_prime == v else 0\
	# for u_prime in range(N)] for v in range(N)] for u in range(N)])

	# partial_P_partial_F_c = np.zeros((N, N, N))
	# U = np.repeat(np.arange(N), 2*N)
	# V = np.tile(np.repeat(np.arange(N), 2), N)
	# U_even = U[np.arange(U.shape[0]) % 2 == 0]
	# V_even = V[np.arange(V.shape[0]) % 2 == 0]
	# exp = np.exp(-F.dot(F.T))
	# partial_P_partial_F_c[U, V, np.column_stack([U_even, V_even]).flatten()] =\
	# F[np.column_stack([V_even, U_even]).flatten(), c] * exp[U, V]

	# # partial_L_G_partial_P = 1.0 / N**2 *\
	# partial_L_G_partial_P = 1.0 / N * \
	# (A / P - (1 - A.todense()) / (1 - P))
	# # np.array([[A[u,v] / P[u,v] - (1 - A[u,v]) / (1 - P[u,v])\
	# # 	for v in range(N)] for u in range(N)])

	# partial_L_G_c_partial_F_c = np.tensordot(partial_L_G_partial_P,partial_P_partial_F_c)

	partial_L_G_partial_P = - 1.0 / N * \
	(A / P - 1 / (1 - P) + A / (1 - P)).reshape(1, -1)

	partial_P_partial_F_c = compute_partial_P_partial_F_c(c, N, A, P, F)

	partial_L_G_c_partial_F_c = partial_L_G_partial_P * partial_P_partial_F_c


	# partial_F_c_partial_theta_c = partial_F_c_partial_h_c\
	# .dot(partial_h_c_partial_x_c)\
	# .dot(partial_x_c_partial_delta_theta_c)\
	# .dot(partial_delta_theta_c_partial_theta_c)


	partial_L_G_c_partial_theta_c = partial_L_G_c_partial_F_c.dot(partial_F_c_partial_theta_c)

	partial_l1_F_c_partial_F_c = np.sign(F[:,c])
	
	# LX
	F = np.column_stack([F, np.ones(N)])
	Q = compute_Q(F, W, attribute_type)

	# partial_L_X_c_partial_Q = 1.0 / (N*K) * np.array([[X[u, k] / Q[u, k] - (1 - X[u, k]) / (1 - Q[u, k])\
	# 	for k in range(K)] for u in range(N)])
	# partial_Q_partial_F_c = np.array([[[Q[u,k] * (1 - Q[u,k]) * W[k,c] if u_prime == u else 0\
	# 	for u_prime in range(N)] for k in range(K)] for u in range(N)])
	# partial_L_X_c_partial_F_c = np.tensordot(partial_L_X_c_partial_Q, partial_Q_partial_F_c)

	# partial_L_X_c_partial_F_c = 1.0 / (N * K) *\
	partial_L_X_c_partial_F_c = - 1.0 / N * \
	(X - Q).dot(W[:,c])
	# np.array([(X[u] - Q[u]).dot(W[:,c]) for u in range(N)])

	partial_L_X_c_partial_theta_c = partial_L_X_c_partial_F_c.dot(partial_F_c_partial_theta_c)

	# print -0.01 * ((1 - alpha) * partial_L_G_c_partial_theta_c + alpha * partial_L_X_c_partial_theta_c\
	# 	+ lamb_F * partial_l1_F_c_partial_F_c.dot(partial_F_c_partial_theta_c)).A1

	return ((1 - alpha) * partial_L_G_c_partial_theta_c + alpha * partial_L_X_c_partial_theta_c\
		+ lamb_F * partial_l1_F_c_partial_F_c.dot(partial_F_c_partial_theta_c)).A1

def update_community_sd_c(c, N, K, A, X, R, thetas, M, W, alpha, lamb_F, attribute_type):
	
	# compute F

	delta_theta, h = hyperbolic_distance(R, thetas, M)
	F = compute_F(h, M)
	P = compute_P(F)
	
	# partial F
	# partial_F_c_partial_sd_c = np.array([h[u, c] ** 2 / M[c, 2] ** 3 * F[u,c] for u in range(N)])
	partial_F_c_partial_sd_c = h[:, c] ** 2 / M[c, 2] ** 3 * F[:,c]
	# partial_F_c_partial_sd_c = F[:,c] * (h[:,c] ** 2 / M[c,2] ** 3 - 1 / M[c,2])

	# partial L_G 
	# partial_P_partial_F_c = np.array([[[F[v, c] * np.exp( - F[u].dot(F[v])) if u_prime == u\
	# else F[u, c] * np.exp( - F[u].dot(F[v])) if u_prime == v else 0\
	# for u_prime in range(N)] for v in range(N)] for u in range(N)])
	# partial_P_partial_F_c = np.zeros((N, N, N))
	# U = np.repeat(np.arange(N), 2*N)
	# V = np.tile(np.repeat(np.arange(N), 2), N)
	# U_even = U[np.arange(U.shape[0]) % 2 == 0]
	# V_even = V[np.arange(V.shape[0]) % 2 == 0]
	# exp = np.exp(-F.dot(F.T))
	# partial_P_partial_F_c[U, V, np.column_stack([U_even, V_even]).flatten()] =\
	# F[np.column_stack([V_even, U_even]).flatten(), c] * exp[U, V]

	# # partial_L_G_partial_P = 1.0 / N**2 *\
	# partial_L_G_partial_P = 1.0 / N * \
	# (A / P - (1 - A.todense()) / (1 - P))
	# # np.array([[A[u,v] / P[u,v] - (1 - A[u,v]) / (1 - P[u,v])\
	# # 	for v in range(N)] for u in range(N)])

	# partial_L_G_c_partial_F_c = np.tensordot(partial_L_G_partial_P,partial_P_partial_F_c)


	partial_L_G_partial_P = - 1.0 / N * \
	(A / P - 1 / (1 - P) + A / (1 - P)).reshape(1, -1)

	partial_P_partial_F_c = compute_partial_P_partial_F_c(c, N, A, P, F)

	partial_L_G_c_partial_F_c = partial_L_G_partial_P * partial_P_partial_F_c

	partial_L_G_c_partial_sd_c = partial_L_G_c_partial_F_c.dot(partial_F_c_partial_sd_c)

	partial_l1_F_c_partial_F_c = np.sign(F[:,c])
	
	# partial L_x 
	F = np.column_stack([F, np.ones(N)])
	Q = compute_Q(F, W, attribute_type)

	# partial_L_X_c_partial_Q = 1.0 / (N*K) * np.array([[X[u, k] / Q[u, k] - (1 - X[u, k]) / (1 - Q[u, k])\
	# 	for k in range(K)] for u in range(N)])
	# partial_Q_partial_F_c = np.array([[[Q[u,k] * (1 - Q[u,k]) * W[k,c] if u_prime == u else 0\
	# 	for u_prime in range(N)] for k in range(K)] for u in range(N)])
	# partial_L_X_c_partial_F_c = np.tensordot(partial_L_X_c_partial_Q, partial_Q_partial_F_c)

	# partial_L_X_c_partial_F_c = 1.0 / (N * K) *\
	partial_L_X_c_partial_F_c = - 1.0 / N * \
	(X - Q).dot(W[:,c])
	# np.array([(X[u] - Q[u]).dot(W[:,c]) for u in range(N)])

	partial_L_X_c_partial_sd_c = partial_L_X_c_partial_F_c.dot(partial_F_c_partial_sd_c)

	# print -0.01 * ((1 - alpha) * partial_L_G_c_partial_sd_c + alpha * partial_L_X_c_partial_sd_c\
	# 	+ lamb_F * partial_l1_F_c_partial_F_c.dot(partial_F_c_partial_sd_c)).A1

	return ((1 - alpha) * partial_L_G_c_partial_sd_c + alpha * partial_L_X_c_partial_sd_c\
		+ lamb_F * partial_l1_F_c_partial_F_c.dot(partial_F_c_partial_sd_c)).A1


def update_W_k(k, N, K, C, X, W, F, lamb_W, alpha, attribute_type):
	
	# kth row of W
	W_k = W[k]
	# print k
	# print W_k
	
	# kth column of Q
	Q__k = compute_Q__k(F, W_k, attribute_type)
	# Q = compute_Q(F, W, attribute_type)

	# partial_L_X_k_partial_Q_k = 1.0 / N * np.array([X[u, k] / Q[u, k] - (1 - X[u, k]) / (1 - Q[u, k])
	# 	for u in range(N)])
	# partial_Q_k_partial_W_k = np.array([[Q[u, k] * (1 - Q[u, k]) * F[u, c] 
	# 	for c in range(C+1)] for u in range(N)])
	# partial_L_X_k_partial_W_k = partial_L_X_k_partial_Q_k.dot(partial_Q_k_partial_W_k)

	# partial_L_X_k_partial_W_k = 1.0 / N *\
	partial_L_X_k_partial_W_k = - 1.0 / N * \
	(X.T[k] - Q__k).dot(F)
	# print partial_L_X_k_partial_W_k.shape
	# np.array([(X[:, k] - Q__k).dot(F[:, c]) for c in range(C + 1)])

	# print (X[:,k] - Q__k)
	# print partial_L_X_k_partial_W_k 

	partial_l1_W_k_partial_W_k = np.sign(W_k)
	
	# print (alpha * partial_L_X_k_partial_W_k - lamb_W * partial_l1_W_k_partial_W_k).A1

	return (alpha * partial_L_X_k_partial_W_k + lamb_W * partial_l1_W_k_partial_W_k).A1

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
	R = R[order_of_appearance]

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

	sigma = R.mean()
	community_radii = R.mean()
	noise = 1e-2
	# community matrix M
	M = np.zeros((C, 3))
	# centre radii
	M[:,0] = np.random.normal(size=(C,), loc=community_radii, scale=noise)
	# center angular coordinate
	M[:,1] = np.random.rand(C) * 2 * np.pi
	# M[:,1] = np.random.normal(size=(C,), scale=1e-2)
	# M[:,1] = np.random.normal(loc = np.arange(C) * 2 * np.pi / C, scale=noise)
	# M[:,1] -= M[:,1].mean()
	# community standard deviations
	M[:,2] = np.random.normal(size=(C,), loc=sigma, scale=noise)


	# initialise logistic weights
	W = np.random.normal(size=(K, C + 1), scale=noise)
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
	thetas = thetas.argsort() * 2 * np.pi / N

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
	num_processes=None, num_epochs=0, true_communities=None, attribute_type="binary"):

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
				M[c, 0] -= eta * update_community_r_c(c, N, K, A, X, R,
					thetas, M, W, alpha, lamb_F, attribute_type)

			# print "community thetas"
			for c in np.random.permutation(C):
				M[c, 1] -= eta * update_community_theta_c(c, N, K, A, X, R,
					thetas, M, W, alpha, lamb_F, attribute_type)
				M[c, 1] = M[c, 1] % (2 * np.pi)

			# print "sd"
			for c in np.random.permutation(C):
				M[c, 2] -= eta * update_community_sd_c(c, N, K, A, X, R,
					thetas, M, W, alpha, lamb_F, attribute_type)

		else:
			delta_thetas = np.concatenate(pool.map(partial(update_theta_u, 
				N=N, K=K, C=C, A=A, X=X, R=R, thetas=thetas, M=M, W=W, 
				alpha=alpha, lamb_F=lamb_F, attribute_type=attribute_type),
				range(N)))

			# norm = np.linalg.norm(delta_thetas)
			# if norm > 1:
			# 	delta_thetas /= norm
			thetas -= eta * delta_thetas
			thetas = thetas % (2 * np.pi)

			delta_M = np.concatenate(pool.map(partial(update_community_r_c, 
				N=N, K=K, A=A, X=X, R=R, thetas=thetas, M=M, W=W, 
				alpha=alpha, lamb_F=lamb_F, attribute_type=attribute_type), range(C)))

			# print "r"
			# print delta_M

			# norm = np.linalg.norm(delta_M)
			# if norm > 1:
			# 	delta_M /= norm
			# print delta_M
			M[:,0] -= eta * delta_M

			delta_M = np.concatenate(pool.map(partial(update_community_theta_c, 
				N=N, K=K, A=A, X=X, R=R, thetas=thetas, M=M, W=W, 
				alpha=alpha, lamb_F=lamb_F, attribute_type=attribute_type), range(C)))

			# print "theta"
			# print delta_M

			# norm = np.linalg.norm(delta_M)
			# if norm > 1:
			# 	delta_M /= norm
			# print delta_M
			M[:,1] -= eta * delta_M
			M[:,1] = M[:,1] % (2 * np.pi)


			delta_M = np.concatenate(pool.map(partial(update_community_sd_c, 
				N=N, K=K, A=A, X=X, R=R, thetas=thetas, M=M, W=W, 
				alpha=alpha, lamb_F=lamb_F, attribute_type=attribute_type), range(C)))

			# print "sd"
			# print delta_M

			# norm = np.linalg.norm(delta_M)
			# if norm > 1:
			# 	delta_M /= norm
			# print delta_M
			M[:,2] -= eta * delta_M
		
		# stdout.write("thetas=\n")
		# stdout.write("{}\n".format(thetas[:10]))

		stdout.write("M=\n")
		stdout.write("{}\n".format(M))
		
		_, h = hyperbolic_distance(R, thetas, M)
		F = compute_F(h, M)
		community_predictions = F.argmax(axis=1)
		F = np.column_stack([F, np.ones(N)])

		stdout.write("F=\n")
		stdout.write("{}\n".format(F[:5]))

		# P = compute_P(F)

		# stdout.write("P=\n")
		# stdout.write("{}\n".format(P[:5, :10]))

		# stdout.write("A=\n")
		# stdout.write("{}\n".format(A[:5, :10].toarray()))

		if  num_processes is None:

			for k in np.random.permutation(K):
				W[k] -= eta * update_W_k(k, N, K, C, X, W, F, lamb_W, alpha, attribute_type)

		else:

			delta_W = np.array(pool.map(partial(update_W_k, 
				N=N, K=K, C=C, X=X, F=F, W=W, 
				alpha=alpha, lamb_W=lamb_W, attribute_type=attribute_type), range(K)))

			W -= eta * delta_W

		# stdout.write("W=\n")
		# stdout.write("{}\n".format(W[:5]))

		# Q = compute_Q(F, W, attribute_type)

		# stdout.write("Q=\n")
		# stdout.write("{}\n".format(Q[:5, :10]))

		# stdout.write("X=\n")
		# stdout.write("{}\n".format(X[:5, :10].toarray()))

		L_G, L_X, l1_F, l1_W, loss = compute_likelihood(A, X, N, K, R, thetas, M, W, 
			lamb_F=lamb_F, lamb_W=lamb_W, alpha=alpha, attribute_type=attribute_type)
		# alpha = L_X / (L_G + L_X)
		stdout.write("epoch={}, alpha={}, L_G={}, L_X={}, l1_F={}, l1_W={}, total_loss={}\n".format(e, alpha, L_G, L_X, l1_F, l1_W, loss) )

		if true_communities is not None:
			# NMI
			stdout.write("NMI: {}\n".format(NMI(true_communities, community_predictions)))

		stdout.flush()

		draw_network(N, C, R, thetas, M, e, L_G, L_X)

	if num_processes is not None:
		pool.close()
		pool.join()

	return thetas, M, W

def draw_network(N, C, R, thetas, M, e, L_G, L_X):

	_, h = hyperbolic_distance(R, thetas, M)
	F = compute_F(h, M)
	assignments = F.argmax(axis=1)
	assignment_strength = np.array([F[i, assignments[i]] for i in range(N)])

	node_cartesian = np.column_stack([R * np.cos(thetas), R * np.sin(thetas)])
	community_cartesian = np.column_stack([M[:,0] * np.cos(M[:,1]), M[:,0] * np.sin(M[:,1])])

	plt.figure(figsize=(15, 15))
	plt.title("Epoch={}, L_G={}, L_X={}".format(e, L_G, L_X))
	plt.scatter(node_cartesian[:,0], node_cartesian[:,1], c=assignments, s=100*assignment_strength)
	plt.scatter(community_cartesian[:,0], community_cartesian[:,1], c=np.arange(C), s=100)
	plt.scatter(community_cartesian[:,0], community_cartesian[:,1], c = "k", s=25)
	# plt.show()
	plt.savefig("plots/epoch_{}.png".format(e))
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

	args = parser.parse_args()

	return args

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

	stdout.write("Training with eta={}, num_epochs={}, lamb_F={}, lamb_W={}, alpha={}, attribute_type={}, num_processes={}\n".format(eta,	
		num_epochs, lamb_F, lamb_W, alpha, attribute_type, num_processes))

	thetas, M, W = train(A, X, N, K, C, R, thetas, M, W, 
		eta=eta, lamb_F=lamb_F, lamb_W=lamb_W, alpha=alpha, 
		num_epochs=num_epochs, true_communities=true_communities, 
		attribute_type=attribute_type, num_processes=num_processes)

	stdout.write("Trained matrices\n") 

	# draw_network(N, C, R, thetas, M)
	
	# stdout.write("Visualised network\n")

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
