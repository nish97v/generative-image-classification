import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
import random
from scipy.stats import multivariate_normal
from numpy.linalg import inv
import math
import cv2
from PIL import Image
from math import *

import data_loader 
import modules

DATA_PATH = '../data/'

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save', action='store_true',
                        help='Save data.')
    parser.add_argument('--load', action='store_true',
                        help='Check data loading.')
    parser.add_argument('--gm', action='store_true',
                        help='Gaussian Model')
    parser.add_argument('--mogm', action='store_true',
                        help='Mixture of Gaussian Model')
    parser.add_argument('--td', action = 'store_true',
                        help = 'T-distribution')
    parser.add_argument('--fa', action = 'store_true',
                        help = 'Factor Analysis')
    parser.add_argument('--test_gm', action = 'store_true',
                        help = 'Test Gaussian Model')
    parser.add_argument('--test_mogm', action = 'store_true',
                        help = 'Test Mixture of Gaussian Model')
    parser.add_argument('--test_td', action = 'store_true',
                        help = 'Test T-distribution')
    parser.add_argument('--test_fa', action = 'store_true',
                        help = 'Test Factor Analysis')
    return parser.parse_args()

def save_data():
	data_loader.loader(DATA_PATH, 1100, face = False)
	data_loader.loader(DATA_PATH, 1300, face = True)
	print('Saved.')

def load_data(face, Train):
	image = data_loader.load_wrapper(DATA_PATH, face, Train)	
	return image

def MLE(face, Train):
	image = load_data(face, Train)
	shape = (10, 10)
	sum_x  = image.sum(axis = 0)
	mu = (sum_x/(len(image)))
	print('Mean(MLE).shape  : ', mu.shape)
	mu_cap = np.reshape(mu, shape)
	cv2.imwrite('../results/gaussian/Mean.jpg', mu_cap)

	sub_var = np.square(np.subtract(image, mu))
	sum_var_x = sub_var.sum(axis = 0)
	covar = (sum_var_x/len(image))
	covar_cap  = np.sqrt(covar)
	print('Covariance(MLE).shape : ', covar.shape)
	covar_cap = np.reshape(covar_cap, shape)
	cv2.imwrite('../results/gaussian/Covar.jpg', covar_cap)

	return image, mu, covar

def MOGM(face, Train):
	x, mu_mle, covar_mle = MLE(face, Train)
	print('\nMOGM - EM Algorithm \n')
	num_of_images = len(x)
	shape = len(x[0])
	print('No of images : ', num_of_images, ', Shape : ', shape)

	## E - Step ##
	K = 5 #hidden variable
	lmbda = np.ones((K))/K
	mu = np.zeros((K, shape))
	for k in range(K):
		mu[k] = x[k*5]
	# mu = x[:4]
	sigma = np.array([np.diag(covar_mle)]*K)
	print('Mu shape : ', mu.shape, ', Sigma shape : ', sigma.shape ,', Lambda shape : ', lmbda.shape)
	no_of_iterations = 10
	print('No of iterations : ', no_of_iterations)
	for no in range(no_of_iterations):

		## E-Step
		a = np.zeros((K,num_of_images))
		b = np.zeros((K,num_of_images))
		for k in range(K):
			a[k] = multivariate_normal.pdf(x, mu[k], sigma[k])
			b[k] = lmbda[k]*a[k]
		c_sum = np.sum(b, axis = 0)
		for k in range(K):
			b[k] = b[k]/c_sum
		mean_b_k = np.mean(np.mean(b, axis = 0), axis = 0)
		for i in range(K):
			for j in range(num_of_images):
				if(math.isnan(b[i][j])):
					# print('nan')
					b[i][j] = mean_b_k
		r_ik = b
		
		## M-Step
		sum_ri = r_ik.sum(axis= 1)
		sum_sum_ri = sum_ri.sum(axis= 0)
		lmbda = sum_ri/sum_sum_ri

		mu_new = np.zeros((K,10,10))
		mu_numerator = np.zeros((K, 100))
		for k in range(K):
			mu_numerator[k] = np.matmul(r_ik[k], x)
			mu[k] = mu_numerator[k]/sum_ri[k]
			mu_new[k] = np.reshape(mu[k], (10,10))
			cv2.imwrite('../results/mogm/Mean_iteration_' + str(no+1) + '_k_' + str(k) + '.jpg',mu_new[k])

		sigma_new = np.zeros((K,10,10))
		sigma_temp = np.zeros((K,100))
		for k in range(K):
			sigma_num = np.matmul(r_ik[k], np.square(x-mu[k]))
			print(sigma_num.shape)
			sigma_temp[k] = sigma_num/sum_ri[k]
			sigma[k] = np.diag(sigma_temp[k])
			sigma_new[k] = np.sqrt(np.reshape(sigma_temp[k], (10,10)))
			cv2.imwrite('../results/mogm/Covar_iteration_' + str(no+1) + '_k_' + str(k) + '.jpg',sigma_new[k])

	print(r_ik)
	print('Finished MOGM')
	return lmbda, mu, sigma

def T_D(face, Train):
	x, mu_mle, covar_mle = MLE(face, Train)
	print('\nT-DISTRIBUTION - EM Algorithm \n')
	D = len(x)
	shape = len(x[0])
	print('No of images : ', D, ', Shape : ', shape)

	mu = x[0]
	sigma = np.diag(covar_mle)
	v = 6.6
	print('Mu shape : ', mu.shape, ', Sigma shape : ', sigma.shape ,', v : ', v)
	no_of_iterations = 10
	print('No of iterations : ', no_of_iterations)
	for no in range(no_of_iterations):
		# print('\n Iteration ', (no+1))
		
		# E-Step
		e_num = v + D
		x_mu = x - mu
		inv = np.linalg.inv(sigma)  
		mat = np.matmul(x_mu, inv)
		e_denom = v + np.diag(np.matmul(mat, x_mu.T))
		E_h_i = np.divide(e_num,e_denom)
		print(E_h_i.shape)

		# M-Step
		mu_num = np.dot(E_h_i,x)
		den = np.sum(E_h_i, axis = 0)
		mu = np.divide(mu_num, den)

		mu_new = np.reshape(mu, (10, 10))
		cv2.imwrite('../results/t-distribution/Mean_iteration' + str(no+1) + '.jpg', mu_new)

		x_mu = x -mu
		sigma_num = np.matmul(E_h_i,  np.square(x_mu))
		sigma = np.divide(sigma_num, den)
		sigma_new = np.reshape(sigma, (10, 10))
		sigma_new = np.sqrt(sigma_new)
		sigma = np.diag(sigma)
		cv2.imwrite('../results/t-distribution/Covar_iteration' + str(no+1) + '.jpg',sigma_new)
	return mu, sigma, v

def F_A(face, Train):
	x, mu_mle, covar_mle = MLE(face, Train)
	print('\nFACTOR ANALYSIS- EM Algorithm \n')
	num_of_images = len(x)
	D = len(x[0])
	print('No of images : ', num_of_images, ', Shape : ', D)

	K = 10 #Dimension for phi
	mu = np.reshape(mu_mle, (100))
	mu_print = np.reshape(mu_mle, (10, 10))
	phi = np.random.random((D, K))
	sigma = np.diag(np.reshape(covar_mle, (100)))
	print('Mu shape : ', mu.shape, ', Sigma shape : ', sigma.shape ,', phi shape : ', phi.shape)
	cv2.imwrite('../results/factor-analysis/Mean' + '.jpg', mu_print)
	no_of_iterations = 10
	for no in range(no_of_iterations):
		#### E-Step
		sigma_inv = inv(sigma)
		sigI_phi = np.matmul(sigma_inv, phi)
		phiT_sigI_phi = np.matmul(np.transpose(phi), sigI_phi)
		phiT_sigI_phiI = inv(phiT_sigI_phi + np.identity(K))
		phiT_sig = np.matmul(np.transpose(phi), sigma_inv)
		phi_sigma_complete = np.matmul(phiT_sigI_phiI, phiT_sig)
		x_mu = np.zeros((1100, 100))
		x_mu = x - mu
		E_hi = np.matmul(x_mu, np.transpose(phi_sigma_complete))
		E_hi = np.reshape(E_hi, (num_of_images, K, 1))

		e_hi_hiT_term2 = np.zeros((num_of_images, K, K))
		for i in range(num_of_images):
			e_hi_hiT_term2[i] = np.matmul(E_hi[i], np.transpose(E_hi[i]))
		e_hi_hiT = e_hi_hiT_term2 + phiT_sigI_phiI

		#### M-Step
		phi_term1temp = np.zeros((num_of_images, D, K))
		x_mu = np.reshape(x_mu, (num_of_images, 100, 1))
		for i in range(num_of_images):
			phi_term1temp[i] = np.matmul(x_mu[i], np.transpose(E_hi[i]))
		phi_term1 = np.sum(phi_term1temp, axis = 0)
		phi_term2 = inv(np.sum(e_hi_hiT, axis = 0))
		phi = np.matmul(phi_term1, phi_term2)
		x = np.reshape(x, (num_of_images, 100, 1))
		sigma_term1 = np.zeros((num_of_images, 100, 100))
		sigma_term2 = np.zeros((num_of_images, 100, 100))
		sigma_temp = np.zeros((num_of_images, 100))
		for i in range(num_of_images):
			sigma_term1[i] = np.matmul(x_mu[i], np.transpose(x_mu[i]))
			temp = np.matmul(E_hi[i], np.transpose(x[i]))
			sigma_term2[i] = np.matmul(phi, temp)
			sigma_temp[i] = np.diag(sigma_term1[i] - sigma_term2[i])
		x = np.reshape(x, (num_of_images, 100))
		sigma = np.sum(sigma_temp, axis = 0)
		sigma = sigma/num_of_images
		sigma_print = np.sqrt(np.reshape(sigma, (10, 10)))
		sigma = np.diag(np.reshape(sigma, (100)))
		cv2.imwrite('../results/factor-analysis/Sigma_iteration' + str(no+1) + '.jpg', sigma_print)
	return mu, sigma, phi

def test_MLE():
	print('Testing MLE')

	a, mu_face, covar_face = MLE(face = True, Train = True)
	b, mu_nonface, covar_nonface = MLE(face = False, Train = True)
	image_face = load_data(face = True, Train = False)
	image_nonface = load_data(face = False, Train = False)
	print(image_face.shape)
	print(image_nonface.shape)
	covar_face = np.diag(covar_face)
	covar_nonface = np.diag(covar_nonface)
	image = np.append(image_face, image_nonface, axis = 0)
	num_of_images = len(image)
	print('Image shape : ', image.shape)
	print('Mu face shape : ', mu_face.shape, ', Sigma face shape : ', covar_face.shape)
	print('Mu non-face shape : ', mu_nonface.shape, ', Sigma non-face shape : ', covar_nonface.shape)

	face_norm = multivariate_normal.logpdf(image, mu_face, covar_face)
	nonface_norm = multivariate_normal.logpdf(image, mu_nonface, covar_nonface)

	face_norm = np.reshape(face_norm, (num_of_images))
	nonface_norm = np.reshape(nonface_norm, (num_of_images))
	print(face_norm.shape)
	print(nonface_norm.shape)

	modules.PlotROC(face_norm, nonface_norm, num_of_images, no_roc = 5)

def test_MOGM():
	

	image_face = load_data(face = True, Train = False)
	image_nonface = load_data(face = False, Train = False)
	lmbda_face, mu_face, covar_face = MOGM(face = True, Train = True)
	lmbda_nonface, mu_nonface, covar_nonface = MOGM(face = False, Train = True)

	print('\nTesting MOGM\n')
	image = np.append(image_face, image_nonface, axis = 0)
	num_of_images = len(image)
	print('Image shape : ', image.shape)
	print('Mu face shape : ', mu_face.shape, ', Sigma face shape : ', covar_face.shape)
	print('Mu non-face shape : ', mu_nonface.shape, ', Sigma non-face shape : ', covar_nonface.shape)
	print('Lambda non-face shape : ', lmbda_nonface.shape, ', lmbda non-face shape : ', lmbda_nonface.shape)

	norm_0_face = lmbda_face[0]*multivariate_normal.pdf(image, mu_face[0], covar_face[0])
	norm_1_face = lmbda_face[1]*multivariate_normal.pdf(image, mu_face[1], covar_face[1])
	face_norm = norm_0_face + norm_1_face

	norm_0_nonface = lmbda_nonface[0]*multivariate_normal.pdf(image, mu_nonface[0], covar_nonface[0])
	norm_1_nonface = lmbda_nonface[1]*multivariate_normal.pdf(image, mu_nonface[1], covar_nonface[1])
	nonface_norm = norm_0_nonface + norm_1_nonface


	print('prob_face : ', face_norm.shape)
	print('prob_face : ', nonface_norm.shape)

	nonface_norm[nonface_norm == 0] = 3e-220
	face_norm[face_norm == 0] = 3e-220
	prob_face = np.log((face_norm)/(face_norm + nonface_norm))
	prob_nonface = np.log((nonface_norm)/(face_norm + nonface_norm))

	modules.PlotROC(prob_face, prob_nonface, num_of_images, no_roc = 5)
	
def multivariate_t_distribution(x,mu,Sigma,v,d):
	term1 = -(1/2)*np.log(np.prod(np.diag(np.sqrt(Sigma))))
	x_mu = x-mu
	temp1 = np.matmul(x_mu, inv(Sigma))
	power_dot = np.diag(np.matmul(temp1, np.transpose(x_mu)))
	term2 = -((d+v)/2)*np.log(1 + (power_dot/v))
	d = term1 + term2
	return d

def test_TD():
	image_face = load_data(face = True, Train = False)
	image_nonface = load_data(face = False, Train = False)
	mu_face, covar_face, v_face = T_D(face = True, Train = True)
	mu_nonface, covar_nonface, v_nonface = T_D(face = False, Train = True)
	print('\nTesting TD\n')

	image = np.append(image_face, image_nonface, axis = 0)
	num_of_images = len(image)
	print('Image shape : ', image.shape)
	print('Mu face shape : ', mu_face.shape, ', Sigma face shape : ', covar_face.shape)
	print('Mu non-face shape : ', mu_nonface.shape, ', Sigma non-face shape : ', covar_nonface.shape)
	print('v face : ', v_face, ', v non-face : ', v_nonface)

	face_norm = multivariate_t_distribution(image, mu_face, covar_face, v_face, num_of_images)
	nonface_norm = multivariate_t_distribution(image, mu_nonface, covar_nonface, v_nonface, num_of_images)

	modules.PlotROC(face_norm, nonface_norm, num_of_images, no_roc = 5)

def test_FA():
	image_face = load_data(face = True, Train = False)
	image_nonface = load_data(face = False, Train = False)
	mu_face, covar_face, phi_face = F_A(face = True, Train = True)
	mu_nonface, covar_nonface, phi_nonface = F_A(face = False, Train = True)

	print('\nTesting FA\n')

	image = np.append(image_face, image_nonface, axis = 0)
	num_of_images = len(image)
	print('Image shape : ', image.shape)
	print('Mu face shape : ', mu_face.shape, ', Sigma face shape : ', covar_face.shape)
	print('Mu non-face shape : ', mu_nonface.shape, ', Sigma non-face shape : ', covar_nonface.shape)
	print('Phi non-face shape : ', phi_face.shape, ', Phi non-face shape : ', phi_nonface.shape)

	new_covar_face = np.add(np.matmul(phi_face, np.transpose(phi_face)), covar_face)	
	new_covar_nonface = np.add(np.matmul(phi_nonface, np.transpose(phi_nonface)), covar_nonface)

	face_norm = multivariate_normal.logpdf(image, mu_face, new_covar_face)
	nonface_norm = multivariate_normal.logpdf(image, mu_nonface, new_covar_nonface)
	modules.PlotROC(face_norm, nonface_norm, num_of_images, no_roc = 5)

if __name__ == '__main__':
    FLAGS = get_args()
    if FLAGS.save:
        save_data()
    if FLAGS.load:
        load_data(True, True)
    if FLAGS.gm:
        MLE(False, True)
    if FLAGS.mogm:
    	MOGM(False, True)
    if FLAGS.td:
        T_D(False, True)
    if FLAGS.fa:
    	F_A(False, True)
    if FLAGS.test_gm:
    	test_MLE()
    if FLAGS.test_mogm:
    	test_MOGM()
    if FLAGS.test_td:
    	test_TD()
    if FLAGS.test_fa:
    	test_FA()
