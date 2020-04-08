import numpy as np
import matplotlib.pyplot as plt
import math

def PlotROC(prob_face, prob_nonface, num_of_images, no_roc=100):
	term1 = prob_face - prob_nonface
	print('Term1 shape: ', term1.shape)

	threshold = np.linspace(np.min(term1), np.max(term1), no_roc)
	print('threshold.shape : ', threshold.shape)

	TP = []
	TN = []
	FP = []
	FN = []

	for k in range(len(threshold)):
		TP.append(term1[:100] >= threshold[k])
		FN.append(term1[:100] < threshold[k])
		TN.append(term1[100:200] < threshold[k])
		FP.append(term1[100:200] >= threshold[k])
	TP = np.sum(TP, axis = 1)
	TN = np.sum(TN, axis = 1)
	FP = np.sum(FP, axis = 1)
	FN = np.sum(FN, axis = 1)
	# print(TP, TN, FP, FN)
	FPRate = np.sum(prob_face[100:200] > prob_nonface[100:200])/100
	FNRate = np.sum(prob_face[:100] < prob_nonface[:100])/100
	MCRate = (FPRate + FNRate)/2

	print('False Positive Rate : ', FPRate)
	print('False Negative Rate : ', FNRate)
	print('Misclassification Rate : ', MCRate)

	plt.plot(FP/100, TP/100, marker='o')
	plt.title('Receiver operating characteristic (ROC) curve')
	plt.xlabel('False Positive Rate (1 - Specificity)')	
	plt.ylabel('True Positive Rate (Sensitivity)')
	plt.xlim(0,1)
	plt.ylim(0,1)	
	plt.show()

