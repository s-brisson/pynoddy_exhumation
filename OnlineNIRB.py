"""
This is an example file to demonstrate the implementation of the online stage
of the non-intrusive RB method.
"""

import numpy as np
import sys
from tensorflow import keras
# import json
# from tensorflow.keras.models import model_from_json


#sys.path.append("/home/denise/git/NI-RB")
import POD as pod
pod=pod.POD()

#import NeuralNetwork as nn
#nn=nn.NeuralNetwork()

#sys.path.append("/Users/denise/git/FastElephant/FileOperations/PythonScripts")
#import MOOSEFiles as mfiles

# %% 
class OnlineNIRB():
    """This class handles the online stage of the non-intrusive RB method."""
# pickle_file = '/Users/denise/git/DwarfElephant_Material_Additional_Code/DispZModel.pickle'
# basis_file = '/Users/denise/git/DwarfElephant_Material_Additional_Code/BasisFunctionDispZ_'
# number_of_variables = 1
# n_b = 7 # number of batches
# load_bfs = True
    networks_batch = []
    pod.basis_fts_matrix_batch = []
    # def load_all_files(self,pickle_file,basis_fts_file, n_var=1, n_batch=0, load_bfs=False):
    #     # Load the offline model
    #     with open(pickle_file, 'rb') as handle:
    #         self.networks_batch = pickle.load(handle)
    
    #     # Load the basis functions (only required if you want to project to full space)    
    #     if(load_bfs==True):
    #         pod.basis_fts_matrix_batch=[]
    #         if(n_batch==0):
    #             pod.basis_fts_matrix_batch.append(
    #                 np.load(basis_fts_file+'.npy'))
    #         else:
    #             for i in range(n_batch):
    #                 pod.basis_fts_matrix_batch.append(
    #                         np.load(basis_fts_file+str(i)+ '.npy'))
                    
    def load_all_files_tf(self,model_file,basis_fts_file, n_var=1, n_batch=0, load_bfs=False):
        # Load the offline model
        # with open('/Users/denise/git/papers/GrSbk/Data/Press3_90Model_total_duration/saved_model.pb', 'r') as json_file:
        #  self.networks_batch = model_from_json(json_file.read())
        self.networks_batch = keras.models.load_model(model_file)
        # self.networks_batch = keras.models.load_model(model_file, custom_objects={'custom_loss': nn.custom_loss})
    
        # Load the basis functions (only required if you want to project to full space)    
        if(load_bfs==True):
            pod.basis_fts_matrix_batch=[]
            if(n_batch==0):
                pod.basis_fts_matrix_batch.append(
                    np.load(basis_fts_file+'.npy'))
            else:
                for i in range(n_batch):
                    pod.basis_fts_matrix_batch.append(
                            np.load(basis_fts_file+str(i)+ '.npy'))
                    
    def get_basis_functions(self):
        return pod.basis_fts_matrix_batch
    
# %%
    def online_stage_full_state(self,mu_online, number_of_variables=1,batch=0, no_batches=True,load_bfs=False, tf=False):
        """This function executes the online stage of the non-instrusive RB method.
        
        Args:
            mu_online = parameters for the online solve
            number_of_variables = number_of_variables
            batch = number of the batch
        Returns:
            full_solutions = solution vector of the full FE space state
                
        """
        full_solutions = [] 
        for n in range (number_of_variables):
            #print('Predicting the reduced coefficients')
            # Compute reduced coefficients
            if(no_batches==True and tf==False):
                rb_coeff = self.networks_batch[n](mu_online,training=False)
            elif(no_batches==True and tf==True):
                 rb_coeff = self.networks_batch(mu_online, training=False)
            elif(no_batches==False and tf==True):
                rb_coeff = self.networks_batch[batch](mu_online, training=False)
            else:
                rb_coeff = self.networks_batch[batch][n](mu_online, training=False)
            
            #print('Projecting the reduced solution onto the high-dimensional space')
            # Project to full space
            if(load_bfs==True):
                if (no_batches==True):
                    full_solutions.append(np.dot(rb_coeff[0], pod.basis_fts_matrix_batch[n]))
                else:
                    full_solutions.append(np.dot(rb_coeff[0], pod.basis_fts_matrix_batch[batch][n]))
            #print(rb_coeff)
            
        if(load_bfs==False):
            return rb_coeff
        else:
            return full_solutions

# %% Example Run for single online execution. Careful with picking the right time
#    batches in the case of a batched data set.
    
# Define online parameters    
#mu_online = [np.array([4.02100000e+03,50925925925.925934,
#                      1.31862140e-07,3.797629636016879e-16,
#                      8.40090090e-02, 1.49450486e-17,
#                      2.80030030e-03])]
#full_solutions=online_stage_full_state(mu_online, number_of_variables,6)
