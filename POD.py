# ---------------------------------- Imports ----------------------------------
import numpy as np
import tensorflow as tf
#------------------------------------------------------------------------------


class POD:
    """This class contains all algorithms for the proper orthogonal
       decomposition (POD)."""
       
    POD_snapshots=[]
    Pod_snapshots_batch=[]
    basis_fts_matrix=[]
    basis_fts_matrix_time=[]
    basis_fts_matrix_batch=[]
    basis_fts_matrix_time_batch=[]
    information_content=[]
    information_content_batch=[]
    #reduced_state = []
       
#    def __init__(self):
#        """
#        """
    
    def POD_batch_data(self, n_time, stepsize):
        """ Batches the data. This is useful for larger data sets.

        Args:
        n_time = number of time steps
        stepsize = number of samples in the batch
        Returns:

        """
        for j in range (0,n_time,stepsize):
            if((j+stepsize <= n_time)and(n_time-(j+stepsize) > 1)):
                self.Pod_snapshots_batch.append(self.Pod_snapshots[:,:,j:j+stepsize+1,:])
            elif((j+stepsize <= n_time)and(int(n_time-(j+stepsize)) == 1)):
                self.Pod_snapshots_batch.append(self.Pod_snapshots[:,:,j:,:])
            elif((j+stepsize > n_time)and(int(n_time-((j-1)+stepsize)) > 1)):
                self.Pod_snapshots_batch.append(self.Pod_snapshots[:,:,j:,:])
    
    def POD_parameters(self, n_variables=1, threshold_percent=[99.999]):
        """ Performs the POD for the parameters. This POD algorithm should be
            used for time-independent (steady-state) applications

        Args:
        n_variables = number of variables for which the POD needs to be performed
        threshold_percent = vector of dimension of the number of variables that
                            defines the desired accuracy

        Returns:

        """
        svd =[]
        for i in range(n_variables):
        # Perform the Singular Value Decomposition
            #svd.append(np.linalg.svd(self.Pod_snapshots[i,:,:])) 
            svd.append(np.linalg.svd(self.Pod_snapshots[i,:,:],  full_matrices=False)) 
    
        for n in range(n_variables):
            sum_eigen_full = np.sum(svd[n][1]**2)
            self.information_content.append([])
            index=0
            
            print('Selecting the basis functions for the reduced basis')
            for i in range(len(svd[n][1])):
                sum_eigen = np.sum(svd[n][1][:i]**2) 
                if((sum_eigen/sum_eigen_full)<=(threshold_percent[n]/100)):
                    self.information_content[n].append(sum_eigen/sum_eigen_full)
                    index+=1
    
            self.basis_fts_matrix.append(np.copy(svd[n][2][:index,:]))
                     
       
    def POD_parameters_one_time_step(self, n_variables=1,timestep=-1, 
                       threshold_percent=[99.999]):
        """ Performs the POD for the parameters. This POD algorithm should be
            used for applications with a single time step.

        Args:
        n_variables = number of variables for which the POD needs to be performed
        timestep = timestep at which you want to extract the snapshot
        threshold_percent = vector of dimension of the number of variables that
                            defines the desired accuracy

        Returns:

        """
        svd =[]
        for i in range(n_variables):
        # Perform the Singular Value Decomposition
            # svd.append(np.linalg.svd(self.Pod_snapshots[i,:,timestep,:])) 
            svd.append(np.linalg.svd(self.Pod_snapshots[i,:,:])) 
    
        for n in range(n_variables):
            sum_eigen_full = np.sum(svd[n][1]**2)
            self.information_content.append([])
            index=0
            
            print('Selecting the basis functions for the reduced basis')
            for i in range(len(svd[n][1])):
                sum_eigen = np.sum(svd[n][1][:i]**2) 
                if((sum_eigen/sum_eigen_full)<(threshold_percent[n]/100)):
                    self.information_content[n].append(sum_eigen/sum_eigen_full)
                    index+=1
    
            self.basis_fts_matrix.append(np.copy(svd[n][2][:index,:]))

#        self.basis_fts_matrix = np.zeros([index, len(self.Pod_snapshots[0])])
#
#        for j in range(index):
#            i = 0
#            while (i<=j):
#                self.basis_fts_matrix[j] = np.dot(svd[0][i][j], 
#                                     self.basis_fts_matrix[i])
#                i+=1
         
            
    def POD_parameters_time_batch(self, n_param_rel, n_variables=1,timestep=-1, 
                       threshold_percent=[99.999], threshold_percent_time=[99.999]):
        """ Performs the POD for the parameters and the time in a two step
            procedure. We implement the two step instead of the one step 
            procedure for efficiency reasons for all batches. This POD algorithm should be
            used for time-dependent (transient) applications. 
            
        Args:
        n_param_rel = number of realizations per parameter
        n_variables = number of variables for which the POD needs to be performed
        timestep = timestep at which you want to extract the snapshot
        threshold_percent = vector of dimension of the number of variables that
                            defines the desired accuracy
        threshold_percent_time = vector of dimension of the number of variables 
                                 that defines the desired accuracy for the time

        Returns:

        """  
        # Clean the list in case this is a re-run
        self.basis_fts_matrix_batch=[]
        self.basis_fts_matrix_time_batch=[]
        self.information_content_batch = []
        self.reduced_state = []
        
        
        for b in range(len(self.Pod_snapshots_batch)):
            self.basis_fts_matrix_time_batch.append([])
            self.basis_fts_matrix_batch.append([])
            self.information_content_batch.append([])
            self.reduced_state.append([])
            
            print('Starting POD for batch ' + str(b))
        
            for n in range(n_variables):
                self.basis_fts_matrix_time_batch[b].append([])
                self.basis_fts_matrix_batch[b].append([])
                self.information_content_batch[b].append([])
                self.reduced_state[b].append([])
    
                print('Performing the singular value decomposition for the time-\
                      trajectory for every parameter')

                for i in range(n_param_rel): 
                    index = 0
     
                    # Perform the Singular Value Decomposition
                    svd = np.linalg.svd(np.asarray(self.Pod_snapshots_batch[b])[n, i, :, :], 
                                        full_matrices=False)
            
                    sum_eigen_full = np.sum(svd[1]**2)

                    print('Selecting the basis functions for the reduced basis')
                    for j in range(len(svd[1])):
                        sum_eigen = np.sum(svd[1][:j]**2) 
                        if((sum_eigen/sum_eigen_full)<(threshold_percent_time[n]/100)):
                            self.information_content_batch[b][n].append(sum_eigen/sum_eigen_full)
                            index+=1
    
                    self.basis_fts_matrix_time_batch[b][n].extend(np.copy(svd[2][:index]))
     

                print('Performing the singular value decomposition for the parameters')
                index = 0
    
                # Perform the Singular Value Decomposition
                svd = np.linalg.svd(np.asarray(self.basis_fts_matrix_time_batch[b][n]),
                                    full_matrices=False)
           
                # Select the basis functions
                sum_eigen_full = np.sum(svd[1]**2)

                print('Selecting the basis functions for the reduced basis')
                for j in range(len(svd[1])):
                    sum_eigen = np.sum(svd[1][:j]**2) 
                
                    if((sum_eigen/sum_eigen_full)<(threshold_percent_time[n]/100)):
                        self.information_content_batch[b][n].append(sum_eigen/sum_eigen_full)
                        index+=1
    
                self.basis_fts_matrix_batch[b][n].extend(np.copy(svd[2][:index]))
                #self.reduced_state[b][n].extend(np.copy(svd[0][:index]))
            
    def POD_parameters_time(self, n_param_rel, n_variables=1,timestep=-1, 
                       threshold_percent=[99.999], threshold_percent_time=[99.999]):
        """ Performs the POD for the parameters and the time in a two step
            procedure. We implement the two step instead of the one step 
            procedure for efficiency reasons. This POD algorithm should be
            used for time-dependent (transient) applications. 
            
        Args:
        n_param_rel = number of realizations per parameter
        n_variables = number of variables for which the POD needs to be performed
        timestep = timestep at which you want to extract the snapshot
        threshold_percent = vector of dimension of the number of variables that
                            defines the desired accuracy
        threshold_percent_time = vector of dimension of the number of variables 
                                 that defines the desired accuracy for the time

        Returns:

        """ 
        # Clean the list in case this is a re-run
        self.basis_fts_matrix=[]
        self.basis_fts_matrix_time=[]
        self.information_content=[]
        
        for n in range(n_variables):
            self.basis_fts_matrix_time.append([])
            self.basis_fts_matrix.append([])
            self.information_content.append([])
    
            print('Performing the singular value decomposition for the time-\
                  trajectory for every parameter')

            for i in range(n_param_rel): 
                index = 0
     
                # Exclude the intial solution
                # Perform the Singular Value Decomposition
                svd = np.linalg.svd(self.Pod_snapshots[n, i, :, :], full_matrices=False)
                #svd = tf.linalg.svd(self.Pod_snapshots[n, i, :, :],full_matrices=True)    
                # Select the basis functions
                sum_eigen_full = np.sum(svd[1]**2)
                #sum_eigen_full = np.sum(svd[0]**2)

                print('Selecting the basis functions for the reduced basis')
                for j in range(len(svd[1])):
                #for j in range(len(svd[0])):
                    sum_eigen = np.sum(svd[1][:j]**2) 
                    #sum_eigen = np.sum(svd[0][:j]**2) 
                    if((sum_eigen/sum_eigen_full)<(threshold_percent_time[n]/100)):
                        self.information_content[n].append(sum_eigen/sum_eigen_full)
                        index+=1
    
                self.basis_fts_matrix_time[n].extend(np.copy(svd[2][:index]))
     

            print('Performing the singular value decomposition for the parameters')
            index = 0
    
            # Perform the Singular Value Decomposition
            svd = np.linalg.svd(np.asarray(self.basis_fts_matrix_time[n]),full_matrices=False)
            #svd = tf.linalg.svd(np.asarray(self.basis_fts_matrix_time[n]), full_matrices=True)   
    
            # Select the basis functions
            sum_eigen_full = np.sum(svd[1]**2)
            #sum_eigen_full = np.sum(svd[0]**2)

            print('Selecting the basis functions for the reduced basis')
            for j in range(len(svd[1])):
            #for j in range(len(svd[0])):
                sum_eigen = np.sum(svd[1][:j]**2) 
                #sum_eigen = np.sum(svd[0][:j]**2) 
                if((sum_eigen/sum_eigen_full)<(threshold_percent_time[n]/100)):
                    self.information_content[n].append(sum_eigen/sum_eigen_full)
                    index+=1
    
            self.basis_fts_matrix[n].extend(np.copy(svd[2][:index]))

        

