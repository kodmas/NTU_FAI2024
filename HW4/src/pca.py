import numpy as np
import matplotlib.pyplot as plt

"""
Implementation of Principal Component Analysis.
"""
class PCA:
    def __init__(self, n_components: int) -> None:
        self.n_components = n_components
        self.mean = None
        self.components = None

    def fit(self, X: np.ndarray) -> None:
        #TODO: 10%
        # Hint: Use existing method to calculate covariance matrix and its eigenvalues and eigenvectors
        self.mean = np.mean(X,axis=0)

        # also transform the mean vector to see the image
        # mean_image = self.mean.reshape(61,80)
        # plt.imshow(mean_image,cmap='gray')
        # plt.title("Mean Image")
        # plt.savefig("mean_image.png")
        # plt.show()


        X -= self.mean

        # calculate eigen vectors
        # X_cov = X.T @ X
        # eigen_values,eigen_vectors = np.linalg.eigh(X_cov)
        
        ## faster way to calculate eigen vectors
        covariance_matrix = np.cov(X,rowvar=False)
        eigen_values,eigen_vectors = np.linalg.eigh(covariance_matrix)

        # print(eigen_values)
    
        sorted_indices = np.argsort(eigen_values)[::-1]

        eigen_values = eigen_values[sorted_indices]
        
        # sort the eigen vectors based on the sorted indices 
        eigen_vectors = eigen_vectors[:,sorted_indices]

        self.components = eigen_vectors[:,0:self.n_components]

        # transform each eigen values back to original space and see the image
        # for i in range(4):
        #     eigen_vector = eigen_vectors[:,i]
        #     eigen_vector = eigen_vector.reshape(61,80)
        #     plt.imshow(eigen_vector,cmap='gray')
        #     plt.title(f"Eigen Vector {i}")
        #     plt.savefig(f"eigen_vector_{i}.png")
        #     plt.show()



    def transform(self, X: np.ndarray) -> np.ndarray:
        #TODO: 2%
        # Hint: Use the calculated principal components to project the data onto a lower dimensional space
        X_centered = X - self.mean
        return X_centered @ self.components
        # return self.components.T @ X_centered.T

    def reconstruct(self, X):
        # Hint: Use the calculated principal components to reconstruct the data back to its original space
        #TODO: 2%
        return self.transform(X) @ self.components.T + self.mean
        # return self.transform(X).T @ self.components.T + self.mean
