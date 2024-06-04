import numpy as np
from fractions import Fraction
P = np.array([[0,1,0.5],[0,0,0.5],[1,0,0]])

v = np.array([1/3,1/3,1/3])

# Compute the eigenvectors of P
eigen_value,v_star = np.linalg.eig(P)

# find the eigenvector corresponding to the eigenvalue 1
v_star_1 = v_star[:,np.argmin(np.abs(eigen_value-1))]

# normalize the eigenvector
v_star_1 = v_star_1 / np.sum(v_star_1)

print(f"v_star_1 = {v_star_1}")

print(f"eigen_value = {eigen_value}")


v_star = v_star[:,0]

# normalize the eigenvector
v_star = v_star / np.sum(v_star)

for i in range(20):
    v = P @ v
    print(f"v_{i+1} = {[Fraction(x).limit_denominator() for x in v]}")

print(f"v_star = {v_star}")