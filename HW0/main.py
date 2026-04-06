''' Ayaan Khan 
    A20505209
    CS584 - S26
'''
# PART A

import numpy as np

# Define vectors
p = np.array([2, -1, 4])
q = np.array([0, 3, 5])
r = np.array([1, -2, 2])

# 1. 
result_1 = 3 * p + 2 * q
print("1. 3p + 2q =", result_1)

# 2.
norm_p = np.linalg.norm(p)
unit_p = p / norm_p
print("2. Unit vector of p =", unit_p)

# 3. 
angle_y = np.arccos(p[1] / norm_p)  
print("3. ||p|| =", norm_p, "| Angle with y-axis =", np.degrees(angle_y), "degrees")

# 4. 
direction_cosines = unit_p
print("4. Direction cosines =", direction_cosines)

# 5. 
dot_pq = np.dot(p, q)
norm_q = np.linalg.norm(q)
angle_pq = np.arccos(dot_pq / (norm_p * norm_q))
print("5. Angle between p and q =", np.degrees(angle_pq), "degrees")

# 6. 
print("6. p · q =", dot_pq, "| q · p =", np.dot(q, p))

# 8. 
scalar_proj = dot_pq / norm_p
print("8. Scalar projection of q onto p =", scalar_proj)

# 9.
v = np.array([1, 2, 0])  # Verify p · v = 0
print("9. Perpendicular vector =", v, "| p · v =", np.dot(p, v))

# 10. 
cross_pq = np.cross(p, q)
cross_qp = np.cross(q, p)
print("10. p × q =", cross_pq, "| q × p =", cross_qp)

# 12. 
matrix = np.column_stack([p, q, r])
rank = np.linalg.matrix_rank(matrix)
print("12. Rank of [p, q, r] =", rank, "| Linearly independent =", (rank == 3))

# PART B
# Define matrices and vector
X = np.array([[2, 1, 0], [-1, 3, 4], [3, 2, -2]])
Y = np.array([[4, -1, 2], [3, 0, -3], [1, 2, 1]])
Z = np.array([[2, 0, -1], [1, 4, 5], [3, 1, 2]])
s = np.array([-1, 4, 0])

# 1.
result_1 = X + 2 * Y
print("1. X + 2Y =\n", result_1)

# 2. 
XY = X @ Y
YX = Y @ X
print("2. XY =\n", XY, "\n\nYX =\n", YX)

# 3. 
print("3. (XY)^T =\n", XY.T, "\n\nY^T X^T =\n", Y.T @ X.T)

# 4. 
det_X = np.linalg.det(X)
det_Z = np.linalg.det(Z)
print("4. |X| =", det_X, "| |Z| =", det_Z)

# 6. 
X_inv = np.linalg.inv(X)
Y_inv = np.linalg.inv(Y)
print("6. X⁻¹ =\n", X_inv, "\n\nY⁻¹ =\n", Y_inv)

# 7. 
Z_inv = np.linalg.inv(Z)
print("7. Z⁻¹ =\n", Z_inv)

# 8. 
Xs = X @ s
print("8. Xs =\n", Xs)

# 12.
t_Y = np.linalg.solve(Y, s)
print("12. Solution for Yt = s:", t_Y)

# 13. 
t_Z = np.linalg.solve(Z, s)
print("13. Solution for Zt = s:", t_Z)

# PART C

# Define matrices
M = np.array([[3, 2], [-1, 4]])
N = np.array([[5, -3], [-3, 6]])
P = np.array([[2, 4], [4, 8]])

# 1. 
eigvals_M, eigvecs_M = np.linalg.eig(M)
print("1. Eigenvalues of M =", eigvals_M)
print("   Eigenvectors of M =\n", eigvecs_M)

# 2.
dot_M = np.dot(eigvecs_M[:, 0], eigvecs_M[:, 1])
print("2. Dot product of M's eigenvectors =", dot_M)

# 3.
eigvals_N, eigvecs_N = np.linalg.eig(N)
dot_N = np.dot(eigvecs_N[:, 0], eigvecs_N[:, 1])
print("3. Dot product of N's eigenvectors =", dot_N)

# 4.
print("4. N is symmetric → Eigenvectors are orthogonal (dot product ≈ 0).")

# 5. 
trivial_sol = np.array([0, 0])
print("5. Trivial solution to Pt = 0 =", trivial_sol)

# 6.
null_space = scipy.linalg.null_space(P)
non_trivial1 = null_space[:, 0]  # First basis vector
non_trivial2 = 2 * non_trivial1   # Second non-trivial solution
print("6. Non-trivial solutions:\n", non_trivial1, "\n", non_trivial2)

# 7. 
det_M = np.linalg.det(M)
if det_M != 0:
    trivial_sol_M = np.linalg.solve(M, np.zeros(2))
    print("7. Only solution to Mt = 0 =", trivial_sol_M, "| Reason: M is invertible (det ≠ 0).")
else:
    print("7. M is singular → Infinitely many solutions.")