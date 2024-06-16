import numpy as np
from scipy.optimize import LinearConstraint
from scipy.optimize import milp

# Nhap he so ham muc tieu, dang tim min
c = np.array([230, 250, 280, 250, 300])

# Nhap he so cua cac rang buoc
A = np.array([[1, 0, 0, 1, 0],
              [1, 1, 0, 0, 1],
              [0, 1, 1, 1, 0],
              [0, 0, 1, 0, 1]])

# Chan duoi cua cac rang buoc
b_l = np.array([9, 20, 11, 17])
# Chan tren cua cac rang buoc (khong co, vi tat ca deu la >=)
b_u = np.inf * np.ones_like(b_l)

# Khoi tao doi tuong he rang buoc scipy.optimize.LinearConstraint:
constraints = LinearConstraint(A, b_l, b_u)

# Mac dinh, cac bien x_i se duoc hieu la khong am nen ta khong can cung cap them thong tin do.
# Ngoai ra, tat ca cac bien x_i deu la so nguyen:
integrality = np.ones_like(c)

# Ket qua bai toan:
res = milp(c=c, constraints=constraints, integrality=integrality)
print(res.x)

# Ket qua ham muc tieu:
print(res.fun)





