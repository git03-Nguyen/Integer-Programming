import numpy as np
from scipy.optimize import LinearConstraint
from scipy.optimize import milp

# Nhập hệ số của hàm mục tiêu, dạng tìm GTNN:
c = np.array([230, 250, 280, 250, 300])

# Nhập hệ số của các ràng buộc (đã được chuẩn hoá) vào ma trận
A = np.array([[1, 0, 0, 1, 0],
              [1, 1, 0, 0, 1],
              [0, 1, 1, 1, 0],
              [0, 0, 1, 0, 1]])

# Chặn dưới của các ràng buộc:
b_l = np.array([9, 20, 11, 17])

# Chặn trên của các ràng buộc: không có chặn trên
b_u = np.inf * np.ones_like(b_l)

# Khởi tạo đối tượng hệ ràng buộc scipy.optimize.LinearConstraint:
constraints = LinearConstraint(A, b_l, b_u)

# Mặc định, các biến x_i sẽ được hiểu là không âm nên ta không cần cung cấp thêm thông tin đó.
# Ngoài ra, tất cả các biến x_i  đều là số nguyên:
integrality = np.ones_like(c)

# Kết quả bài toán:
res = milp(c=c, constraints=constraints, integrality=integrality)
print(res.x)



