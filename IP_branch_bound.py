import numpy as np
from scipy.optimize import LinearConstraint
from scipy.optimize import milp

# Hàm giải bài toán quy hoạch tuyến tính
def solve_LP(c, A, b_l, b_u):

    # Khởi tạo đối tượng hệ ràng buộc scipy.optimize.LinearConstraint:
    constraints = LinearConstraint(A, b_l, b_u)

    # Mặc định, các biến x_i >= 0
    # Thêm điều kiện các biến x_i là số thực:
    integrality = np.zeros_like(c)

    # Giải bài toán:
    res = milp(c=c, constraints=constraints, integrality=integrality)
    
    print(f"Kết quả bài toán QHTT {len(c)} biến, {len(b_l)} ràng buộc: {res.x}, {res.fun}")
    return res.x, res.fun

# Hàm đệ quy sử dụng phương pháp branch and bound
# Thuật toán:
# 1. Giải bài toán quy hoạch tuyến tính, bỏ đi điều kiện số nguyên
# 2. Nếu kết quả không phải là số nguyên:
#    2.1. Chọn một biến x_i không phải số nguyên, chia thành 2 nhánh:
#         2.1.1. x_i >= ceil(x_i)
#         2.1.2. x_i <= floor(x_i)
#    2.2. Đệ quy với các bài toán con
# 3. Nếu kết quả là số nguyên:
#    3.1. Kiểm tra xem có tốt hơn kết quả hiện tại không
#    3.2. Nếu có, cập nhật kết quả
def branch_and_bound(c, A, b_l, b_u, N_LOOPS = 1e6):
    # Giải bài toán quy hoạch tuyến tính
    x, f = solve_LP(c, A, b_l, b_u)

    # Kết thúc nếu không tìm thấy nghiệm hoặc đã đạt số lần lặp tối đa
    if x is None or N_LOOPS <= 0:
        return None, None

    # Kiểm tra xem kết quả có phải số nguyên không
    if np.all(np.abs(x - np.round(x)) < 1e-6):
        return x, f

    # Chọn một biến không phải số nguyên
    i = np.argmax(np.abs(x - np.round(x)))
    print(f"Chia nhánh với x_{i + 1} = {x[i]}")

    # Nhánh 1: Thêm x_i >= ceil(x_i)
    A1 = np.copy(A)
    A1 = np.append(A1, np.zeros((1, A1.shape[1])), axis=0)
    A1[-1][i] = 1
    b_l1 = np.copy(b_l)
    b_l1 = np.append(b_l1, np.ceil(x[i]))
    b_u1 = np.copy(b_u)
    b_u1 = np.append(b_u1, np.inf)
    print(f"Thêm ĐK: x_{i + 1} >= {np.ceil(x[i])}")
    x1, f1 = branch_and_bound(c, A1, b_l1, b_u1, N_LOOPS - 1)

    # Nhánh 2: Thêm x_i <= floor(x_i)
    A2 = np.copy(A)
    A2 = np.append(A2, np.zeros((1, A2.shape[1])), axis=0)
    A2[-1][i] = 1
    b_u2 = np.copy(b_u)
    b_u2 = np.append(b_u2, np.floor(x[i]))
    b_l2 = np.copy(b_l)
    b_l2 = np.append(b_l2, -np.inf)
    print(f"Thêm ĐK: x_{i + 1} <= {np.floor(x[i])}")
    x2, f2 = branch_and_bound(c, A2, b_l2, b_u2, N_LOOPS - 1)

    # Chọn kết quả tốt nhất
    if x1 is None and x2 is None:
        return None, None
    elif x1 is None or f1 > f2:
        return x2, f2
    elif x2 is None or f1 < f2:
        return x1, f1
    else:
        return x1, f1
      
# Hàm test
def test():
    # Nhập hệ số hàm mục tiêu, đang tìm min
    c = np.array([230, 250, 280, 250, 300])

    # Nhập hệ số của các ràng buộc
    A = np.array([[1, 0, 0, 1, 0],
                  [1, 1, 0, 0, 1],
                  [0, 1, 1, 1, 0],
                  [0, 0, 1, 0, 1]])

    # Chặn dưới của các ràng buộc
    b_l = np.array([9, 20, 11, 17])
    # Chặn trên của các ràng buộc (không có, vì tất cả đều là >=)
    b_u = np.inf * np.ones_like(b_l)

    # Kết quả bài toán:
    x, f = branch_and_bound(c, A, b_l, b_u)
    print(x)
    print(f)

# Chạy thử hàm test
test()


