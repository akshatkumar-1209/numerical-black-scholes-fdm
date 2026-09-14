import numpy as np
import numpy.linalg as la

def thomas_solve(a, b, c, d):
    n  = len(b)
    bb = b.copy().astype(float)
    cc = c.copy().astype(float)
    dd = d.copy().astype(float)
    for i in range(1, n):
        if abs(bb[i-1]) < 1e-15:
            raise ValueError("Zero pivot at row %d" % (i-1))
        factor = a[i] / bb[i-1]
        bb[i] -= factor * cc[i-1]
        dd[i] -= factor * dd[i-1]
    x = np.zeros(n)
    x[n-1] = dd[n-1] / bb[n-1]
    for i in range(n-2, -1, -1):
        x[i] = (dd[i] - cc[i] * x[i+1]) / bb[i]
    return x

if __name__ == "__main__":
    print("=" * 60)
    print("  Thomas Algorithm Verification")
    print("=" * 60)

    print()
    print("Test 1: 4x4 system  (exact solution = [1,1,1,1])")
    n = 4
    a = np.array([0.0, -1.0, -1.0, -1.0])
    b = np.array([2.0,  2.0,  2.0,  2.0])
    c = np.array([-1.0, -1.0, -1.0, 0.0])
    d = np.array([1.0,  0.0,  0.0,  1.0])
    x_t = thomas_solve(a, b, c, d)
    A   = np.diag(b) + np.diag(a[1:], -1) + np.diag(c[:-1], 1)
    x_n = la.solve(A, d)
    err = la.norm(x_t - x_n)
    print("  Thomas  :", np.round(x_t, 8))
    print("  NumPy   :", np.round(x_n, 8))
    print("  ||diff||:", "%.2e" % err, "  PASSED" if err < 1e-10 else "  FAILED")

    print()
    print("Test 2: Random 100x100 diagonally dominant system")
    np.random.seed(42)
    n = 100
    a_r = np.concatenate([[0.0], -np.random.rand(n-1)])
    c_r = np.concatenate([-np.random.rand(n-1), [0.0]])
    b_r = np.abs(a_r) + np.abs(c_r) + 1.0
    d_r = np.random.rand(n)
    x_t2 = thomas_solve(a_r, b_r, c_r, d_r)
    A_r  = np.diag(b_r) + np.diag(a_r[1:], -1) + np.diag(c_r[:-1], 1)
    x_n2 = la.solve(A_r, d_r)
    err2 = la.norm(x_t2 - x_n2)
    print("  ||diff||:", "%.2e" % err2, "  PASSED" if err2 < 1e-8 else "  FAILED")

    print()
    print("Thomas Algorithm ready for use in Black-Scholes FDM.")
