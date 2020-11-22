from sympy import *
from sympy.parsing.sympy_parser import parse_expr

letters = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N"]


"""
Syntax for inputting a matrix is:
s [scalar] (optional scalar)
[entry] ... (n number of entries per row)
.
.
.
/ (end flag) 
"""
def read_matrix(matrix_name = "A"):
    print(f"{matrix_name}:")
    i = ""
    scalar = None
    matrix = []
    while True:
        i = input().strip()
        if i == "/":
            break
        split = i.split(" ")
        if not split:
            print("Invalid syntax")
            return None
        if split[0] == "s":
            if scalar is not None:
                print("Scalar already specified")
                return None
            scalar = parse_expr(split[1])
        else:
            row = list(map(parse_expr, split))
            matrix.append(row)

    return scalar, matrix

"""
Syntax for inputting vectors:
(a, b, c) (d, e, f) ... (separated by a space)
"""
def read_vectors():
    print("Vectors:")
    i = input().strip()
    split = i.split(" ")
    par = list(map(parse_expr, split))
    vectors = list(map(Matrix, par))
    return vectors

def show_matrix_info(show_det=True, show_rref=True, show_ker=True, show_eigen=True):
    sc, ma = read_matrix()
    if ma is None or not ma:
        return

    n = len(ma)
    m = len(ma[0])

    matrix = Matrix(ma)
    if sc is not None:
        matrix *= sc
    print("\nA =")
    pprint(matrix)
    print(f"{n}x{m}")

    if n == m and show_det:
        deter = matrix.det()
        print(f"\ndet(A) = {deter}")
        if deter != 0:
            print("\ninv(A) =")
            pprint(matrix.inv())
        else:
            print("\nA is not invertible!")

    if show_rref:
        print("\nrref(A) = ")
        pprint(matrix.rref(pivots=False))

    if show_ker:
        print("\nker(A) = ")
        pprint(matrix.nullspace())

        print("\nim(A) = ")
        pprint(matrix.columnspace())

    if n == m and show_eigen:
        print("\nCharacteristic polynomial: ")
        lamda = symbols("lamda")
        p = matrix.charpoly(lamda)
        pprint(factor(p.as_expr()))

        vects = matrix.eigenvects()
        e = []
        for v in vects:
            e.append(f"({str(v[0])}) almu={v[1]}")
        print(f"\nEigenvalues ({len(e)}):")
        print(", ".join(e))

        print("\nEigenvectors: ")
        for v in vects:
            print(f"E({str(v[0])}): ")
            pprint(v[2])

        diagonalizable = matrix.is_diagonalizable(reals_only=True)
        print(f'\nA is {"" if diagonalizable else "not"} diagonalizable!')
        if diagonalizable:
            print("\nDiagonalized (P, D): ")
            pprint(matrix.diagonalize())

    print()

def mult_matrices():
    num = int(input("Num matrices: "))
    matrices = []
    for i in range(num):
        letter = letters[i]
        sc, ma = read_matrix(letter)
        matrix = Matrix(ma)
        if sc is not None:
            matrix *= sc
        matrices.append(matrix)

    p = matrices[0]
    for m in matrices[1:]:
        p *= m

    print("\nProduct:")
    pprint(p)
    print()

def gram_schmidt(vecs=None):
    if vecs is None:
        vectors = read_vectors()
    else:
        vectors = vecs
    print("\nOrthogonalization:")

    ovectors = []
    for i in range(len(vectors)):
        calc = f"v_{i+1} = {v2s(vectors[i])}"
        v = vectors[i]
        for j in range(i):
            calc += f" - ({v2s(vectors[i])} o {v2s(ovectors[j])})/({v2s(ovectors[j])} o {v2s(ovectors[j])})*{v2s(ovectors[j])}"
            v -= ((vectors[i].dot(ovectors[j])) / (ovectors[j].dot(ovectors[j]))) * ovectors[j]
        ovectors.append(v)
        print(calc)

    print("\nOrthogonal vectors:")
    pprint(ovectors)

    print("\nNormalize vectors:")
    nor = list(map(lambda vec: vec.normalized(), ovectors))
    pprint(nor)
    print()
    return nor

def qr_fact():
    vectors = read_vectors()
    print("\nOrthonormalization: (normalize u_i to get vectors)")

    ovectors = []
    for i in range(len(vectors)):
        calc = f"u_{i+1} = {v2s(vectors[i])}"
        u = vectors[i]
        for j in range(i):
            calc += f" - ({v2s(vectors[i])} o {v2s(ovectors[j])})*{v2s(ovectors[j])}"
            u -= (vectors[i].dot(ovectors[j])) * ovectors[j]

        print(calc + " = " + v2s(u))
        n = u.normalized()
        ovectors.append(n)

    print("\nQ =")
    pprint(ovectors)

    M_m = len(vectors)
    Q_m = len(ovectors)

    def gen_r(i, j):
        if i == 0 and j == 0:
            norm = vectors[0].norm()
            print(f"r_11 = ||{v2s(vectors[0])}|| = {norm}")
            return norm
        elif i > j:
            return 0
        elif i < j:
            p = ovectors[i].dot(vectors[j])
            print(f"r_{i+1}{j+1} = {v2s(ovectors[i])} o {v2s(vectors[j])} = {p}")
            return p
        else:
            v_perp = vectors[i]
            for m in range(i):
                v_perp -= (vectors[i].dot(ovectors[m])) * ovectors[m]
            norm = v_perp.norm()
            print(f"r_{i+1}{j+1} = ||{v2s(v_perp)}|| = {norm}")
            return norm

    R = Matrix(Q_m, M_m, gen_r)
    print("\nR =")
    pprint(R)

    print()

def least_squares():
    sc, ma = read_matrix()
    sc2, ma2 = read_matrix("b")

    A = Matrix(ma)
    b = Matrix(ma2)

    ATA = A.T * A
    print("\nATA =")
    pprint(ATA)

    ATb = A.T * b
    print("\nATb =")
    pprint(ATb)

    matrix = ATA.row_join(ATb)
    print("\nrref = ")
    pprint(matrix.rref(pivots=False))

    print()


def v2s(v):
    s = "("
    for i in v:
        s += f"{i},"
    s = s[:-1]
    s += ")"
    return s

def main():
    while True:
        print("1. Matrix All Info (det and inverse (11), rref (12), ker and im (13), eigenvalues/vectors (14))")
        print("2. Multiply n matrices")
        print("3. Gram-Schmidt")
        print("4. QR Factorization")
        print("5. Least Squares")
        s = input()
        c = int(s)

        if s.startswith("1"):
            if len(s) == 1:
                show_matrix_info()
            else:
                show_matrix_info(
                    show_det = s[1] == "1",
                    show_rref = s[1] == "2",
                    show_ker = s[1] == "3",
                    show_eigen = s[1] == "4"
                )
        elif c == 2:
            mult_matrices()
        elif c == 3:
            gram_schmidt()
        elif c == 4:
            qr_fact()
        elif c == 5:
            least_squares()


if __name__ == "__main__":
    main()