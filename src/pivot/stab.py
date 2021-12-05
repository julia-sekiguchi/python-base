"""Generate a stab pivot."""


from scipy.sparse.csr import csr_matrix


def stab_pivot(a_mat: csr_matrix, zi_mat: csr_matrix):
    """Generate a stab pivot."""

    aux1 = a_mat.dot(zi_mat)
    aux2 = zi_mat.T.dot(aux1)

    return aux2
