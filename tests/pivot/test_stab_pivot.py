import unittest
from scipy.sparse import csr_matrix

from src.pivot.stab import stab_pivot

a_mat = [[1, 0, 5, 12],
         [-5, 0, 9, 0],
         [-6, 7, 0, 19],
         [0, 10, 9, 0]]

zi_mat = [[1, 0],
          [5, 12],
          [-5, 0],
          [9, 0]]

result = [[-1121,   660],
          [-600,     0]]

wrongResult = [[-1121,   0],
               [-600,    1]]

result = csr_matrix(result)
a_mat = csr_matrix(a_mat)
zi_mat = csr_matrix(zi_mat)
wrongResult = csr_matrix(wrongResult)


class TestStabPivot(unittest.TestCase):
    def test_stab_pivot_equality(self):
        result_mat = stab_pivot(a_mat, zi_mat)
        self.assertEqual((result != result_mat).nnz, 0)

    def test_stab_pivot_equality_false(self):
        result_mat = stab_pivot(a_mat, zi_mat)
        self.assertEqual((wrongResult != result_mat).nnz, 2)
