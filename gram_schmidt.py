from copy import deepcopy
import unittest
from math import sqrt
from random import randint
from enum import Enum
from matrixops import dot_product
from functools import cmp_to_key

# local files
from matrixops import *
from constants import EPSILON
from ordset import ordset

class Order(Enum):
  LT = -1
  EQ = 0
  GT = 1

def lex_order(v, w):
  for i in range(codomain(v)):
    for j in range(domain(v)):
      delta = w[i][j] - v[i][j]
      if delta < -EPSILON:
        return Order.GT
      elif delta > EPSILON:
        return Order.LT
  return Order.EQ

def is_orthogonal(v, w):
  return abs(dot_product(v, w)) < EPSILON

def project(v, w):
  if equals_zero(w):
    return w
  coeff = dot_product(v, w) / dot_product(w, w)
  return mul(coeff, w)

def norm(v):
  return sqrt(dot_product(v, v))

def normalize(v):
  v_norm = norm(v)
  if abs(v_norm) < EPSILON:
    raise ValueError(f'Vector is approximately zero, cannot normalize: {v}')
  return mul(1 / v_norm, v)

def subtract_component(v, w):
  return sub(v, project(v, w))

def orthogonalize(vs):
  if len(vs) == 0:
    return frozenset()
  out = None
  first_loop = True
  for v in vs:
    if first_loop:
      out = set([v])
      first_loop = False
      continue
    for w in out:
      v = subtract_component(v, w)
    if not equals_zero(v):
      out.add(v)
  return frozenset(out)

def gram_schmidt(vs):
  return ordset(map(normalize, orthogonalize(vs)))

# applies lex order
def mk_onb(vs):
  orthonormal_set = list(gram_schmidt(vs))
  return ordset(sorted(orthonormal_set, key=cmp_to_key(lambda v, w: lex_order(v, w).value)))

def lies_in_subspace(v, generators):
  for e in generators:
    v = subtract_component(v, e)
  return equals_zero(v)

# work in progress, untested fn
def into_standard_basis(ordered_vectors):
  mat = list()
  for v in ordered_vectors:
    mat.append(v)
  return transpose(mat)

# work in progress, untested fn
def transition_matrix(basis1, basis2):
  b1_into_std = into_standard_basis(basis1)
  b2_into_std = into_standard_basis(basis2)
  std_into_b2 = invert(b2_into_std)
  return compose(std_into_b2, b1_into_std)

# work in progress, untested fn
def row_reduce(mat):
  mat = unfreeze(mat)
  currently_reducing_row = 0
  while currently_reducing_row < codomain(mat):
    leftmost_nonzero_column = None
    upmost_nonzero_row = None
    for j in range(domain(mat)):
      if leftmost_nonzero_column is not None:
        break
      for i in range(currently_reducing_row, codomain(mat)):
        if abs(mat[i][j]) > EPSILON:
          leftmost_nonzero_column = j
          upmost_nonzero_row = i
          break
    if leftmost_nonzero_column is None:
      return mat
    if upmost_nonzero_row > currently_reducing_row + 1:
      temp = mat[currently_reducing_row + 1]
      mat[currently_reducing_row] = mat[upmost_nonzero_row]
      mat[upmost_nonzero_row] = temp
    pivot_coeff = mat[currently_reducing_row][currently_reducing_row]
    mat[currently_reducing_row] = list(map(lambda x: x / pivot_coeff), 5)
  return freeze(mat)


class Tests(unittest.TestCase):
  def mk_fiveish_random_nonzero_vecs_in_R3(self):
    mk_entry = lambda: randint(-5, 5)
    mk_v = lambda: freeze([[mk_entry()], [mk_entry()], [mk_entry()]])
    return set(filter(lambda v: not equals_zero(v), [mk_v(), mk_v(), mk_v(), mk_v(), mk_v()]))

  def test_orthogonalize_orthogonalizes(self):
    for i in range(1_000):
      vs = self.mk_fiveish_random_nonzero_vecs_in_R3()
      orthogonal_set = list(orthogonalize(vs))

      # Check that the set is not greater in size than the space's dimension,
      # and check that it is pairwise orthogonal.
      assert len(orthogonal_set) <= 3
      for i in range(len(orthogonal_set)):
        for j in range(len(orthogonal_set)):
          if i == j:
            continue
          assert is_orthogonal(orthogonal_set[i], orthogonal_set[j])

  def test_gram_schmidt_normalizes(self):
    for i in range(1_000):
      vs = self.mk_fiveish_random_nonzero_vecs_in_R3()
      orthonormal_set = gram_schmidt(vs)
      for v in orthonormal_set:
        assert abs(1 - norm(v)) < EPSILON
  
  def test_mk_onb_has_correct_lex_order(self):
    for i in range(1_000):
      vs = self.mk_fiveish_random_nonzero_vecs_in_R3()
      basis = mk_onb(vs)
      i = 0
      for b1 in basis:
        j = 0
        for b2 in basis:
          if i == j:
            assert lex_order(b1, b2) is Order.EQ
          elif i < j:
            assert lex_order(b1, b2) is Order.LT
          else:
            assert lex_order(b1, b2) is Order.GT
          j += 1
        i += 1
  
  def test_matrix_composition_works_as_expected(self):
    m = freeze([[1, 1], [0, 1]])
    result = freeze([[1, 0], [0, 1]])
    for i in range(100):
      result = compose(m, result)
    assert result == freeze([[1, 100], [0, 1]])

  def test_matrix_composition_has_correct_dimensions(self):
    l = freeze([[1, 1, 1], [1, 1, 1]])
    m = freeze([[1, 1], [1, 1], [1, 1]])
    l_m = compose(l, m)
    assert codomain(l_m) == 2 and domain(l_m) == 2

  def test_subspace_membership(self):
    v1 = freeze([[0.5], [0.5], [0]])
    v2 = freeze([[0], [0.5], [1]])
    onb = gram_schmidt(set([v1, v2]))
    standard_basis_e1 = freeze([[1], [0], [0]])
    test_vec = freeze([[500], [488], [-24]])
    assert not lies_in_subspace(standard_basis_e1, onb)
    assert lies_in_subspace(test_vec, onb)

if __name__ == '__main__':
  unittest.main()
