from copy import deepcopy
from constants import EPSILON

# In this module, an mxn matrix is represented by
# a tuple with m elements, each of which is an
# n-element tuple of numbers.
# This means that an m-dimensional column vector is
# a tuple of the form ((1,), (2,), ..., (m,)).
# An m-dimensional row vector is a tuple 
# of the form ((1, 2, ..., m),).

# Turn a list of lists into a tuple of tuples.
# Tuples are sometimes advantageous because they are hashable, unlike lists.
# Hashable entities can be added to sets.
def freeze(m):
  return tuple(map(lambda row: tuple(row), m))

# Turn a tuple of tuples into a list of lists.
# Lists are sometimes advantageous because they are mutable, and tuples are not.
def unfreeze(m):
  return list(map(lambda row: list(row), m))

# we characterize a space by its dimension, an integer
def domain(m) -> int:
  # number of columns of the matrix
  try:
    return len(m[0])
  except IndexError:
    return 0

# we characterize a space by its dimension, an integer
def codomain(m) -> int:
  # number of rows of the matrix
  return len(m)

def transpose(mat):
  out = list()
  for j in range(domain(mat)):
    row = list()
    for i in range(codomain(mat)):
      row.append(mat[i][j])
    out.append(row)
  return freeze(out)

def compose(l, t):
  out = list()
  for i in range(codomain(l)):
    row = list()
    for j in range(domain(t)):
      l_row = [[l[i][k]] for k in range(domain(l))]
      t_col = [[t[k][j]] for k in range(codomain(t))]
      row.append(dot_product(l_row, t_col))
    out.append(row)
  return freeze(out)

def negate(m):
  l = unfreeze(deepcopy(m))
  for i in range(codomain(l)):
    for j in range(domain(l)):
      l[i][j] = -l[i][j]
  return freeze(l)

def equals_zero(m):
  for i in range(codomain(m)):
    for j in range(domain(m)):
      if abs(m[i][j]) > EPSILON:
        return False
  return True

def add(t, l):
  out = unfreeze(deepcopy(t))
  for i in range(codomain(t)):
    for j in range(domain(t)):
      out[i][j] += l[i][j]
  return freeze(out)

def mul(r, t):
  out = unfreeze(deepcopy(t))
  for i in range(codomain(t)):
    for j in range(domain(t)):
      out[i][j] = r * out[i][j]
  return freeze(out)

def sub(t, l):
  return add(t, mul(-1, l))

def equals(t, l):
  return equals_zero(add(t, negate(l)))


def dot_product(v, w):
  out = 0
  for i in range(len(v)):
    out += v[i][0] * w[i][0]
  return out
