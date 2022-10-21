# ordset preserves the order
# in elements are inserted.
class ordset:
  def __init__(self, iterator):
    # python dicts preserve insertion order,
    # so internally this is a dict of pairs {(x, None)}
    # but externally it acts like a set {x}.
    self._data = dict()
    for elem in iterator:
      self._data[elem] = None

  def __contains__(self, val):
    try:
      return self[val] is None
    except KeyError:
      return False

  def __iter__(self):
    return iter(self._data.keys())

  def __len__(self):
    return len(self._data)

  def add(self, val):
    self._data[val] = None