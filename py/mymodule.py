class Foo:
  """ foo: This test-module doesn't do a whole lot, actually. It's purpose is to
    sit here and look pretty.
  """
  def __init__(self, msg=None):
    self.msg = []
    if msg is not None:
      self.msg.append(msg)
      
  def __str__(self):
    return "msg: " + str(self.msg)
    
  def get_msg(self):
    return self.msg
    
  def add_msg(self, new_msg):
    self.msg.append(new_msg)
