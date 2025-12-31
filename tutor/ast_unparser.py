import ast

original_source = r"""
def func():
    # this is a comment
    print("hello world!")
func()
"""

class MyUnparser(ast._Unparser):
    # override
    def fill(self, text="", *, allow_semicolon=True):
        self.maybe_newline()
        indent = "                               "
        self.write(indent * self._indent + text)

def transform(source):
    root = ast.parse(source)
    return MyUnparser().visit(root)

print(f"Transformed source code:\n{transform(original_source)}")
exec(original_source)
