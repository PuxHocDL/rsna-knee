import re, base64, ast, os
T = os.environ["TEMP"]
s = open(os.path.join(T, "r224.py"), encoding="utf-8").read()
b = re.search(r'_FASTREAD_B64 = "([^"]+)"', s).group(1)
src = base64.b64decode(b).decode()
open(os.path.join(T, "fastread.py"), "w", encoding="utf-8").write(src)
print(len(src))
t = ast.parse(src)
print([n.name for n in t.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))])
for l in src.splitlines():
    if l.startswith(("import", "from", "CROP", "IMG", "LAYOUT", "def make_builder", "VARIANTS")):
        print(l)
print("---- r224 head")
print("\n".join(s.split("\n")[:30])[:3000])
