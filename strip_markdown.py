import nbformat as nbf
src = r".\notebooks\facial_recognition.ipynb"
dst = r".\notebooks\facial_recognition_code_only.ipynb"

nb = nbf.read(src, as_version=4)
new = nbf.v4.new_notebook(metadata=nb.metadata)
new.cells = []

for cell in nb.cells:
    if cell.cell_type == "code":
        c = nbf.v4.new_code_cell(cell.source)
        c.outputs = []              # pas de sorties
        c.execution_count = None    # pas de numéros d'exécution
        c.metadata = {}             # metadata clean
        new.cells.append(c)

nbf.write(new, dst)
print("OK ->", dst, "with", len(new.cells), "code cells")
