import nbformat as nbf

p = r".\notebooks\facial_recognition_code_only.ipynb"
nb = nbf.read(p, as_version=4)

# kernelspec / language_info propres (GitHub en a souvent besoin)
nb.metadata.setdefault("kernelspec", {
    "display_name": "Python 3",
    "language": "python",
    "name": "python3",
})
nb.metadata.setdefault("language_info", {
    "name": "python",
    "pygments_lexer": "ipython3"
})

# sources en chaîne + sorties/numéros vidés + métadonnées de cellule simples
for c in nb.cells:
    if c.cell_type == "code":
        if isinstance(c.source, list):
            c.source = "".join(c.source)
        c.outputs = []
        c.execution_count = None
        c.metadata = {}

nbf.write(nb, p)
print("Notebook fixed for GitHub:", p)
