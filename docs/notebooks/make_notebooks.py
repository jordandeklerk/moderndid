"""Write the runnable notebooks in docs/notebooks from the example pages and the R comparisons."""

import posixpath
import re
from pathlib import Path

import jupytext
import nbformat

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "docs" / "source"
OUTPUT = ROOT / "docs" / "notebooks"
DOCS = "https://moderndid.readthedocs.io/en/stable/"
# Since the CI run executes notebooks before a release's tag exists, the URL names the main branch.
STATIC_URL = "https://raw.githubusercontent.com/jordandeklerk/moderndid/main/docs/source/_static/"
EXAMPLES = (
    "example_staggered_did",
    "example_triple_did",
    "example_inter_did",
    "example_cont_did",
    "example_cont_did_replication",
    "example_dyn_balancing",
    "example_honest_did",
    "example_etwfe",
    "example_npiv",
)
# Colab's r2u apt repository installs these CRAN packages as prebuilt binaries in seconds.
R_COMPARISONS = {
    "r_staggered": ["r-cran-did"],
    "r_two_period": ["r-cran-drdid"],
    "r_triple": ["r-cran-triplediff"],
    "r_continuous": ["r-cran-contdid", "r-cran-splines2"],
    "r_intertemporal": ["r-cran-didmultiplegtdyn"],
    "r_honest": ["r-cran-honestdid"],
    "r_etwfe": ["r-cran-etwfe", "r-cran-fixest"],
    "r_npiv": ["r-cran-npiv"],
}
ROLE = re.compile(r"\{(?P<role>[a-z:]+)\}`(?P<target>[^`]+)`")
LINK = re.compile(r"\]\((?P<target>[^)\s]+)\)")
DIRECTIVE = re.compile(r"(?P<fence>:{3,}|`{3,})\{(?P<name>[\w-]+)\}[ \t]*(?P<title>.*)")
TITLED = re.compile(r"(?P<text>.+?)\s*<(?P<target>[^<>]+)>")
UNDERLINE = re.compile(r"([=\-~^\"'*+#`])\1*")
STATIC_PATH = re.compile(r"([\"'])\.\./_static/")


def main():
    """Write every notebook and remove the notebooks whose source is gone."""
    version_file = (ROOT / "moderndid" / "_version.py").read_text(encoding="utf-8")
    version = re.search(r'__version__ = "([^"]+)"', version_file)[1]
    links = Links()
    notebooks = {}
    install = f'!pip install -q "moderndid[all]=={version}"'
    for name in EXAMPLES:
        page = jupytext.read(SOURCE / "user_guide" / f"{name}.md", fmt="md:myst")
        cells = notebook_cells(page, f"user_guide/{name}", links, install)
        notebooks[OUTPUT / f"{name}.ipynb"] = (cells, molab_header(version))
    for name, packages in R_COMPARISONS.items():
        comparison = nbformat.read(SOURCE / "r_comparisons" / f"{name}.ipynb", as_version=4)
        docname = f"r_comparisons/{name}"
        cells = notebook_cells(comparison, docname, links, colab_setup(version, packages, docname))
        notebooks[OUTPUT / "r_comparisons" / f"{name}.ipynb"] = (cells, None)
    for path, (cells, header) in notebooks.items():
        if write(cells, path, header):
            print(f"Wrote {path.relative_to(ROOT)}")
    for path in sorted(OUTPUT.rglob("*.ipynb")):
        if path not in notebooks:
            path.unlink()
            print(f"Removed {path.relative_to(ROOT)}")


class Links:
    """Resolve the docs' cross-references to links in the stable documentation."""

    def __init__(self):
        self.labels = {}
        self.titles = {}
        self.api = {}
        for path in sorted(SOURCE.rglob("*")):
            parts = path.relative_to(SOURCE).parts
            if path.suffix not in {".md", ".rst", ".ipynb"} or any(p.startswith(("_", ".")) for p in parts):
                continue
            if "generated" in parts or "prerun" in parts:
                continue
            docname = path.relative_to(SOURCE).with_suffix("").as_posix()
            lines = _source_lines(path)
            rst = path.suffix == ".rst"
            self.titles[docname] = _next_heading(lines, 0, rst)
            for index, line in enumerate(lines):
                if match := re.fullmatch(r"\((?P<md>[\w-]+)\)=|\.\. _(?P<rst>[\w-]+):", line.strip()):
                    label = match["md"] or match["rst"]
                    anchor = re.sub(r"[^a-z0-9]+", "-", label.lower()).strip("-")
                    self.labels[label] = (f"{DOCS}{docname}.html#{anchor}", _next_heading(lines, index + 1, rst))
        for path in sorted((SOURCE / "api").glob("*.rst")):
            module, folder = None, None
            for line in path.read_text(encoding="utf-8").splitlines():
                if match := re.match(r"\.\. currentmodule:: (\S+)", line):
                    module = match[1]
                elif line.startswith(".. autosummary::"):
                    folder = ""
                elif folder is not None and (match := re.match(r"\s+:toctree: (\S+)", line)):
                    folder = match[1].strip("/")
                elif folder and (match := re.fullmatch(r"\s+([\w.]+)", line)):
                    self.api[f"{module}.{match[1]}"] = f"api/{folder}/{module}.{match[1]}.html"
                elif line.strip() and not line.startswith(" "):
                    folder = None

    def role(self, role, target, docname):
        """Return the Markdown for one MyST role."""
        if role == "math":
            return f"${target}$"
        target = " ".join(target.split())
        match = TITLED.fullmatch(target)
        text, name = (match["text"], match["target"]) if match else ("", target)
        if role in {"func", "class", "meth", "attr", "mod"}:
            full = name.lstrip("~")
            if full not in self.api:
                raise SystemExit(f"{docname}: no API page for {{{role}}}`{target}`")
            shown = text or (full.rsplit(".", 1)[-1] if name.startswith("~") else full)
            return f"[`{shown}`]({DOCS}{self.api[full]})"
        if role == "ref":
            if name not in self.labels:
                raise SystemExit(f"{docname}: no label for {{ref}}`{target}`")
            url, title = self.labels[name]
            return f"[{text or title}]({url})"
        if role == "doc":
            path = name.lstrip("/") if name.startswith("/") else posixpath.join(posixpath.dirname(docname), name)
            path = posixpath.normpath(path)
            if path not in self.titles:
                raise SystemExit(f"{docname}: no page for {{doc}}`{target}`")
            return f"[{text or self.titles[path]}]({DOCS}{path}.html)"
        raise SystemExit(f"{docname}: no Markdown form for the {{{role}}} role")

    def link(self, target, docname):
        """Return the absolute address of a link written relative to a page."""
        if re.match(r"[a-z]+:", target):
            return target
        path, _, anchor = target.partition("#")
        if path:
            path = posixpath.normpath(posixpath.join(posixpath.dirname(docname), path))
            stem, suffix = posixpath.splitext(path)
            if suffix in {"", ".md", ".rst", ".ipynb"}:
                path = f"{stem}.html"
        else:
            path = f"{docname}.html"
        return f"{DOCS}{path}" + (f"#{anchor}" if anchor else "")


def _source_lines(path):
    """Return a page's lines without the code blocks whose comments could pass for headings."""
    if path.suffix == ".ipynb":
        notebook = nbformat.read(path, as_version=4)
        text = "\n\n".join(cell.source for cell in notebook.cells if cell.cell_type == "markdown")
    else:
        text = path.read_text(encoding="utf-8")
    if path.suffix != ".rst":
        text = re.sub(r"^```.*?^```", "", text, flags=re.S | re.M)
    return text.splitlines()


def _next_heading(lines, start, rst):
    """Return the text of the first heading at or after a line."""
    for index in range(start, len(lines)):
        line = lines[index]
        if not rst and (match := re.match(r"#{1,6}\s+(.+)", line)):
            return match[1].strip()
        below = lines[index + 1] if index + 1 < len(lines) else ""
        if (
            rst
            and line.strip()
            and not line[0].isspace()
            and not UNDERLINE.fullmatch(line.strip())
            and UNDERLINE.fullmatch(below.strip())
            and not below[:1].isspace()
            and len(below.strip()) >= len(line.strip())
        ):
            return line.strip()
    return None


def markdown(text, docname, links):
    """Rewrite one Markdown cell's MyST markup as Markdown that Colab and marimo render."""
    text = ROLE.sub(lambda m: links.role(m["role"], m["target"], docname), text)
    lines, kept, index = text.splitlines(), [], 0
    while index < len(lines):
        line = lines[index]
        if re.fullmatch(r"\([\w-]+\)=", line.strip()):
            index += 1
            continue
        if match := DIRECTIVE.fullmatch(line.strip()):
            if match["name"] != "admonition":
                raise SystemExit(f"{docname}: no Markdown form for the {match['name']} directive")
            index += 1
            while index < len(lines) and re.match(r":[\w-]+:", lines[index].strip()):
                index += 1
            body = []
            while index < len(lines) and lines[index].strip() != match["fence"]:
                body.append(lines[index])
                index += 1
            if index == len(lines):
                raise SystemExit(f"{docname}: no closing fence for the admonition {match['title'].strip()!r}")
            index += 1
            while body and not body[0].strip():
                body.pop(0)
            while body and not body[-1].strip():
                body.pop()
            kept += [f"> **{match['title'].strip()}**", ">"] + [f"> {item}".rstrip() for item in body]
            continue
        kept.append(line)
        index += 1
    result = LINK.sub(lambda m: f"]({links.link(m['target'], docname)})", "\n".join(kept))
    if leftover := re.search(r"\{[a-z:-]+\}`.*|^[ \t>]*(?::{3,}|`{3,})\{.*|^[ \t>]*\([\w-]+\)=.*", result, re.M):
        construct = leftover[0].lstrip(" \t>")
        raise SystemExit(f"{docname}: cannot translate {construct!r}")
    return result.strip()


def setup_source(source):
    """Keep a hidden setup cell's display settings and drop what only the docs build needs."""
    lines = [line for line in source.splitlines() if line.strip() != "from prerun import stored"]
    lines = [line for line in lines if not line.startswith("%")]
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()


def catch_error(source):
    """Wrap a cell that raises on purpose because an exception stops Run all in Colab."""
    body = "\n".join(f"    {line}" if line else "" for line in source.splitlines())
    return f'try:\n{body}\nexcept Exception as error:\n    print(f"{{type(error).__name__}}: {{error}}")'


def code_source(source):
    """Point a path into the docs' static folder at GitHub because that folder exists only inside the docs build."""
    return STATIC_PATH.sub(lambda match: f"{match[1]}{STATIC_URL}", source)


def with_header(cells, docname, install):
    """Add a link back to the page under the title and put the install cell after the opening text."""
    first = cells[0]
    title = re.search(r"^# (.+)$", first.source, re.M)
    back = (
        f"This notebook is generated from the [{title[1]}]({DOCS}{docname}.html) page of the ModernDiD documentation."
    )
    first.source = first.source[: title.end()] + f"\n\n{back}" + first.source[title.end() :]
    return [first, nbformat.v4.new_code_cell(install), *cells[1:]]


def notebook_cells(page, docname, links, install):
    """Turn a page into the cells of a notebook that shows no output until the reader runs it."""
    cells = []
    for cell in page.cells:
        tags = set(cell.metadata.get("tags", []))
        if cell.cell_type == "markdown":
            if text := markdown(cell.source, docname, links):
                cells.append(nbformat.v4.new_markdown_cell(text))
        elif cell.cell_type != "code":
            raise SystemExit(f"{docname}: no notebook form for a {cell.cell_type} cell")
        elif "remove-cell" in tags and "stored(" in cell.source:
            continue
        elif "remove-cell" in tags:
            if source := setup_source(cell.source):
                cells.append(nbformat.v4.new_code_cell(code_source(source)))
        elif "raises-exception" in tags:
            cells.append(nbformat.v4.new_code_cell(catch_error(code_source(cell.source))))
        else:
            cells.append(nbformat.v4.new_code_cell(code_source(cell.source)))
    return with_header(cells, docname, install)


def colab_setup(version, packages, docname):
    """Return the code cell that installs a comparison's packages in Colab."""
    apt = ", ".join(f'"{package}"' for package in packages)
    polars = ""
    if docname.endswith("r_intertemporal"):
        # install.packages only warns on failure. The requireNamespace check turns that into a nonzero exit.
        polars = (
            "\n    # R polars isn't on CRAN. With NOT_CRAN set, its installer downloads a prebuilt Rust library."
            '\n    subprocess.run(["Rscript", "-e", \'Sys.setenv(NOT_CRAN = "true"); '
            'install.packages("polars", repos = "https://rpolars.r-universe.dev"); '
            'if (!requireNamespace("polars", quietly = TRUE)) quit(status = 1)\'], check=True)'
        )
    return (
        "import subprocess\nimport sys\n\n"
        'if "google.colab" in sys.modules:\n'
        "    # Colab ships rpy2 3.5 as a single package. Uninstalling it first keeps pip from deleting\n"
        "    # the files it shares with the split rpy2 3.6 packages.\n"
        '    subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "-q", "rpy2"], check=True)\n'
        "    # rpy2 supports Colab's R 4.6 from release 3.6.7 on.\n"
        '    subprocess.run([sys.executable, "-m", "pip", "install", "-q", '
        f'"moderndid[all]=={version}", "rpy2>=3.6.7"], check=True)\n'
        '    subprocess.run(["apt-get", "update", "-qq"], check=True)\n'
        f'    subprocess.run(["apt-get", "install", "-y", "-qq", {apt}], check=True){polars}\n'
        "else:\n"
        f'    print("Outside Colab, set up R as {DOCS}r_comparisons/index.html#r-comparisons-running describes.")'
    )


def molab_header(version):
    """Return the dependency block molab installs because marimo's converter drops the extra and the pin."""
    return f'# /// script\n# requires-python = ">=3.12"\n# dependencies = ["moderndid[all]=={version}"]\n# ///'


def write(cells, path, header=None):
    """Write a notebook when its content changed and report whether it did."""
    notebook = nbformat.v4.new_notebook(cells=cells)
    notebook.metadata = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
    }
    if header:
        notebook.metadata["marimo"] = {"header": header}
    for index, cell in enumerate(notebook.cells):
        # Fixed ids keep an unchanged page from rewriting its notebook on every run.
        cell.id = f"cell-{index}"
        cell.metadata = {}
    nbformat.validate(notebook)
    text = nbformat.writes(notebook).rstrip("\n") + "\n"
    if path.exists() and path.read_text(encoding="utf-8") == text:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")
    return True


if __name__ == "__main__":
    main()
