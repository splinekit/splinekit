# splinekit/docs/conf.py

project = "splinekit"
extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "matplotlib.sphinxext.mathmpl",
#    "matplotlib.sphinxext.only_directives",
    "matplotlib.sphinxext.plot_directive",
#    "matplotlib.sphinxext.ipython_directive",
    "sphinx.ext.doctest",
#    "ipython_console_highlighting",
#    "inheritance_diagram",
#    "numpydoc',
    "nbsphinx",
    "nbsphinx_link",
    "sphinx_design",
]

# Do not treat .html as a source type:
source_suffix = [".rst", ".md"]

# Do not treat docs/_build as source
exclude_patterns = [
    "_build",          # docs/_build
    "auto_examples",   # optional, if used
    "gen_modules",     # optional, if used
    "**/.ipynb_checkpoints",
    "Thumbs.db",
    ".DS_Store",
]

plot_html_show_source_link = False
plot_html_show_formats = False

# THEME
html_theme = "sphinx_rtd_theme"

# Static files (CSS, images, etc.)
html_static_path = ["_static"]

# Load our CSS overrides
html_css_files = [
    "custom.css",
]

# GitHub integration for sphinx_rtd_theme
html_context = {
    "display_github": True,              # Enable "Edit on GitHub" links
    "github_user": "Philippe-Thevenaz",  # GitHub username/org
    "github_repo": "splinekit",          # Repo name
    "github_version": "main",            # Branch: use "main" or "master"
    "conf_py_path": "/docs/",            # Path from repo root to docs root
}

# Execute notebooks on every build
nbsphinx_execute = "always"  # default is "auto"
nbsphinx_timeout = 120       # seconds per cell, adjust as needed

# We don't use any custom notebook formats (jupytext etc.)
nbsphinx_custom_formats = {}