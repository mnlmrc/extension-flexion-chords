#!/usr/bin/env bash
#
# setup_env.sh -- make this project and its pinned dependencies importable.
#
#   ./setup_env.sh
#
# Run this ONCE, after creating the virtualenv and installing requirements.txt.
# It is NOT sourced and NOT re-run per shell. Re-run it only if you move the
# repository, recreate the virtualenv, or add/remove a dependency below.
#
#
# ---------------------------------------------------------------------------
# WHAT PROBLEM THIS SOLVES
# ---------------------------------------------------------------------------
#
# Seven lab packages are not on PyPI in the versions this project needs, so they
# are pinned as git submodules under external/ (see .gitmodules). Neither they
# nor this project are pip-installed -- you clone and run. Python therefore has
# no idea those directories exist, and `import PcmPy` fails.
#
# Something has to put them on sys.path. This script does that by registering
# them with the virtualenv itself.
#
#
# ---------------------------------------------------------------------------
# WHY A .pth FILE AND NOT PYTHONPATH
# ---------------------------------------------------------------------------
#
# The obvious approach is `export PYTHONPATH=...`. It was tried and it does not
# hold up:
#
#   * PYTHONPATH is per-shell. Anything that starts Python without inheriting
#     that shell's environment sees nothing -- most importantly the VS Code
#     "Run Python File" button, which does NOT apply the python.envFile setting.
#     That produced a bare `ModuleNotFoundError: No module named
#     'EFC_learningfMRI'` with no obvious cause.
#
#   * Relative entries (like the old `PYTHONPATH=.`) resolve against the CURRENT
#     WORKING DIRECTORY, not the repo. They work when you launch from the repo
#     root and break the moment you `cd scripts/`.
#
# A .pth file has neither problem. Python's `site` module reads every *.pth file
# in site-packages at interpreter startup, before any user code runs. So the
# paths are present for every way of starting Python against this venv:
# terminal, VS Code run button, debugger, notebook kernels, subprocesses,
# cluster jobs -- from any working directory, with no environment to set up.
#
# This is exactly the mechanism `pip install -e` uses internally, so it is a
# well-trodden path rather than a trick.
#
#
# ---------------------------------------------------------------------------
# WHY THE .pth USES AN `import` LINE
# ---------------------------------------------------------------------------
#
# A .pth file supports two kinds of line:
#
#   1. A plain directory path, which site.py APPENDS to sys.path -- i.e. AFTER
#      site-packages, so an installed package of the same name wins.
#   2. A line starting with `import`, which site.py hands to exec(). This lets
#      us manipulate sys.path directly, and therefore PREPEND.
#
# We need form 2. `SUITPy` depends on `neuroimagingtools`, which ships a
# top-level `nitools` package. With form 1, a fresh `pip install -r
# requirements.txt` would put that release ahead of our pinned
# external/nitools, silently swapping the dependency. Prepending makes the pin
# win. (This shadowing already happened once with rsatoolbox under the old
# hand-written packages.pth, and went unnoticed for a long time.)
#
# The same trick is used by setuptools' own distutils-precedence.pth, which you
# can see sitting next to the file this script generates.
#
# GOTCHA: the generated line must contain NO comprehensions and NO lambdas.
# site.py runs it via exec(), whose nested scopes cannot see names assigned on
# the same line, so `[os.path.join(_e, n) for n in (...)]` dies with
# `NameError: name '_e' is not defined`. Hence the plain list literal below.
#
#
# ---------------------------------------------------------------------------
# WHICH DIRECTORIES GO ON sys.path, AND WHY
# ---------------------------------------------------------------------------
#
# The repos have two different layouts, which is why the list is not uniform:
#
#   repo/Package/__init__.py   -> the REPO ROOT must be on sys.path
#       external/nitools, external/Functional_Fusion,
#       external/PcmPy, external/AnatSearchlight
#
#   repo/*.py  (modules at the top level)  -> the PARENT of the repo must be on
#   sys.path, i.e. external/ itself
#       external/surfAnalysisPy   (its __init__.py is at the repo root, so the
#                                  repo directory IS the package)
#       external/imaging_pipelines (bare .py files, imported as a namespace
#                                  package)
#
# Plus the repo root itself, for `EFC_learningfMRI` and `scripts`.
#
# NOT included: external/rsatoolbox. That fork has a Cython extension
# (cengine.similarity) which must be compiled, so it cannot be used from source
# on sys.path. rsatoolbox is installed from PyPI via requirements.txt instead.
# To use the fork, run `pip install -e external/rsatoolbox` (needs a C
# compiler). Only depreciated/ imports it.
#
# KNOWN WART: putting repo roots on sys.path also exposes their sibling
# directories, so `docs`, `tests`, `scripts` and `notebooks` become importable
# and merge across repos as namespace packages. The repo root is prepended
# first, so our own `scripts` wins and nothing breaks today. It is still
# fragile: if Functional_Fusion ever adds a `scripts/foo.py` that we do not
# have, `import scripts.foo` would silently resolve to theirs.
#
#
# ---------------------------------------------------------------------------
# TO UNDO
# ---------------------------------------------------------------------------
#
#   rm venv/lib/python3.10/site-packages/_efcl_paths.pth
#
# The file lives inside venv/, which is gitignored, so it is purely local state
# -- like the venv itself. Nothing here is committed except this script.
#
# ---------------------------------------------------------------------------

set -euo pipefail

# Locate the repository root via git rather than $PWD, so the script works when
# invoked from anywhere (./setup_env.sh, bash /abs/path/setup_env.sh, ...).
# The paths baked into the .pth are absolute, which is what frees callers from
# having to be in any particular directory afterwards.
ROOT="$(git -C "$(dirname "${BASH_SOURCE[0]:-$0}")" rev-parse --show-toplevel)"

# Fail early with a useful message if the submodules were never checked out --
# otherwise the .pth would point at empty directories and every import would
# fail later with a confusing ModuleNotFoundError.
if [ ! -e "$ROOT/external/PcmPy/PcmPy/__init__.py" ]; then
    echo "setup_env.sh: external/ is empty -- submodules were not checked out." >&2
    echo "  run: git submodule update --init --recursive" >&2
    exit 1
fi

# Choose the interpreter whose site-packages we write into, in priority order:
#   $PYTHON (explicit override)  ->  activated venv  ->  ./venv
PY="${PYTHON:-}"
if [ -z "$PY" ]; then
    if [ -n "${VIRTUAL_ENV:-}" ]; then PY="$VIRTUAL_ENV/bin/python"
    elif [ -x "$ROOT/venv/bin/python" ]; then PY="$ROOT/venv/bin/python"
    else
        echo "setup_env.sh: no virtualenv found. Activate one, or set PYTHON=..." >&2
        exit 1
    fi
fi

# Ask Python itself where its site-packages is, rather than guessing a path like
# venv/lib/python3.10/site-packages -- that keeps this working across Python
# versions and on platforms that use a different layout.
SITE="$("$PY" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
PTH="$SITE/_efcl_paths.pth"

# Write the .pth. Everything must be on ONE line: site.py processes a .pth file
# line by line, so a multi-line statement would be executed as separate,
# individually invalid fragments. See the GOTCHA note above about scoping.
# This heredoc is intentionally UNQUOTED so that $ROOT is expanded now, baking
# the absolute path into the generated file.
cat > "$PTH" <<PTHEOF
import sys;_r=r'$ROOT';_e=_r+'/external';sys.path[0:0]=[_r,_e,_e+'/nitools',_e+'/Functional_Fusion',_e+'/PcmPy',_e+'/AnatSearchlight']
PTHEOF

echo "wrote $PTH"
echo "  repo root: $ROOT"

# Prove it actually works, in a FRESH interpreter that has just read the file we
# wrote. Without this the script could report success while leaving the
# environment broken -- which is how the exec() scoping bug above was caught.
# This heredoc IS quoted, so the Python below is passed through untouched.
"$PY" - <<'VERIFY'
import importlib, sys
mods = ["EFC_learningfMRI", "nitools", "PcmPy", "Functional_Fusion.atlas_map",
        "surfAnalysisPy", "AnatSearchlight.searchlight", "imaging_pipelines.betas"]
bad = []
for m in mods:
    try:
        importlib.import_module(m)
    except Exception as e:
        bad.append(f"{m}: {type(e).__name__}: {e}")
if bad:
    print("  FAILED:")
    for b in bad:
        print("   ", b)
    sys.exit(1)
print(f"  verified {len(mods)} imports OK")
VERIFY
