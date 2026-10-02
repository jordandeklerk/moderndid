# Install R packages needed for validation tests that conda-forge lacks or
# carries in an older release.  Supported on Linux and macOS (see pixi.toml
# platforms).  Packages install into R's site library, ahead of conda's library
# on R's search path.
#
# This script works around two build issues:
#
#   1. R polars (from r-universe) — Rust/jemalloc fails when R's MAKEFLAGS
#      propagate into cargo's subprocess make.  We temporarily override
#      MAKEFLAGS via a cargo config entry.
#
#   2. Rglpk — On macOS the configure script uses dyn.load("conftest.so")
#      but R produces .dylib, so the probe always fails.  On Linux the
#      probe works but can still fail if GLPK headers are in a non-default
#      path.  We patch the configure to use system GLPK directly (safe on
#      both platforms since conda provides the glpk dependency).
#
# Usage:  pixi run -e validation setup-r

set -euo pipefail

PREFIX="${CONDA_PREFIX:?CONDA_PREFIX must be set}"

# Packages installed here take precedence over conda's copies without overwriting them.
mkdir -p "$(R RHOME)/site-library"

is_installed() {
  R --vanilla --quiet -e "quit(status = if ('$1' %in% installed.packages()[,'Package']) 0L else 1L)" \
    2>/dev/null
}

if ! is_installed polars; then
  echo ">>> Installing R polars (requires Rust toolchain) ..."

  # Temporarily tell cargo to clear MAKEFLAGS so jemalloc builds correctly
  CARGO_CFG="${HOME}/.cargo/config.toml"
  CARGO_BAK=""
  NEED_CLEANUP=false

  if [ -f "$CARGO_CFG" ]; then
    if ! grep -q 'MAKEFLAGS.*force.*true' "$CARGO_CFG" 2>/dev/null; then
      CARGO_BAK=$(mktemp)
      cp "$CARGO_CFG" "$CARGO_BAK"
      printf '\n[env]\nMAKEFLAGS = { value = "", force = true }\n' >> "$CARGO_CFG"
      NEED_CLEANUP=true
    fi
  else
    mkdir -p "$(dirname "$CARGO_CFG")"
    printf '[env]\nMAKEFLAGS = { value = "", force = true }\n' > "$CARGO_CFG"
    NEED_CLEANUP=true
  fi

  R --vanilla --quiet -e 'install.packages("polars", repos="https://rpolars.r-universe.dev")' 2>&1

  if [ "$NEED_CLEANUP" = true ]; then
    if [ -n "$CARGO_BAK" ]; then
      mv "$CARGO_BAK" "$CARGO_CFG"
    else
      rm -f "$CARGO_CFG"
    fi
  fi

  if is_installed polars; then
    echo ">>> polars installed successfully"
  else
    echo ">>> WARNING: polars failed to install (didinter tests will be skipped)"
  fi
else
  echo ">>> polars already installed"
fi

if ! is_installed Rglpk; then
  echo ">>> Installing Rglpk (patching configure to use conda GLPK) ..."

  tmpdir=$(mktemp -d)
  R --vanilla --quiet -e "download.packages('Rglpk', destdir='$tmpdir', repos='https://cloud.r-project.org')" 2>/dev/null
  tar xzf "$tmpdir"/Rglpk_*.tar.gz -C "$tmpdir"

  # Replace configure: skip broken dyn.load test, use system GLPK from conda
  cat > "$tmpdir/Rglpk/configure" << 'ENDCFG'
#!/bin/sh
: ${R_HOME=`R RHOME`}
sed -e "s|@GLPK_INCLUDE_PATH@||" \
    -e "s|@GLPK_LIB_PATH@||" \
    -e "s|@GLPK_LIBS@|-lglpk|" \
    -e "s|@GLPK_TS@||" \
    src/Makevars.in > src/Makevars
ENDCFG
  chmod +x "$tmpdir/Rglpk/configure"

  R CMD INSTALL "$tmpdir/Rglpk" 2>&1
  rm -rf "$tmpdir"

  if is_installed Rglpk; then
    echo ">>> Rglpk installed successfully"
  else
    echo ">>> WARNING: Rglpk failed to install (didhonest tests will be skipped)"
  fi
else
  echo ">>> Rglpk already installed"
fi

# conda-forge lags CRAN for these packages. On Apple Silicon it stops at did 2.1.2.
CRAN_PKGS="BMisc DRDID did ptetools contdid triplediff HonestDiD DIDmultiplegtDYN npiv etwfe"
echo ">>> Installing or updating CRAN packages ..."
R --vanilla --quiet -e "
  pkgs <- strsplit('$CRAN_PKGS', ' ')[[1]]
  # Since HonestDiD's current release needs CVXR 1.8 and CVXR 1.8 needs R 4.4, older R keeps conda's copy.
  if (getRversion() < '4.4.0') pkgs <- setdiff(pkgs, 'HonestDiD')
  repos <- 'https://cloud.r-project.org/'
  available <- available.packages(repos = repos)
  installed <- installed.packages()
  installed <- installed[!duplicated(installed[, 'Package']), , drop = FALSE]
  current <- vapply(pkgs, function(pkg) {
    if (!pkg %in% rownames(installed)) return(FALSE)
    # The index has no row when CRAN is unreachable or offers no release this R can use.
    # Either way there is nothing newer to install.
    if (!pkg %in% rownames(available)) return(TRUE)
    package_version(installed[pkg, 'Version']) >= package_version(available[pkg, 'Version'])
  }, logical(1))
  if (any(!current)) {
    install.packages(pkgs[!current], lib = file.path(R.home(), 'site-library'), repos = repos,
                     Ncpus = parallel::detectCores())
  }
" 2>&1

echo ""
echo "=== R package status ==="
ALL_PKGS="polars Rglpk $CRAN_PKGS"
for pkg in $ALL_PKGS; do
  if is_installed "$pkg"; then
    echo "  $pkg: $(R --vanilla --no-echo -e "cat(as.character(packageVersion('$pkg')))" 2>/dev/null)"
  else
    echo "  $pkg: MISSING"
  fi
done
