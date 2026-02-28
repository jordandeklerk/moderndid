#!/usr/bin/env bash
# Install R packages needed for validation tests that are not available on
# conda-forge.  Supported on Linux and macOS (see pixi.toml platforms).
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

CRAN_PKGS="contdid triplediff HonestDiD DIDmultiplegtDYN"
MISSING=""
for pkg in $CRAN_PKGS; do
  if ! is_installed "$pkg"; then
    MISSING="$MISSING \"$pkg\""
  fi
done

if [ -n "$MISSING" ]; then
  echo ">>> Installing CRAN packages:$MISSING ..."
  R --vanilla --quiet -e "
    pkgs <- c(${MISSING})
    install.packages(pkgs, repos='https://cloud.r-project.org/', Ncpus=parallel::detectCores())
  " 2>&1
fi

echo ""
echo "=== R package status ==="
ALL_PKGS="polars contdid triplediff HonestDiD DIDmultiplegtDYN Rglpk"
for pkg in $ALL_PKGS; do
  if is_installed "$pkg"; then
    echo "  $pkg: OK"
  else
    echo "  $pkg: MISSING"
  fi
done
