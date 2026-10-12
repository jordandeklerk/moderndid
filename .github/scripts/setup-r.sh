# Install R packages needed for validation tests that conda-forge lacks or
# carries in an older release. Supported on Linux and macOS (see pixi.toml
# platforms). Packages install into R's site library, ahead of conda's library
# on R's search path.
#
# R polars comes from r-universe. Its Rust jemalloc build fails when R's
# MAKEFLAGS reach cargo's subprocess make. A temporary cargo config entry clears
# MAKEFLAGS for that build.
#
# On macOS, Rglpk's configure script probes dyn.load("conftest.so") and always
# fails because R builds .dylib files. On Linux the probe can still fail when
# the GLPK headers sit in a nondefault path. Since conda provides GLPK on both
# platforms, the script replaces the configure step with one that links conda's
# GLPK directly.
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
  # A failed download or build falls through to the warning below instead of stopping the setup.
  if tar xzf "$tmpdir"/Rglpk_*.tar.gz -C "$tmpdir"; then
    # The replacement configure skips the broken dyn.load probe and links conda's GLPK.
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

    # Since R CMD INSTALL installs no dependencies and conda lacks Rglpk's slam, slam comes from CRAN first.
    if ! is_installed slam; then
      R --vanilla --quiet -e "install.packages('slam', lib = file.path(R.home(), 'site-library'),
                                               repos = 'https://cloud.r-project.org/')" 2>&1
    fi
    R CMD INSTALL "$tmpdir/Rglpk" 2>&1 || true
  fi
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
  repos <- 'https://cloud.r-project.org/'
  lib <- file.path(R.home(), 'site-library')
  available <- available.packages(repos = repos)
  installed <- installed.packages()
  installed <- installed[!duplicated(installed[, 'Package']), , drop = FALSE]
  # Since HonestDiD 0.2.8 needs CVXR 1.8 and CVXR 1.8 needs R 4.4 through Matrix 1.7, older R gets the newest
  # releases that work together from the CRAN archive. CVXR 1.0-15 calls the osqp interface that osqp 1.0 replaced.
  archived <- if (getRversion() < '4.4.0') c(osqp = '0.6.3.3', CVXR = '1.0-15', HonestDiD = '0.2.7') else character()
  pkgs <- setdiff(pkgs, names(archived))
  stale <- names(archived)[vapply(names(archived), function(pkg) {
    !pkg %in% rownames(installed) || package_version(installed[pkg, 'Version']) != archived[[pkg]]
  }, logical(1))]
  if (length(stale) > 0) {
    # If the archive is unreachable, the setup skips this step instead of stopping.
    tryCatch({
      archive <- tempfile()
      contrib <- file.path(archive, 'src', 'contrib')
      dir.create(contrib, recursive = TRUE)
      for (pkg in stale) {
        tarball <- paste0(pkg, '_', archived[[pkg]], '.tar.gz')
        download.file(paste0(repos, 'src/contrib/Archive/', pkg, '/', tarball), file.path(contrib, tarball))
      }
      tools::write_PACKAGES(contrib, type = 'source')
      archive <- paste0('file://', archive)
      # With CRAN's newer releases left out of the index, dependencies resolve to the archived ones.
      index <- rbind(available.packages(repos = archive, type = 'source'),
                     available[!rownames(available) %in% names(archived), , drop = FALSE])
      install.packages(stale, lib = lib, repos = c(archive, repos), available = index, type = 'source',
                       Ncpus = parallel::detectCores())
    }, error = function(e) message('Could not install from the CRAN archive: ', conditionMessage(e)))
  }
  current <- vapply(pkgs, function(pkg) {
    if (!pkg %in% rownames(installed)) return(FALSE)
    # The index has no row when CRAN is unreachable or offers no release this R can use.
    # Either way there is nothing newer to install.
    if (!pkg %in% rownames(available)) return(TRUE)
    package_version(installed[pkg, 'Version']) >= package_version(available[pkg, 'Version'])
  }, logical(1))
  if (any(!current)) {
    install.packages(pkgs[!current], lib = lib, repos = repos, Ncpus = parallel::detectCores())
  }
" 2>&1

echo ""
echo "=== R package status ==="
# CVXR and osqp show which solver releases HonestDiD runs on.
ALL_PKGS="polars Rglpk $CRAN_PKGS CVXR osqp"
for pkg in $ALL_PKGS; do
  if is_installed "$pkg"; then
    echo "  $pkg: $(R --vanilla --no-echo -e "cat(as.character(packageVersion('$pkg')))" 2>/dev/null)"
  else
    echo "  $pkg: MISSING"
  fi
done
