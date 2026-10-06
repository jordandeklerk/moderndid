"""Results that the executed docs pages load instead of computing."""

import os
import pickle
from pathlib import Path


def stored(name, compute):
    """Load the stored result called name.

    A missing file is computed with ``compute`` and saved first, so a local build fills in a new
    result and Read the Docs only loads what was committed.

    Parameters
    ----------
    name : str
        Name of the stored file in this folder, without its extension.
    compute : callable
        Function that takes no arguments and returns the result.

    Returns
    -------
    object
        The stored result.
    """
    path = Path(__file__).with_name(f"{name}.pkl")
    if not path.exists():
        # Read the Docs stops a build at fifteen minutes, so it never runs the slow estimators.
        if os.environ.get("READTHEDOCS") == "True":
            raise FileNotFoundError(f"{path.name} is missing. Build the docs locally to save it, then commit it")
        result = compute()
        with path.open("wb") as file:
            pickle.dump(result, file)
    with path.open("rb") as file:
        loaded = pickle.load(file)
    return loaded
