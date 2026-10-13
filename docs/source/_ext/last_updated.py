"""Show at the foot of each page the date of the last commit that changed it.

The theme prints a page's ``last_updated`` value below its content. Sphinx's own
``html_last_updated_fmt`` would give every page the build date, so this reads each source file's
last commit from git instead. A page git doesn't track, such as a generated API page, shows no date.
"""

import subprocess
from datetime import date
from pathlib import Path


def setup(app):
    """Register the handler that dates each page."""
    # Run before the theme's own handler, which copies last_updated into the page's footer.
    app.connect("html-page-context", _add_last_updated, priority=400)
    metadata = {"parallel_read_safe": True, "parallel_write_safe": True}
    return metadata


def _add_last_updated(app, pagename, templatename, context, doctree):
    """Set a page's last_updated value to the date of the last commit that changed its source."""
    # Pages such as the search page have no source file to date.
    if doctree is None:
        return
    committed = _last_commit(Path(app.env.doc2path(pagename)))
    if committed is not None:
        context["last_updated"] = f"{committed:%B} {committed.day}, {committed.year}"


def _last_commit(source):
    """Return the date of the last commit that changed a file, or None when git has no record of it."""
    try:
        result = subprocess.run(
            ["git", "log", "-1", "--format=%cs", "--", source.name],
            cwd=source.parent,
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    stamp = result.stdout.strip()
    committed = date.fromisoformat(stamp) if stamp else None
    return committed
