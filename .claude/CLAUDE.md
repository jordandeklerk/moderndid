# moderndid

moderndid is a Python library for difference-in-differences and related causal inference estimators. The package is in `moderndid/`, tests are in `tests/`, and the Sphinx docs are in `docs/source/`.

## Commands

- `pixi run lint` runs the pre-commit hooks. Run it after every code change and resolve every complaint before calling the work done.
- When running tests, run only the files that cover a change, as in `pixi run -e dev pytest tests/did`, never the whole suite.
- `pixi run docs` is the maintainer's own build into `docs/_build` and `docs/_doctree`, so never run it or touch those folders. Build into a scratch directory instead, one build at a time, as the writing-docs-pages skill describes.
- Work on a branch, since pre-commit refuses commits to `main`. The user commits by hand, so never commit or push unless asked.

## Code

- Docstrings follow numpydoc with no `Raises` section, and a function documented that way takes no type hints in its signature, since the docstring owns the types. A one-line summary has no commas, a Parameters entry says only what the argument is and requires, and a Returns section stays short.
- A docstring body runs in short paragraphs and keeps heavy math in Notes, in `.. math::` blocks, without pointing to its own sections. Each piece of math lives in one docstring, and others link to it with `:func:`.
- A Returns section lists each field of a result on its own line as `- **name**: description`.
- When a guide walks through a public function, its docstring has no Examples section. The main body ends instead with a sentence such as "See the :ref:`staggered DiD example <example_staggered_did>` for a full analysis of the minimum wage data," since the guide carries the worked example and docstring code runs on every docs build.
- Every NamedTuple result type lives in its package's `container.py`.
- Imports go at the top of a module, except to break a circular import. Write one-off numbers as literals where they are used, never as module-level constants.
- Never suppress warnings in library code. Fix the cause, such as a solver's iteration limit, or let the warning show.
- An estimator's paper citations live in the References section of its user-facing function and are cited inline as `[1]_`. Helpers, containers, and formatters name no authors.
- Tests are flat functions with fixtures in `conftest.py`, imports at the top, and no comments.
- When two public names read equally well at the call site, ask the user instead of picking one.

## Prose

Prose means docs pages, the README, docstrings, and code comments.

- Docs pages read as a conversation between us and the reader, with "we" for what the guide does and decides and "you" for what the reader sees and can change. "We" appears about once a section, for a decision the guide makes or the path it takes. The other sentences take the data, the estimator, or "you" as their subject. Pages say each thing once and briefly. Callout boxes break up any dense stretches that remain.
- Write the way the introduction of the staggered DiD example reads. Name the question a page answers, the problem its method avoids, and what the reader will see, in full and natural sentences. Since brevity comes from cutting repetition and recaps rather than from compressing sentences, a full phrase such as "minimum wage" beats a clipped one such as "minimum".
- IMPORTANT: apply the [avoid-ai-writing](https://github.com/conorbronsdon/avoid-ai-writing) skill to every piece of prose you write or edit, and run its detector when you finish. Treat its findings as signals, since "features" as a noun and low vocabulary diversity on long technical pages are known false positives.
- Write brief narrative paragraphs, with no colons that introduce a clause, no em dashes, and nothing tacked onto a sentence's end with ", which", ", with", ", then", or ", <verb>ing". Keep the commas a sentence needs, since rewording them away for their own sake makes the prose worse. Docstrings and comments use no bulleted or numbered lists except the Returns bullets above. Docs pages and the README may use a list where the content really is one, such as steps or parallel options, and callout boxes (`important`, `warning`, `tip`, `note`) for what a reader must not miss or could get wrong, without overdoing either.
- Don't join two independent clauses with a comma and a conjunction, as in ", and", ", so", ", but", or ", while". Give each idea its own sentence, or fold one into the other with a leading clause such as "Since..." or "If...", and vary those openers. A plain list such as "A, B, and C" is fine.
- Skip tech-marketing idioms such as "ships with", "out of the box", "under the hood", or "lands", and say plainly what the code does, as in "load_mpdta loads the data".
- Write "percent" rather than "%", "such as" or "for example" rather than Latin abbreviations, and no bold or italic emphasis in running prose.
- Use the reader's words and the names the API already gives an idea, as in "the data" rather than "the frame", and never cycle synonyms to vary the vocabulary.
- Never mention R, an R package, or another library's behavior in docs or shipped code. Describe the method itself and cite its paper, as in Callaway and Sant'Anna (2021).
- Comments explain why in one or two lines.
