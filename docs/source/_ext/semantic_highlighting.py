"""Color the names in Python code by their role.

Pygments tags a call, a function's parameter, and a local variable alike as a plain name, so the
theme prints all three in one color. This extension retags them in every Python code block. A
call's name becomes a function or a class, a keyword argument's name becomes an attribute, and a
function's parameters become variables in its signature and wherever its body reads them.
"""

from functools import partial
from itertools import pairwise

from ipython_pygments_lexers import IPyLexer
from pygments.filter import Filter
from pygments.lexers.python import PythonConsoleLexer, PythonLexer
from pygments.token import Comment, Generic, Keyword, Name, Operator, Punctuation, String, Text
from sphinx import highlighting


def setup(app):
    """Register Python lexers that tag each name with its role."""
    app.add_lexer("python", partial(_Python, stripnl=False))
    app.add_lexer("pycon", partial(_PythonConsole, stripnl=False))
    # IPython's highlighting extension stores lexer instances, which Sphinx prefers over any
    # registered lexer, so notebook cells need new instances in their place.
    for language in ("ipython", "ipython3"):
        lexer = IPyLexer()
        lexer.add_filter(_Roles())
        highlighting.lexers[language] = lexer
    metadata = {"parallel_read_safe": True, "parallel_write_safe": True}
    return metadata


class _Python(PythonLexer):
    """Python lexer that tags each name with its role."""

    def __init__(self, **options):
        super().__init__(**options)
        self.add_filter(_Roles())


class _PythonConsole(PythonConsoleLexer):
    """Python session lexer that tags each name with its role."""

    def __init__(self, **options):
        super().__init__(**options)
        self.add_filter(_Roles())


class _Roles(Filter):
    """Retag plain names as calls, keyword arguments, and parameters."""

    def filter(self, lexer, stream):
        """Yield each token with the type its role calls for."""
        tokens = list(stream)
        roles = _roles(tokens)
        for index, (ttype, value) in enumerate(tokens):
            yield roles.get(index, ttype), value


def _roles(tokens):
    """Map token positions to the token types their roles call for."""
    significant = [index for index, token in enumerate(tokens) if not _skipped(token)]
    following = dict(pairwise(significant))
    preceding = {after: before for before, after in pairwise(significant)}
    enclosing = _enclosing_brackets(tokens)
    roles = {}
    scopes, signatures = _function_scopes(tokens, significant, enclosing, roles)
    for index in significant:
        ttype, value = tokens[index]
        if index in roles or ttype not in (Name, Name.Builtin):
            continue
        after = following.get(index)
        before = preceding.get(index)
        next_value = tokens[after][1] if after is not None else ""
        attribute = before is not None and tokens[before] == (Operator, ".")
        in_scope = any(start < index < end and value in names for start, end, names in scopes)
        opener = enclosing[index]
        in_call = opener is not None and tokens[opener][1] == "(" and opener not in signatures
        # A name before "=" inside a call's parentheses names the callee's parameter.
        if next_value == "=" and in_call:
            roles[index] = Name.Attribute
        elif in_scope and not attribute:
            roles[index] = Name.Variable
        elif next_value == "(" and ttype is Name:
            roles[index] = Name.Class if value[:1].isupper() else Name.Function
    return roles


def _skipped(token):
    """Return whether the roles look past a token, as they do whitespace, comments, and prompts."""
    ttype, value = token
    skipped = ttype in Comment or ttype in Generic.Prompt or (ttype in Text and not value.strip())
    return skipped


def _enclosing_brackets(tokens):
    """Map each token position to the position of the innermost bracket open around it."""
    stack = []
    enclosing = {}
    for index, (ttype, value) in enumerate(tokens):
        if ttype is Punctuation and value in (")", "]", "}") and stack:
            stack.pop()
        enclosing[index] = stack[-1] if stack else None
        if ttype is Punctuation and value in ("(", "[", "{"):
            stack.append(index)
    return enclosing


def _function_scopes(tokens, significant, enclosing, roles):
    """Tag each function's and lambda's parameters.

    Returns the spans where their bodies read those parameters, and the positions of the
    parentheses that open function signatures.
    """
    starts = _statement_starts(tokens)
    scopes = []
    signatures = set()
    for place, index in enumerate(significant):
        ttype, value = tokens[index]
        opens_signature = place + 2 < len(significant) and tokens[significant[place + 2]][1] == "("
        if ttype is Keyword and value == "def" and opens_signature:
            names, closing = _parameters(tokens, significant[place + 3 :], ")", roles)
            end = _body_end(starts, index, len(tokens))
            scopes.append((closing, end, names))
            signatures.add(significant[place + 2])
        elif ttype is Keyword and value == "lambda":
            names, colon = _parameters(tokens, significant[place + 1 :], ":", roles)
            end = _lambda_end(tokens, colon, enclosing[index] is None)
            scopes.append((colon, end, names))
    return scopes, signatures


def _statement_starts(tokens):
    """Return the position and column of the first token of each line that begins a statement."""
    starts = []
    depth = 0
    column = 0
    fresh = True
    continued = False
    for index, (ttype, value) in enumerate(tokens):
        # A line inside brackets, after a backslash, or inside a string continues a statement.
        if fresh and not _skipped((ttype, value)):
            fresh = False
            if depth == 0 and not continued and ttype not in String:
                starts.append((index, column))
        if ttype is Punctuation and value in ("(", "[", "{"):
            depth += 1
        elif ttype is Punctuation and value in (")", "]", "}"):
            depth = max(depth - 1, 0)
        if "\n" in value:
            fresh = True
            continued = value.endswith("\\\n")
            column = len(value) - value.rindex("\n") - 1
        else:
            column += len(value)
    return starts


def _parameters(tokens, candidates, closer, roles):
    """Tag the parameter names before a signature's closer and return them with the closer's position."""
    names = set()
    depth = 0
    expecting = True
    end = len(tokens)
    for index in candidates:
        ttype, value = tokens[index]
        if ttype is Punctuation and value in ("(", "[", "{"):
            depth += 1
        elif ttype is Punctuation and value in (")", "]", "}"):
            if depth == 0:
                end = index
                break
            depth -= 1
        elif depth == 0 and value == closer:
            end = index
            break
        elif depth == 0 and value == ",":
            expecting = True
        elif depth == 0 and value in (":", "="):
            # An annotation or a default follows the name and holds no parameter of its own.
            expecting = False
        elif depth == 0 and expecting and ttype in (Name, Name.Builtin):
            names.add(value)
            roles[index] = Name.Variable
            expecting = False
    return names, end


def _body_end(starts, definition, length):
    """Return the position of the first statement indented no deeper than a definition."""
    indent = next((column for index, column in reversed(starts) if index <= definition), 0)
    end = next((index for index, column in starts if index > definition and column <= indent), length)
    return end


def _lambda_end(tokens, colon, top_level):
    """Return the position where a lambda's body ends."""
    depth = 0
    for index in range(colon + 1, len(tokens)):
        ttype, value = tokens[index]
        if ttype is Punctuation and value in ("(", "[", "{"):
            depth += 1
        elif ttype is Punctuation and value in (")", "]", "}"):
            if depth == 0:
                return index
            depth -= 1
        elif (depth == 0 and ttype is Punctuation and value == ",") or (
            depth == 0 and top_level and ttype in Text and "\n" in value
        ):
            return index
    return len(tokens)
