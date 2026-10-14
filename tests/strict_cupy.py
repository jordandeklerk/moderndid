"""Strict CuPy stand-in that runs moderndid's GPU routes on a machine without a GPU.

The stand-in follows CuPy 14 wherever a GPU route can trip. Its device array keeps its data in a
private NumPy array without being one. Implicit conversion to NumPy raises, so ``np.asarray``, SciPy,
statsmodels and polars fail on a device array as they do on a CuPy array. Arithmetic, ufuncs and NumPy
functions refuse NumPy array operands, reductions return 0-d device arrays, and device computations
raise no floating-point warnings. ``linalg.solve``, ``linalg.inv`` and ``linalg.cholesky`` return
non-finite or meaningless values for a singular matrix without raising, as CuPy's default
``linalg='ignore'`` mode does, and the random Generator offers only the methods CuPy's has.

Each behavior follows CuPy's source: ``cupy/_core/core.pyx`` for the array, ``_kernel.pyx`` and
``_scalar.pyx`` for ufunc arguments, ``_routines_indexing.pyx`` for indexing, ``cupy/linalg`` for
linear algebra and ``cupy/random/_generator_api.pyx`` for the Generator. The draws come from NumPy's
Philox generator, so they differ from both NumPy's default stream and CuPy's XORWOW stream.
"""

import functools
import importlib.machinery
import threading
import types
import warnings

import numpy as np
import scipy.interpolate
import scipy.linalg
import scipy.sparse
import scipy.special

_ERRSTATE = np.errstate
_LINALG_CONFIG = threading.local()

_MISSING_NAMES = frozenset(
    {
        "asmatrix",
        "block",
        "bmat",
        "busday_count",
        "busday_offset",
        "busdaycalendar",
        "bytes_",
        "character",
        "clongdouble",
        "complex256",
        "datetime64",
        "datetime_as_string",
        "datetime_data",
        "einsum_path",
        "errstate",
        "flexible",
        "float128",
        "frompyfunc",
        "fromregex",
        "geomspace",
        "get_include",
        "getbufsize",
        "geterr",
        "geterrcall",
        "histogram_bin_edges",
        "info",
        "insert",
        "is_busday",
        "isnat",
        "longdouble",
        "matrix",
        "matvec",
        "memmap",
        "nanpercentile",
        "nanquantile",
        "ndenumerate",
        "nditer",
        "nested_iters",
        "object_",
        "polyder",
        "polydiv",
        "polyint",
        "recarray",
        "record",
        "setbufsize",
        "seterr",
        "seterrcall",
        "show_runtime",
        "spacing",
        "str_",
        "timedelta64",
        "unstack",
        "vecdot",
        "vecmat",
        "void",
    }
)

_SPECIAL_UFUNC_NAMES = frozenset(
    [
        "bdtr",
        "bdtrc",
        "bdtri",
        "beta",
        "betainc",
        "betaincinv",
        "betaln",
        "binom",
        "boxcox",
        "boxcox1p",
        "cbrt",
        "chdtr",
        "chdtrc",
        "chdtri",
        "cosdg",
        "cosm1",
        "cotdg",
        "digamma",
        "ellipeinc",
        "ellipj",
        "ellipk",
        "ellipkinc",
        "ellipkm1",
        "entr",
        "erf",
        "erfc",
        "erfcinv",
        "erfcx",
        "erfinv",
        "exp1",
        "exp10",
        "exp2",
        "expi",
        "expit",
        "expm1",
        "expn",
        "exprel",
        "fdtr",
        "fdtrc",
        "fdtri",
        "gamma",
        "gammainc",
        "gammaincc",
        "gammainccinv",
        "gammaincinv",
        "gammaln",
        "gammasgn",
        "gdtr",
        "gdtrc",
        "huber",
        "i0",
        "i0e",
        "i1",
        "i1e",
        "inv_boxcox",
        "inv_boxcox1p",
        "j0",
        "j1",
        "k0",
        "k0e",
        "k1",
        "k1e",
        "kl_div",
        "log1p",
        "log_expit",
        "log_ndtr",
        "loggamma",
        "logit",
        "lpmv",
        "nbdtr",
        "nbdtrc",
        "nbdtri",
        "ndtr",
        "ndtri",
        "pdtr",
        "pdtrc",
        "pdtri",
        "poch",
        "pseudo_huber",
        "psi",
        "radian",
        "rel_entr",
        "rgamma",
        "round",
        "shichi",
        "sici",
        "sindg",
        "tandg",
        "wright_bessel",
        "xlog1py",
        "xlogy",
        "y0",
        "y1",
        "yn",
        "zetac",
    ]
)

_COMPARISONS = frozenset({"equal", "not_equal", "less", "less_equal", "greater", "greater_equal"})

_SCATTER_OPS = {
    "add": ("add", "float16 float32 float64 int32 uint32 uint64"),
    "subtract": ("sub", "int32 uint32"),
    "maximum": ("max", "float32 float64 int32 uint32 uint64"),
    "minimum": ("min", "float32 float64 int32 uint32 uint64"),
    "bitwise_and": ("and", "int32 int64 uint32 uint64"),
    "bitwise_or": ("or", "int32 int64 uint32 uint64"),
    "bitwise_xor": ("xor", "int32 int64 uint32 uint64"),
}

_IMPLICIT_CONVERSION = (
    "Implicit conversion to a NumPy array is not allowed. Please use `.get()` to construct a NumPy array explicitly."
)


def _quiet():
    """Silence floating-point warnings since device kernels raise none."""
    return _ERRSTATE(all="ignore")


def _is_scalar(value):
    """Tell whether a kernel accepts the value as a scalar, as CuPy's ``CScalar`` does."""
    return type(value) in (int, float, bool, complex) or isinstance(value, np.generic)


def _unsupported(value):
    """Build the error CuPy raises for a kernel argument that is neither a device array nor a scalar."""
    return TypeError(f"Unsupported type {type(value)}")


def _holds_host_array(value):
    """Tell whether a value or a list, tuple or dict inside it holds a NumPy array with dimensions."""
    if isinstance(value, np.ndarray):
        return value.ndim > 0
    if isinstance(value, (list, tuple)):
        return any(_holds_host_array(item) for item in value)
    if isinstance(value, dict):
        return any(_holds_host_array(item) for item in value.values())
    return False


def _refuse_host_arrays(args, kwargs):
    """Raise the TypeError CuPy raises when a NumPy array reaches a device function."""
    if _holds_host_array(args) or _holds_host_array(kwargs):
        raise TypeError(f"Unsupported type {np.ndarray}")


def _holds_device_array(value):
    """Tell whether a value or a list or tuple inside it holds a device array."""
    if isinstance(value, ndarray):
        return True
    if isinstance(value, (list, tuple)):
        return any(_holds_device_array(item) for item in value)
    return False


def _only_device_leaves(value):
    """Tell whether every leaf of a nested list or tuple is a device array."""
    if isinstance(value, (list, tuple)):
        return len(value) > 0 and all(_only_device_leaves(item) for item in value)
    return isinstance(value, ndarray)


def _unwrap(value):
    """Replace each device array in a value with its data, inside lists, tuples and dicts too."""
    if isinstance(value, ndarray):
        return value._data
    if isinstance(value, list):
        return [_unwrap(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_unwrap(item) for item in value)
    if isinstance(value, dict):
        return {key: _unwrap(item) for key, item in value.items()}
    return value


def _wrap(value):
    """Turn NumPy arrays and scalars in a result into device arrays, as CuPy returns them."""
    if isinstance(value, np.ndarray):
        return ndarray._from_data(value)
    if isinstance(value, np.generic):
        return ndarray._from_data(np.asarray(value))
    if isinstance(value, tuple):
        items = [_wrap(item) for item in value]
        return type(value)(*items) if hasattr(value, "_fields") else tuple(items)
    if isinstance(value, list):
        return [_wrap(item) for item in value]
    return value


def _check_dtype(dtype):
    """Refuse the dtypes CuPy has no arrays for, such as object and string dtypes."""
    dtype = np.dtype(dtype)
    if dtype.kind in "OSUVMm":
        raise ValueError(f"Unsupported dtype {dtype}")
    return dtype


def _device_index(index):
    """Convert one index component, since CuPy turns NumPy and device index arrays into device arrays."""
    if isinstance(index, ndarray):
        return index._data
    if isinstance(index, (list, tuple)) and _holds_device_array(index):
        return np.asarray(_unwrap(index))
    return index


def _device_key(key):
    """Convert an index key for the wrapped data."""
    if isinstance(key, tuple):
        return tuple(_device_index(part) for part in key)
    return _device_index(key)


def _is_advanced(key):
    """Tell whether a converted key indexes with an array, which makes CuPy accept host values on assignment."""
    parts = key if isinstance(key, tuple) else (key,)
    return any(isinstance(part, (np.ndarray, list, tuple, bool, np.bool_)) for part in parts)


def _integer_index(key):
    """Tell whether a converted key holds an integer index array."""
    parts = key if isinstance(key, tuple) else (key,)
    return any(isinstance(part, (np.ndarray, list)) and np.asarray(part).dtype.kind in "iu" for part in parts)


def _array_method(name):
    """Build a device array method that runs the NumPy method of the same name on the data."""

    def method(self, *args, **kwargs):
        _refuse_host_arrays(args, kwargs)
        with _quiet():
            result = getattr(self._data, name)(*_unwrap(args), **_unwrap(kwargs))
        return _wrap(result)

    method.__name__ = name
    return method


def _binary_operator(name, reflected=False):
    """Build an arithmetic operator that refuses host arrays as CuPy's kernels do."""

    def operator(self, other):
        if not (isinstance(other, ndarray) or _is_scalar(other)):
            if isinstance(other, (np.ndarray, list, tuple)):
                raise _unsupported(other)
            if getattr(other, "__array_ufunc__", self) is None or getattr(other, "__array_priority__", 0) > 100:
                return NotImplemented
            raise _unsupported(other)
        operands = (other, self) if reflected else (self, other)
        return _UFUNCS[name](*operands)

    return operator


def _inplace_operator(name):
    """Build an in-place operator that writes into the device array."""

    def operator(self, other):
        _UFUNCS[name](self, other, out=self)
        return self

    return operator


def _comparison_operator(name):
    """Build a comparison that turns a 0-d host array into a scalar, as CuPy's ``__richcmp__`` does."""

    def operator(self, other):
        if isinstance(other, np.ndarray) and other.ndim == 0:
            other = other.item()
        if name in ("equal", "not_equal") and (other is None or type(other).__eq__ is object.__eq__):
            return ndarray._from_data(np.full(self._data.shape, name == "not_equal"))
        if not (isinstance(other, ndarray) or _is_scalar(other)):
            raise _unsupported(other)
        return _UFUNCS[name](self, other)

    return operator


class ndarray:
    """Device array whose data lives in a private NumPy array."""

    __array_priority__ = 100
    __hash__ = None

    def __init__(self, shape, dtype=float, memptr=None, strides=None, order="C"):
        self._data = np.empty(shape, dtype=_check_dtype(dtype), order=order)

    @classmethod
    def _from_data(cls, data):
        array = object.__new__(cls)
        array._data = data
        return array

    @property
    def shape(self):
        return self._data.shape

    @shape.setter
    def shape(self, value):
        self._data.shape = value

    @property
    def dtype(self):
        return self._data.dtype

    @property
    def ndim(self):
        return self._data.ndim

    @property
    def size(self):
        return self._data.size

    @property
    def itemsize(self):
        return self._data.itemsize

    @property
    def nbytes(self):
        return self._data.nbytes

    @property
    def strides(self):
        return self._data.strides

    @property
    def flags(self):
        return self._data.flags

    @property
    def T(self):
        return ndarray._from_data(self._data.T)

    @property
    def mT(self):
        return ndarray._from_data(self._data.mT)

    @property
    def real(self):
        return ndarray._from_data(self._data.real)

    @property
    def imag(self):
        return ndarray._from_data(self._data.imag)

    @property
    def device(self):
        return _Device(0)

    def __array__(self, dtype=None, copy=None):
        raise TypeError(_IMPLICIT_CONVERSION)

    def get(self, stream=None, order="C", out=None, blocking=True):
        if out is not None:
            out[...] = self._data
            return out
        return np.array(self._data, order=order, copy=True)

    def set(self, arr, stream=None):
        if not isinstance(arr, np.ndarray):
            raise TypeError("Only numpy.ndarray can be set to cupy.ndarray")
        if arr.shape != self._data.shape:
            raise ValueError(f"Shape mismatch. Old shape: {self._data.shape}, new shape: {arr.shape}")
        self._data[...] = arr

    def item(self, *args):
        return self._data.item(*args)

    def tolist(self):
        return self._data.tolist()

    def tobytes(self, order="C"):
        return self._data.tobytes(order=order)

    def astype(self, dtype, order="K", casting=None, subok=None, copy=True):
        data = self._data.astype(_check_dtype(dtype), order=order, casting=casting or "unsafe", copy=copy)
        return self if data is self._data else ndarray._from_data(data)

    def fill(self, value):
        if isinstance(value, ndarray):
            if value.shape != ():
                raise ValueError("non-scalar cupy.ndarray cannot be used for fill")
            value = value._data
        value = np.asarray(value, dtype=self._data.dtype)
        if value.shape != ():
            raise ValueError("non-scalar numpy.ndarray cannot be used for fill")
        self._data.fill(value[()])

    def dot(self, b, out=None):
        if not isinstance(b, ndarray):
            raise TypeError(f"Argument 'b' has incorrect type (expected cupy.ndarray, got {type(b).__name__})")
        with _quiet():
            return _wrap(self._data.dot(b._data, out=_unwrap(out)))

    def take(self, indices, axis=None, out=None):
        return _wrap(self._data.take(_unwrap(indices), axis=axis, out=_unwrap(out)))

    def put(self, indices, values, mode="wrap"):
        np.put(self._data, _unwrap(indices), _unwrap(values), mode=mode)

    def compress(self, condition, axis=None, out=None):
        return _wrap(self._data.compress(_unwrap(condition), axis=axis, out=_unwrap(out)))

    def repeat(self, repeats, axis=None):
        if isinstance(repeats, ndarray):
            raise ValueError("cupy.ndaray cannot be specified as `repeats` argument.")
        return _wrap(self._data.repeat(repeats, axis=axis))

    def ptp(self, axis=None, out=None, keepdims=False):
        return _wrap(np.ptp(self._data, axis=axis, out=_unwrap(out), keepdims=keepdims))

    def scatter_add(self, slices, value):
        warnings.warn("`ndarray.scatter_add` is deprecated. Please use `cupy.add.at` instead.", DeprecationWarning)
        self._scatter_op(slices, value, "add")

    def _scatter_op(self, slices, value, op):
        key = _device_key(slices)
        name = {scatter: ufunc for ufunc, (scatter, _) in _SCATTER_OPS.items()}[op]
        if _integer_index(key) and self._data.dtype.name not in _SCATTER_OPS[name][1].split():
            raise TypeError(f"cupy.{name}.at only supports {', '.join(_SCATTER_OPS[name][1].split())} as data type")
        with _quiet():
            getattr(np, name).at(self._data, key, np.asarray(_unwrap(value), dtype=self._data.dtype))

    all = _array_method("all")
    any = _array_method("any")
    argmax = _array_method("argmax")
    argmin = _array_method("argmin")
    argpartition = _array_method("argpartition")
    argsort = _array_method("argsort")
    clip = _array_method("clip")
    conj = _array_method("conj")
    conjugate = _array_method("conjugate")
    copy = _array_method("copy")
    cumprod = _array_method("cumprod")
    cumsum = _array_method("cumsum")
    diagonal = _array_method("diagonal")
    flatten = _array_method("flatten")
    max = _array_method("max")
    mean = _array_method("mean")
    min = _array_method("min")
    nonzero = _array_method("nonzero")
    partition = _array_method("partition")
    prod = _array_method("prod")
    ravel = _array_method("ravel")
    reshape = _array_method("reshape")
    round = _array_method("round")
    searchsorted = _array_method("searchsorted")
    sort = _array_method("sort")
    squeeze = _array_method("squeeze")
    std = _array_method("std")
    sum = _array_method("sum")
    swapaxes = _array_method("swapaxes")
    trace = _array_method("trace")
    transpose = _array_method("transpose")
    var = _array_method("var")
    view = _array_method("view")

    def __bool__(self):
        if self._data.size == 0:
            warnings.warn(
                "The truth value of an empty array is ambiguous. Returning False, but in future this will result "
                "in an error. Use `array.size > 0` to check that an array is not empty.",
                DeprecationWarning,
            )
            return False
        return bool(self._data)

    def __int__(self):
        return int(self._data)

    def __float__(self):
        return float(self._data)

    def __complex__(self):
        return complex(self._data)

    def __len__(self):
        if self._data.ndim == 0:
            raise TypeError("len() of unsized object")
        return self._data.shape[0]

    def __iter__(self):
        if self._data.ndim == 0:
            raise TypeError("iteration over a 0-d array")
        return (self[i] for i in range(self._data.shape[0]))

    def __repr__(self):
        return repr(self._data)

    def __str__(self):
        return str(self._data)

    def __format__(self, format_spec):
        return format(self._data, format_spec)

    def __copy__(self):
        return self.copy()

    def __deepcopy__(self, memo):
        return self.copy()

    def __reduce__(self):
        return array, (self._data.copy(),)

    def __getitem__(self, key):
        return _wrap(self._data[_device_key(key)])

    def __setitem__(self, key, value):
        key = _device_key(key)
        if isinstance(value, ndarray):
            self._data[key] = value._data
        elif _is_advanced(key):
            self._data[key] = np.asarray(_unwrap(value), dtype=self._data.dtype)
        else:
            value = np.asarray(value, dtype=self._data.dtype)
            if value.shape != ():
                raise ValueError("non-scalar numpy.ndarray cannot be used for fill")
            self._data[key] = value

    __add__ = _binary_operator("add")
    __radd__ = _binary_operator("add", reflected=True)
    __sub__ = _binary_operator("subtract")
    __rsub__ = _binary_operator("subtract", reflected=True)
    __mul__ = _binary_operator("multiply")
    __rmul__ = _binary_operator("multiply", reflected=True)
    __truediv__ = _binary_operator("true_divide")
    __rtruediv__ = _binary_operator("true_divide", reflected=True)
    __floordiv__ = _binary_operator("floor_divide")
    __rfloordiv__ = _binary_operator("floor_divide", reflected=True)
    __mod__ = _binary_operator("remainder")
    __rmod__ = _binary_operator("remainder", reflected=True)
    __divmod__ = _binary_operator("divmod")
    __rdivmod__ = _binary_operator("divmod", reflected=True)
    __pow__ = _binary_operator("power")
    __rpow__ = _binary_operator("power", reflected=True)
    __and__ = _binary_operator("bitwise_and")
    __rand__ = _binary_operator("bitwise_and", reflected=True)
    __or__ = _binary_operator("bitwise_or")
    __ror__ = _binary_operator("bitwise_or", reflected=True)
    __xor__ = _binary_operator("bitwise_xor")
    __rxor__ = _binary_operator("bitwise_xor", reflected=True)
    __lshift__ = _binary_operator("left_shift")
    __rlshift__ = _binary_operator("left_shift", reflected=True)
    __rshift__ = _binary_operator("right_shift")
    __rrshift__ = _binary_operator("right_shift", reflected=True)
    __iadd__ = _inplace_operator("add")
    __isub__ = _inplace_operator("subtract")
    __imul__ = _inplace_operator("multiply")
    __itruediv__ = _inplace_operator("true_divide")
    __ifloordiv__ = _inplace_operator("floor_divide")
    __imod__ = _inplace_operator("remainder")
    __ipow__ = _inplace_operator("power")
    __iand__ = _inplace_operator("bitwise_and")
    __ior__ = _inplace_operator("bitwise_or")
    __ixor__ = _inplace_operator("bitwise_xor")
    __ilshift__ = _inplace_operator("left_shift")
    __irshift__ = _inplace_operator("right_shift")
    __eq__ = _comparison_operator("equal")
    __ne__ = _comparison_operator("not_equal")
    __lt__ = _comparison_operator("less")
    __le__ = _comparison_operator("less_equal")
    __gt__ = _comparison_operator("greater")
    __ge__ = _comparison_operator("greater_equal")

    def __matmul__(self, other):
        if isinstance(other, (np.ndarray, list, tuple)):
            raise _unsupported(other)
        with _quiet():
            return _wrap(np.matmul(self._data, _unwrap(other)))

    def __rmatmul__(self, other):
        if isinstance(other, (np.ndarray, list, tuple)):
            raise _unsupported(other)
        with _quiet():
            return _wrap(np.matmul(other, self._data))

    def __neg__(self):
        return _UFUNCS["negative"](self)

    def __pos__(self):
        return _UFUNCS["positive"](self)

    def __abs__(self):
        return _UFUNCS["absolute"](self)

    def __invert__(self):
        return _UFUNCS["invert"](self)

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        if method not in ("__call__", "outer", "at", "reduce", "accumulate", "reduceat"):
            return NotImplemented
        name = ufunc.__name__
        target = _UFUNCS.get(name)
        if target is None:
            if name not in _SPECIAL_UFUNC_NAMES:
                return NotImplemented
            target = _Ufunc(ufunc)
        inout = inputs
        if "out" in kwargs:
            out = kwargs["out"]
            if len(out) != 1:
                raise ValueError("The 'out' parameter must have exactly one array value")
            inout = (*inputs, out[0])
            kwargs["out"] = out[0]
        for value in inout:
            if not (_is_scalar(value) or isinstance(value, (ndarray, np.ndarray))):
                return NotImplemented
        if name in _COMPARISONS:
            inputs = tuple(x.item() if isinstance(x, np.ndarray) and x.ndim == 0 else x for x in inputs)
        if method == "__call__":
            return target(*inputs, **kwargs)
        return getattr(target, method)(*inputs, **kwargs)

    def __array_function__(self, func, types, args, kwargs):
        target = _function_for(func)
        if target is None:
            return NotImplemented
        for kind in types:
            if not issubclass(kind, (ndarray, np.ndarray)):
                return NotImplemented
        return target(*args, **kwargs)


class _Ufunc:
    """Ufunc whose arguments must be device arrays or scalars, as CuPy's ufuncs require."""

    def __init__(self, ufunc):
        self._ufunc = ufunc
        self.__name__ = ufunc.__name__
        self.nin = ufunc.nin
        self.nout = ufunc.nout
        self.nargs = ufunc.nargs

    def __repr__(self):
        return f"<ufunc '{self.__name__}'>"

    def __call__(self, *args, out=None, dtype=None, casting="same_kind", **kwargs):
        if kwargs:
            raise TypeError(f"Wrong arguments {kwargs}")
        inputs, outputs = args[: self.nin], args[self.nin :]
        if out is not None:
            outputs = out if isinstance(out, tuple) else (out,)
        if all(value is None for value in outputs):
            outputs = ()
        for value in inputs:
            if not (isinstance(value, ndarray) or _is_scalar(value)):
                raise _unsupported(value)
        for value in outputs:
            if value is not None and not isinstance(value, ndarray):
                raise _unsupported(value)
        out_data = tuple(_unwrap(value) for value in outputs) if outputs else None
        with _quiet():
            result = self._ufunc(*_unwrap(inputs), out=out_data, dtype=dtype, casting=casting)
        if outputs:
            return outputs[0] if self.nout == 1 else tuple(outputs)
        return _wrap(result)

    def outer(self, a, b, **kwargs):
        a, b = asarray(a), asarray(b)
        return self(a.reshape(a.shape + (1,) * b.ndim), b.reshape((1,) * a.ndim + b.shape), **kwargs)

    def at(self, a, indices, b=None):
        if self.__name__ not in _SCATTER_OPS:
            raise NotImplementedError(f"`{self.__name__}.at` is not supported yet")
        a._scatter_op(indices, b, _SCATTER_OPS[self.__name__][0])

    def reduce(self, array, axis=0, dtype=None, out=None, keepdims=False):
        if self.__name__ == "add":
            return array.sum(axis, dtype, out, keepdims)
        if self.__name__ == "multiply":
            return array.prod(axis, dtype, out, keepdims)
        raise NotImplementedError(f"`{self.__name__}.reduce` is not supported yet")

    def accumulate(self, array, axis=0, dtype=None, out=None):
        if self.__name__ == "add":
            return array.cumsum(axis, dtype, out)
        if self.__name__ == "multiply":
            return array.cumprod(axis, dtype, out)
        raise NotImplementedError(f"`{self.__name__}.accumulate` is not supported yet")

    def reduceat(self, array, indices, axis=0, dtype=None, out=None):
        if self.__name__ != "add":
            raise NotImplementedError(f"`{self.__name__}.reduceat` is not supported yet")
        with _quiet():
            return _wrap(np.add.reduceat(array._data, _unwrap(indices), axis=axis, dtype=dtype, out=_unwrap(out)))


_UFUNCS = {
    name: _Ufunc(value)
    for name, value in vars(np).items()
    if isinstance(value, np.ufunc) and name not in _MISSING_NAMES and not name.startswith("_")
}


def _strict_function(function, scalar_results_as_arrays=False):
    """Wrap a NumPy function so that it refuses host arrays and returns device arrays."""

    @functools.wraps(function)
    def strict(*args, **kwargs):
        _refuse_host_arrays(args, kwargs)
        with _quiet():
            result = function(*_unwrap(args), **_unwrap(kwargs))
        if scalar_results_as_arrays and type(result) in (int, float, bool):
            result = np.asarray(result)
        return _wrap(result)

    return strict


def _order_statistic(function):
    """Wrap a NumPy quantile function so that it takes ``q`` from the host, as CuPy's does."""

    @functools.wraps(function)
    def statistic(a, q, *args, **kwargs):
        _refuse_host_arrays((a, *args), kwargs)
        with _quiet():
            return _wrap(function(_unwrap(a), _unwrap(q), *_unwrap(args), **_unwrap(kwargs)))

    return statistic


def _join(function, message):
    """Wrap a NumPy function that joins arrays so that it takes device arrays only, as CuPy's do."""

    @functools.wraps(function)
    def join(arrays, *args, **kwargs):
        arrays = list(arrays)
        if not all(isinstance(item, ndarray) for item in arrays):
            raise TypeError(message)
        _refuse_host_arrays(args, kwargs)
        return _wrap(function([item._data for item in arrays], *args, **_unwrap(kwargs)))

    return join


def _to_device(obj, dtype, copy, order, ndmin):
    """Copy an object to the device as ``cupy.array`` does."""
    if isinstance(obj, ndarray):
        data = np.array(obj._data, dtype=dtype, copy=copy, order=order or "K", ndmin=ndmin)
        return obj if data is obj._data else ndarray._from_data(data)
    if _holds_device_array(obj):
        if not _only_device_leaves(obj):
            raise TypeError(_IMPLICIT_CONVERSION)
        return ndarray._from_data(np.array(_unwrap(obj), dtype=dtype, ndmin=ndmin))
    data = np.array(obj, dtype=dtype, copy=True, order=order or "K", ndmin=ndmin)
    _check_dtype(data.dtype)
    return ndarray._from_data(data)


def array(obj, dtype=None, copy=True, order="K", subok=False, ndmin=0, *, blocking=False):
    """Copy an array to the device as ``cupy.array`` does."""
    return _to_device(obj, dtype, copy, order, ndmin)


def asarray(a, dtype=None, order=None, *, copy=None, blocking=False):
    """Move an array to the device as ``cupy.asarray`` does."""
    return _to_device(a, dtype, copy, order, 0)


def asnumpy(a, stream=None, order="C", out=None, *, blocking=True):
    """Copy a device array to the host as ``cupy.asnumpy`` does."""
    if isinstance(a, ndarray):
        return a.get(stream=stream, order=order, out=out)
    if out is not None:
        out[...] = a
        return out
    return np.asarray(a, order=order)


def ascontiguousarray(a, dtype=None):
    """Return a C-contiguous device array, refusing host arrays as CuPy's typed argument does."""
    if not isinstance(a, ndarray):
        raise TypeError(
            f"Argument 'a' has incorrect type (expected cupy._core.core._ndarray_base, got {type(a).__name__})"
        )
    return _wrap(np.ascontiguousarray(a._data, dtype=dtype))


def asfortranarray(a, dtype=None):
    """Return a Fortran-contiguous device array, refusing host arrays as CuPy's typed argument does."""
    if not isinstance(a, ndarray):
        raise TypeError(
            f"Argument 'a' has incorrect type (expected cupy._core.core._ndarray_base, got {type(a).__name__})"
        )
    return _wrap(np.asfortranarray(a._data, dtype=dtype))


def get_array_module(*args):
    """Return the stand-in for device arrays and NumPy otherwise."""
    for arg in args:
        if isinstance(arg, (ndarray, csr_matrix)):
            return cupy
    return np


def is_available():
    """Report a device, as ``cupy.is_available`` does on a machine with a GPU."""
    return True


class _MemoryPool:
    """Memory pool with the methods callers use to inspect or release CuPy's pool."""

    def free_all_blocks(self, stream=None):
        return None

    def used_bytes(self):
        return 0

    def total_bytes(self):
        return 0

    def set_limit(self, size=None, fraction=None):
        return None

    def get_limit(self):
        return 0


_MEMORY_POOL = _MemoryPool()


def get_default_memory_pool():
    """Return the stand-in's memory pool."""
    return _MEMORY_POOL


def get_default_pinned_memory_pool():
    """Return the stand-in's pinned memory pool."""
    return _MEMORY_POOL


_EXPLICIT = {
    "ndarray": ndarray,
    "array": array,
    "asarray": asarray,
    "asanyarray": asarray,
    "asnumpy": asnumpy,
    "ascontiguousarray": ascontiguousarray,
    "asfortranarray": asfortranarray,
    "get_array_module": get_array_module,
    "is_available": is_available,
    "get_default_memory_pool": get_default_memory_pool,
    "get_default_pinned_memory_pool": get_default_pinned_memory_pool,
    "concatenate": _join(np.concatenate, "Only cupy arrays can be concatenated"),
    "concat": _join(np.concatenate, "Only cupy arrays can be concatenated"),
    "stack": _join(np.stack, "Only cupy arrays can be concatenated"),
    "hstack": _join(np.hstack, "Only cupy arrays can be concatenated"),
    "vstack": _join(np.vstack, "Only cupy arrays can be concatenated"),
    "row_stack": _join(np.vstack, "Only cupy arrays can be concatenated"),
    "dstack": _join(np.dstack, "Only cupy arrays can be concatenated"),
    "column_stack": _join(np.column_stack, "Only cupy arrays can be column stacked"),
    "quantile": _order_statistic(np.quantile),
    "percentile": _order_statistic(np.percentile),
    "count_nonzero": _strict_function(np.count_nonzero, scalar_results_as_arrays=True),
    "array_equal": _strict_function(np.array_equal, scalar_results_as_arrays=True),
    "array_equiv": _strict_function(np.array_equiv, scalar_results_as_arrays=True),
    "allclose": _strict_function(np.allclose, scalar_results_as_arrays=True),
}


@functools.cache
def _cupy_attribute(name):
    """Resolve a ``cupy`` attribute from NumPy's namespace minus what CuPy lacks."""
    if name in _EXPLICIT:
        return _EXPLICIT[name]
    if name in _MISSING_NAMES or name.startswith("_") or not hasattr(np, name):
        raise AttributeError(f"module 'cupy' has no attribute {name!r}")
    value = getattr(np, name)
    if isinstance(value, np.ufunc):
        return _UFUNCS[name]
    if isinstance(value, types.ModuleType):
        raise AttributeError(f"module 'cupy' has no attribute {name!r}")
    if callable(value) and not isinstance(value, type):
        return _strict_function(value)
    return value


def _cupy_getattr(name):
    return _cupy_attribute(name)


def _function_for(func):
    """Find the stand-in function that a NumPy function dispatches to, as CuPy's ``__array_function__`` does."""
    module = getattr(func, "__module__", None) or ""
    name = getattr(func, "__name__", "")
    if module == "numpy":
        try:
            return _cupy_attribute(name)
        except AttributeError:
            return None
    if module == "numpy.linalg":
        return getattr(_LINALG, name, None)
    return None


def _linalg_mode():
    """Return the thread's linalg error mode, which ``cupyx.errstate`` sets."""
    return getattr(_LINALG_CONFIG, "linalg", "ignore")


def _require_device(*arrays):
    """Raise the LinAlgError CuPy's linalg raises for arrays that are not on the device."""
    for value in arrays:
        if not isinstance(value, ndarray):
            raise np.linalg.LinAlgError("cupy.linalg only supports cupy.ndarray")


def _require_stacked_square(a):
    """Raise CuPy's LinAlgError unless an array holds square matrices."""
    if a.ndim < 2:
        raise np.linalg.LinAlgError(f"{a.ndim}-dimensional array given. Array must be at least two-dimensional")
    if a.shape[-1] != a.shape[-2]:
        raise np.linalg.LinAlgError("Last 2 dimensions of the array must be square")


def _linalg_dtype(*arrays):
    """Return the dtype CuPy's linalg computes in."""
    dtypes = [value.dtype for value in arrays]
    if np.dtype(np.float16) in dtypes:
        raise TypeError("float16 is unsupported in linalg")
    return np.result_type(*[dtype if dtype.kind in "fc" else np.float64 for dtype in dtypes])


def _report(routine, info):
    """Raise CuPy's LinAlgError for a failed cuSOLVER call when ``cupyx.errstate(linalg='raise')`` is active."""
    if info != 0 and _linalg_mode() == "raise":
        raise np.linalg.LinAlgError(
            f"Error reported by {routine} in cuSOLVER. devInfo = {info}. Please refer to the cuSOLVER documentation."
        )


def _gesv(a, b):
    """Solve by LU factorization with partial pivoting without checking the pivots, as cuSOLVER's getrf and getrs do."""
    getrf, getrs = scipy.linalg.get_lapack_funcs(("getrf", "getrs"), (a, b))
    lu, piv, info = getrf(a)
    _report("getrf", info)
    x, _ = getrs(lu, piv, b)
    return x


def _solve(a, b):
    _require_device(a, b)
    _require_stacked_square(a)
    if b.ndim == 0:
        raise ValueError("b must have at least one dimension")
    if b.ndim == 1:
        if a.shape[-1] != b.size:
            raise ValueError("a must have (..., M, M) shape and b must have (M,) for one-dimensional b")
        rhs = np.broadcast_to(b._data, a.shape[:-1])
    elif a.shape[:-1] != b.shape[:-1]:
        raise ValueError("a must have (..., M, M) shape and b must have (..., M, K) for multidimensional b")
    else:
        rhs = b._data
    dtype = _linalg_dtype(a, b)
    lhs, rhs = a._data.astype(dtype), np.array(rhs, dtype=dtype)
    with _quiet():
        if lhs.ndim == 2:
            return _wrap(_gesv(lhs, rhs))
        out = np.empty(rhs.shape, dtype)
        for index in np.ndindex(lhs.shape[:-2]):
            out[index] = _gesv(lhs[index], rhs[index])
    return _wrap(out)


def _inv(a):
    _require_device(a)
    _require_stacked_square(a)
    dtype = _linalg_dtype(a)
    data = a._data.astype(dtype)
    identity = np.eye(a.shape[-1], dtype=dtype)
    with _quiet():
        if data.ndim == 2:
            return _wrap(_gesv(data, identity))
        out = np.empty(data.shape, dtype)
        for index in np.ndindex(data.shape[:-2]):
            out[index] = _gesv(data[index], identity)
    return _wrap(out)


def _cholesky(a):
    _require_device(a)
    _require_stacked_square(a)
    data = a._data.astype(_linalg_dtype(a))
    potrf = scipy.linalg.get_lapack_funcs("potrf", (data,))
    out = np.empty(data.shape, data.dtype)
    with _quiet():
        for index in np.ndindex(data.shape[:-2]):
            factor, info = potrf(data[index], lower=True, clean=True)
            _report("potrf", info)
            out[index] = np.tril(factor)
    return _wrap(out)


def _qr(a, mode="reduced"):
    _require_device(a)
    if mode not in ("reduced", "complete", "r", "raw"):
        raise ValueError(f"Unrecognized mode '{mode}'")
    if a.ndim < 2:
        raise np.linalg.LinAlgError(f"{a.ndim}-dimensional array given. Array must be at least two-dimensional")
    with _quiet():
        result = np.linalg.qr(a._data, mode=mode)
    return _wrap(result if mode == "r" else tuple(result))


def _failed_svd(data, full_matrices, compute_uv):
    m, n = data.shape[-2:]
    k = min(m, n)
    batch = data.shape[:-2]
    s = np.full((*batch, k), np.nan)
    if not compute_uv:
        return s
    u = np.full((*batch, m, m if full_matrices else k), np.nan)
    vh = np.full((*batch, n if full_matrices else k, n), np.nan)
    return u, s, vh


def _svd(a, full_matrices=True, compute_uv=True):
    _require_device(a)
    if a.ndim < 2:
        raise np.linalg.LinAlgError(f"{a.ndim}-dimensional array given. Array must be at least two-dimensional")
    try:
        with _quiet():
            result = np.linalg.svd(a._data, full_matrices=full_matrices, compute_uv=compute_uv)
    except np.linalg.LinAlgError:
        _report("gesvd", 1)
        result = _failed_svd(a._data, full_matrices, compute_uv)
    return _wrap(result if not compute_uv else tuple(result))


def _require_device_matrix(a):
    if not isinstance(a, ndarray):
        raise _unsupported(a)
    if a.ndim < 2:
        raise np.linalg.LinAlgError(f"{a.ndim}-dimensional array given. Array must be at least two-dimensional")


def _eigh(a, UPLO="L"):
    _require_device_matrix(a)
    if UPLO not in ("L", "U"):
        raise ValueError("UPLO argument must be 'L' or 'U'")
    try:
        with _quiet():
            w, v = np.linalg.eigh(a._data, UPLO=UPLO)
    except np.linalg.LinAlgError:
        _report("syevd", 1)
        w, v = np.full(a.shape[:-1], np.nan), np.full(a.shape, np.nan)
    return _wrap(w), _wrap(v)


def _eigvalsh(a, UPLO="L"):
    _require_device_matrix(a)
    if UPLO not in ("L", "U"):
        raise ValueError("UPLO argument must be 'L' or 'U'")
    try:
        with _quiet():
            return _wrap(np.linalg.eigvalsh(a._data, UPLO=UPLO))
    except np.linalg.LinAlgError:
        _report("syevd", 1)
        return _wrap(np.full(a.shape[:-1], np.nan))


def _eig(a):
    _require_device_matrix(a)
    try:
        with _quiet():
            w, v = np.linalg.eig(a._data)
    except np.linalg.LinAlgError:
        _report("geev", 1)
        w, v = np.full(a.shape[:-1], np.nan + 0j), np.full(a.shape, np.nan + 0j)
    return _wrap(w), _wrap(v)


def _eigvals(a):
    return _eig(a)[0]


def _slogdet(a):
    _require_device_matrix(a)
    _require_stacked_square(a)
    with _quiet():
        sign, logdet = np.linalg.slogdet(a._data.astype(_linalg_dtype(a)))
    return _wrap(sign), _wrap(logdet)


def _det(a):
    sign, logdet = _slogdet(a)
    with _quiet():
        return _wrap(sign._data * np.exp(logdet._data))


def _matrix_rank(M, tol=None):
    if M.ndim < 2:
        return (M != 0).any().astype(int)
    s = _svd(M, compute_uv=False)._data
    with _quiet():
        if tol is None:
            tol = s.max(axis=-1, keepdims=True) * max(M.shape[-2:]) * np.finfo(s.dtype).eps
        return _wrap((s > _unwrap(tol)).sum(axis=-1, dtype=np.intp))


def _norm(x, ord=None, axis=None, keepdims=False):
    if not isinstance(x, ndarray):
        raise _unsupported(x)
    with _quiet():
        return _wrap(np.linalg.norm(x._data, ord=ord, axis=axis, keepdims=keepdims))


def _cond(x, p=None):
    x = asarray(x)
    if x.size == 0:
        raise np.linalg.LinAlgError("cond is not defined on empty arrays")
    with _quiet():
        if p is None or p == 2 or p == -2:
            s = _svd(x, compute_uv=False)._data
            r = s[..., -1] / s[..., 0] if p == -2 else s[..., 0] / s[..., -1]
        else:
            _require_stacked_square(x)
            previous = _linalg_mode()
            _LINALG_CONFIG.linalg = "ignore"
            try:
                inverse = _inv(x)._data
            finally:
                _LINALG_CONFIG.linalg = previous
            r = np.linalg.norm(x._data, p, axis=(-2, -1)) * np.linalg.norm(inverse, p, axis=(-2, -1))
        r = np.asarray(r)
        missing = np.isnan(r) & ~np.isnan(x._data).any(axis=(-2, -1))
        return _wrap(np.where(missing, np.inf, r))


def _lstsq(a, b, rcond=None):
    _require_device(a, b)
    if a.ndim != 2:
        raise np.linalg.LinAlgError(f"{a.ndim}-dimensional array given. Array must be two-dimensional")
    if b.ndim > 2:
        raise np.linalg.LinAlgError(f"{b.ndim}-dimensional array given. Array must be at most two-dimensional")
    m, n = a.shape
    if m != b.shape[0]:
        raise np.linalg.LinAlgError("Incompatible dimensions")
    u, s, vh = (value._data for value in _svd(a, full_matrices=False))
    if rcond is None:
        rcond = np.finfo(s.dtype).eps * max(m, n)
    elif rcond <= 0 or rcond >= 1:
        rcond = np.finfo(s.dtype).eps
    with _quiet():
        s1 = 1 / s
        rank = s.size
        if s.size > 0:
            singular = s <= rcond * s.max()
            s1[singular] = 0
            rank -= int(singular.sum())
        x = vh.T.conj() @ (b._data.T @ u.conj() * s1).T
        if m <= n or rank != n:
            residuals = np.empty((0,), dtype=s.dtype)
        else:
            error = b._data - a._data @ x
            residuals = np.atleast_1d(np.sum(np.abs(error.T) ** 2, axis=-1))
    return _wrap(x), _wrap(residuals), _wrap(np.array(rank, np.int32)), _wrap(s)


def _pinv(a, rcond=1e-15):
    _require_device(a)
    u, s, vt = (value._data for value in _svd(a.conj(), full_matrices=False))
    with _quiet():
        cutoff = _unwrap(rcond) * np.amax(s, axis=-1)
        small = s <= cutoff[..., None]
        s = 1 / s
        s[small] = 0
        return _wrap(np.matmul(vt.swapaxes(-2, -1), s[..., None] * u.swapaxes(-2, -1)))


def _tensorsolve(a, b, axes=None):
    if axes is not None:
        order = list(range(a.ndim))
        for k in axes:
            order.remove(k)
            order.insert(a.ndim, k)
        a = a.transpose(order)
    old_shape = a.shape[-(a.ndim - b.ndim) :]
    result = _solve(a.reshape(-1, int(np.prod(old_shape))), b.ravel())
    return result.reshape(old_shape)


def _tensorinv(a, ind=2):
    if ind <= 0:
        raise ValueError("Invalid ind argument")
    old_shape = a.shape
    inv_shape = old_shape[ind:] + old_shape[:ind]
    return _inv(a.reshape(int(np.prod(old_shape[:ind])), -1)).reshape(*inv_shape)


def _module(name, attributes=None, getattr_function=None):
    """Build a module object that imports and attribute lookups find like the real one."""
    module = types.ModuleType(name)
    module.__spec__ = importlib.machinery.ModuleSpec(name, None)
    module.__dict__.update(attributes or {})
    if getattr_function is not None:
        module.__getattr__ = getattr_function
    return module


_LINALG = _module(
    "cupy.linalg",
    {
        "LinAlgError": np.linalg.LinAlgError,
        "solve": _solve,
        "inv": _inv,
        "cholesky": _cholesky,
        "qr": _qr,
        "svd": _svd,
        "eigh": _eigh,
        "eigvalsh": _eigvalsh,
        "eig": _eig,
        "eigvals": _eigvals,
        "det": _det,
        "slogdet": _slogdet,
        "matrix_rank": _matrix_rank,
        "norm": _norm,
        "cond": _cond,
        "lstsq": _lstsq,
        "pinv": _pinv,
        "tensorsolve": _tensorsolve,
        "tensorinv": _tensorinv,
        **{
            name: _strict_function(getattr(np.linalg, name))
            for name in ("matrix_power", "cross", "matmul", "matrix_transpose")
            if hasattr(np.linalg, name)
        },
    },
)


def _distribution_parameter(value, name, dtype, scalar_types=(float, int)):
    """Read a distribution parameter the way CuPy's Generator does, which takes device arrays or Python numbers."""
    if isinstance(value, ndarray):
        return value._data.astype(dtype, copy=False)
    if type(value) in scalar_types:
        return np.asarray(value, dtype=dtype)
    raise TypeError(f"{name} is required to be a cupy.ndarray or a scalar")


class BitGenerator:
    """Bit generator seeded through ``numpy.random.SeedSequence`` like CuPy's."""

    def __init__(self, seed=None):
        if isinstance(seed, ndarray):
            seed = asnumpy(seed)
        self._seed_seq = seed if isinstance(seed, np.random.SeedSequence) else np.random.SeedSequence(seed)
        self._numpy_bit_generator = np.random.Philox(self._seed_seq)


class XORWOW(BitGenerator):
    """CuPy's default bit generator, emulated with NumPy's Philox stream."""


class Generator:
    """Random generator with only the methods of CuPy's Generator."""

    def __init__(self, bit_generator):
        self.bit_generator = bit_generator
        self._rng = np.random.Generator(bit_generator._numpy_bit_generator)

    def _finish(self, values, dtype=None, out=None):
        values = np.asarray(values) if dtype is None else np.asarray(values, dtype=dtype)
        if out is not None:
            out[...] = ndarray._from_data(values)
            return out
        return ndarray._from_data(values)

    def random(self, size=None, dtype=np.float64, out=None):
        return self._finish(self._rng.random(size=size, dtype=dtype), out=out)

    def uniform(self, low=0.0, high=1.0, size=None, dtype=np.float64):
        low, high = asarray(low)._data, asarray(high)._data
        if size is None:
            size = np.broadcast(low, high).shape
        return self._finish(low + (high - low) * self._rng.random(size=size), dtype=dtype)

    def integers(self, low, high=None, size=None, dtype=np.int64, endpoint=False):
        lo, hi = (0, int(low)) if high is None else (int(low), int(high))
        if not endpoint:
            hi -= 1
        if lo > hi:
            raise ValueError("low >= high")
        return self._finish(self._rng.integers(lo, hi, size=size, dtype=dtype, endpoint=True))

    def standard_normal(self, size=None, dtype=np.float64, out=None):
        if np.dtype(dtype).char not in "fd":
            raise TypeError(f"Unsupported dtype {np.dtype(dtype).name} for standard_normal")
        return self._finish(self._rng.standard_normal(size=size, dtype=dtype), out=out)

    def standard_exponential(self, size=None, dtype=np.float64, method="inv", out=None):
        if method == "zig":
            raise NotImplementedError("Ziggurat method is not supported")
        return self._finish(self._rng.standard_exponential(size=size, dtype=dtype, method="inv"), out=out)

    def exponential(self, scale=1.0, size=None):
        return self.standard_exponential(size) * scale

    def standard_gamma(self, shape, size=None, dtype=np.float64, out=None):
        shape = _distribution_parameter(shape, "shape", np.float64)
        return self._finish(self._rng.standard_gamma(shape, size=size), dtype=dtype, out=out)

    def gamma(self, shape, scale=1.0, size=None):
        return self.standard_gamma(shape, size) * scale

    def beta(self, a, b, size=None, dtype=np.float64):
        a = _distribution_parameter(a, "a", np.float64)
        b = _distribution_parameter(b, "b", np.float64)
        return self._finish(self._rng.beta(a, b, size=size), dtype=dtype)

    def binomial(self, n, p, size=None):
        n = _distribution_parameter(n, "n", np.int64)
        p = _distribution_parameter(p, "p", np.float64, scalar_types=(float,))
        return self._finish(self._rng.binomial(n, p, size=size), dtype=np.int64)

    def chisquare(self, df, size=None):
        return self._finish(self._rng.chisquare(_distribution_parameter(df, "df", np.float64), size=size))

    def dirichlet(self, alpha, size=None):
        return self._finish(self._rng.dirichlet(_distribution_parameter(alpha, "alpha", np.float64), size=size))

    def f(self, dfnum, dfden, size=None):
        dfnum = _distribution_parameter(dfnum, "dfnum", np.float64)
        dfden = _distribution_parameter(dfden, "dfden", np.float64)
        return self._finish(self._rng.f(dfnum, dfden, size=size))

    def geometric(self, p, size=None):
        return self._finish(self._rng.geometric(_distribution_parameter(p, "p", np.float64), size=size), np.int64)

    def hypergeometric(self, ngood, nbad, nsample, size=None):
        ngood = _distribution_parameter(ngood, "ngood", np.int64)
        nbad = _distribution_parameter(nbad, "nbad", np.int64)
        nsample = _distribution_parameter(nsample, "nsample", np.int64)
        return self._finish(self._rng.hypergeometric(ngood, nbad, nsample, size=size), np.int64)

    def logseries(self, p, size=None):
        return self._finish(self._rng.logseries(_distribution_parameter(p, "p", np.float64), size=size), np.int64)

    def poisson(self, lam=1.0, size=None):
        return self._finish(self._rng.poisson(_distribution_parameter(lam, "lam", np.float64), size=size), np.int64)

    def power(self, a, size=None):
        return self._finish(self._rng.power(_distribution_parameter(a, "a", np.float64), size=size))


def default_rng(seed=None):
    """Construct a Generator as ``cupy.random.default_rng`` does."""
    if isinstance(seed, BitGenerator):
        return Generator(seed)
    if isinstance(seed, Generator):
        return seed
    return Generator(XORWOW(seed))


class RandomState:
    """Legacy random state with the module-level functions CuPy keeps beside its Generator."""

    def __init__(self, seed=None):
        self.seed(seed)

    def seed(self, seed=None):
        self._rng = np.random.Generator(np.random.Philox(np.random.SeedSequence(seed)))

    def rand(self, *size, dtype=float):
        return _wrap(np.asarray(self._rng.random(size=size), dtype=dtype))

    def randn(self, *size, dtype=float):
        return _wrap(np.asarray(self._rng.standard_normal(size=size), dtype=dtype))

    def random_sample(self, size=None, dtype=float):
        return _wrap(np.asarray(self._rng.random(size=size), dtype=dtype))

    def uniform(self, low=0.0, high=1.0, size=None, dtype=float):
        return _wrap(np.asarray(self._rng.uniform(_unwrap(low), _unwrap(high), size=size), dtype=dtype))

    def normal(self, loc=0.0, scale=1.0, size=None, dtype=float):
        return _wrap(np.asarray(self._rng.normal(_unwrap(loc), _unwrap(scale), size=size), dtype=dtype))

    def standard_normal(self, size=None, dtype=float):
        return _wrap(np.asarray(self._rng.standard_normal(size=size), dtype=dtype))

    def binomial(self, n, p, size=None, dtype=int):
        return _wrap(np.asarray(self._rng.binomial(_unwrap(n), _unwrap(p), size=size), dtype=dtype))

    def exponential(self, scale=1.0, size=None, dtype=float):
        return _wrap(np.asarray(self._rng.exponential(_unwrap(scale), size=size), dtype=dtype))

    def randint(self, low, high=None, size=None, dtype=int):
        return _wrap(np.asarray(self._rng.integers(low, high, size=size), dtype=dtype))

    def choice(self, a, size=None, replace=True, p=None):
        return _wrap(np.asarray(self._rng.choice(_unwrap(a), size=size, replace=replace, p=_unwrap(p))))

    def permutation(self, num):
        return _wrap(self._rng.permutation(_unwrap(num)))

    def shuffle(self, a):
        if not isinstance(a, ndarray):
            raise TypeError("The array must be cupy.ndarray")
        self._rng.shuffle(a._data)


_RANDOM_STATE = RandomState()


def _random_state_function(name):
    """Build a module-level legacy random function that draws from the shared random state."""

    def draw(*args, **kwargs):
        return getattr(_RANDOM_STATE, name)(*args, **kwargs)

    draw.__name__ = name
    return draw


_RANDOM = _module(
    "cupy.random",
    {
        "default_rng": default_rng,
        "Generator": Generator,
        "BitGenerator": BitGenerator,
        "XORWOW": XORWOW,
        "RandomState": RandomState,
        "get_random_state": lambda: _RANDOM_STATE,
        "random": _random_state_function("random_sample"),
        **{
            name: _random_state_function(name)
            for name in (
                "seed",
                "rand",
                "randn",
                "random_sample",
                "uniform",
                "normal",
                "standard_normal",
                "binomial",
                "exponential",
                "randint",
                "choice",
                "permutation",
                "shuffle",
            )
        },
    },
)


class _Device:
    """CUDA device that works as a context manager."""

    def __init__(self, device=None):
        self.id = 0 if device is None else int(device)

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False

    def __eq__(self, other):
        return isinstance(other, _Device) and other.id == self.id

    def __hash__(self):
        return hash(self.id)

    def use(self):
        return None

    def synchronize(self):
        return None


class _Stream:
    """CUDA stream whose synchronization returns at once."""

    def synchronize(self):
        return None


_Stream.null = _Stream()

_ALLOCATOR = {"current": None}

_CUDA = _module(
    "cupy.cuda",
    {
        "Device": _Device,
        "Stream": _Stream,
        "is_available": is_available,
        "get_device_id": lambda: 0,
        "get_allocator": lambda: _ALLOCATOR["current"],
        "set_allocator": lambda allocator=None: _ALLOCATOR.update(current=allocator),
        "runtime": _module("cupy.cuda.runtime", {"getDeviceCount": lambda: 1}),
    },
)

cupy = _module(
    "cupy",
    {"__version__": "14.0.0", "linalg": _LINALG, "random": _RANDOM, "cuda": _CUDA},
    _cupy_getattr,
)


class csr_matrix:
    """Device CSR matrix that keeps a SciPy CSR matrix on the host."""

    format = "csr"
    ndim = 2

    def __init__(self, arg1, shape=None, dtype=None, copy=False):
        if isinstance(arg1, csr_matrix):
            host = arg1._host.copy() if copy else arg1._host
        elif scipy.sparse.issparse(arg1):
            host = scipy.sparse.csr_matrix(arg1)
        elif isinstance(arg1, tuple) and len(arg1) == 2 and all(isinstance(v, (int, np.integer)) for v in arg1):
            host = scipy.sparse.csr_matrix(arg1, dtype=dtype or "d")
        elif isinstance(arg1, tuple) and len(arg1) == 3:
            if not all(isinstance(part, ndarray) and part.ndim == 1 for part in arg1):
                raise ValueError("data, indices, and indptr should be 1-D")
            host = scipy.sparse.csr_matrix(tuple(part._data for part in arg1), shape=shape)
        elif isinstance(arg1, ndarray):
            if arg1.ndim > 2:
                raise ValueError("expected dimension <= 2 array or matrix")
            host = scipy.sparse.csr_matrix(np.atleast_2d(arg1._data))
        else:
            raise ValueError("Unsupported initializer format")
        if np.dtype(dtype or host.dtype).char not in "?fdFD":
            raise ValueError("Only bool, float32, float64, complex64 and complex128 are supported")
        self._host = host.astype(dtype) if dtype is not None else host

    @property
    def shape(self):
        return self._host.shape

    @property
    def dtype(self):
        return self._host.dtype

    @property
    def nnz(self):
        return self._host.nnz

    @property
    def data(self):
        return ndarray._from_data(self._host.data)

    @property
    def indices(self):
        return ndarray._from_data(self._host.indices)

    @property
    def indptr(self):
        return ndarray._from_data(self._host.indptr)

    @property
    def T(self):
        return self.transpose()

    def toarray(self, order=None, out=None):
        return ndarray._from_data(self._host.toarray(order=order))

    def todense(self, order=None, out=None):
        return self.toarray(order=order)

    def get(self, stream=None):
        return self._host.copy()

    def tocsr(self, copy=False):
        return csr_matrix(self, copy=copy)

    def copy(self):
        return csr_matrix(self, copy=True)

    def transpose(self, axes=None, copy=False):
        return csr_matrix(self._host.T.tocsr())

    def sum(self, axis=None, dtype=None, out=None):
        return _wrap(np.asarray(self._host.sum(axis=axis, dtype=dtype)))

    def dot(self, other):
        if isinstance(other, ndarray):
            with _quiet():
                return _wrap(np.asarray(self._host @ other._data))
        if isinstance(other, csr_matrix):
            return csr_matrix(self._host @ other._host)
        if _is_scalar(other):
            return csr_matrix(self._host * other)
        raise _unsupported(other)

    __matmul__ = dot

    def __repr__(self):
        return f"<{self.shape[0]}x{self.shape[1]} device sparse matrix of type '{self.dtype}' with {self.nnz} entries>"


def issparse(x):
    """Tell whether a value is a device sparse matrix."""
    return isinstance(x, csr_matrix)


class BSpline:
    """Device B-spline that evaluates through SciPy on the host, with CuPy's argument rules."""

    def __init__(self, t, c, k, extrapolate=True, axis=0):
        knots = ascontiguousarray(t, dtype=np.float64)
        coefficients = asarray(c)
        self._spline = scipy.interpolate.BSpline(knots._data, coefficients._data, k, extrapolate=extrapolate, axis=axis)
        self.t = knots
        self.c = coefficients
        self.k = int(k)
        self.extrapolate = self._spline.extrapolate
        self.axis = self._spline.axis

    @classmethod
    def construct_fast(cls, t, c, k, extrapolate=True, axis=0):
        spline = object.__new__(cls)
        spline._spline = scipy.interpolate.BSpline.construct_fast(t._data, c._data, k, extrapolate, axis)
        spline.t, spline.c, spline.k = t, c, k
        spline.extrapolate, spline.axis = extrapolate, axis
        return spline

    @classmethod
    def _from_host(cls, spline):
        return cls.construct_fast(
            ndarray._from_data(spline.t), ndarray._from_data(spline.c), spline.k, spline.extrapolate, spline.axis
        )

    @property
    def tck(self):
        return self.t, self.c, self.k

    @classmethod
    def design_matrix(cls, x, t, k, extrapolate=False):
        x = asarray(x)._data.astype(np.float64)
        t = asarray(t)._data.astype(np.float64)
        if not (np.all(np.isfinite(x)) and np.all(np.isfinite(t))):
            raise ValueError("Array must not contain infs or nans.")
        if not extrapolate and (x.min() < t[k] or x.max() > t[t.shape[0] - k - 1]):
            raise ValueError(f"Out of bounds w/ x = {x}.")
        return csr_matrix(scipy.interpolate.BSpline.design_matrix(x, t, k, extrapolate=bool(extrapolate)))

    def __call__(self, x, nu=0, extrapolate=None):
        x = asarray(x)
        with _quiet():
            return ndarray._from_data(np.asarray(self._spline(x._data, nu=nu, extrapolate=extrapolate)))

    def derivative(self, nu=1):
        return BSpline._from_host(self._spline.derivative(nu))

    def antiderivative(self, nu=1):
        return BSpline._from_host(self._spline.antiderivative(nu))

    def integrate(self, a, b, extrapolate=None):
        return _wrap(np.asarray(self._spline.integrate(float(a), float(b), extrapolate=extrapolate)))


def solve_triangular(a, b, trans=0, lower=False, unit_diagonal=False, overwrite_b=False, check_finite=False):
    """Solve a triangular system without checking the diagonal, as cuBLAS's trsm does."""
    _require_device(a, b)
    if a.ndim != 2 or a.shape[0] != a.shape[1]:
        raise ValueError("expected square matrix")
    if a.shape[0] != b.shape[0]:
        raise ValueError(f"shapes of a {a.shape} and b {b.shape} are incompatible")
    trans = {"N": 0, "T": 1, "C": 2}.get(trans, trans)
    dtype = _linalg_dtype(a, b)
    rhs = b._data.astype(dtype).reshape(b.shape[0], -1)
    trsm = scipy.linalg.get_blas_funcs("trsm", (a._data.astype(dtype), rhs))
    with _quiet():
        x = trsm(1.0, a._data.astype(dtype), rhs, lower=lower, trans_a=trans, diag=unit_diagonal)
    return ndarray._from_data(x.reshape(b.shape))


def errstate(*, divide=None, over=None, under=None, invalid=None, linalg=None):
    """Set the linalg error mode within a block, as ``cupyx.errstate`` does."""
    return _ErrorState(divide=divide, over=over, under=under, invalid=invalid, linalg=linalg)


class _ErrorState:
    """Context manager behind :func:`errstate`."""

    def __init__(self, **settings):
        self._settings = settings

    def __enter__(self):
        self._old = seterr(**self._settings)

    def __exit__(self, *exc_info):
        seterr(**self._old)
        return False


def seterr(*, divide=None, over=None, under=None, invalid=None, linalg=None):
    """Set the linalg error mode, which is the only mode CuPy implements."""
    old = geterr()
    if divide is not None or over is not None or under is not None or invalid is not None:
        raise NotImplementedError()
    if linalg is not None:
        if linalg not in ("ignore", "raise"):
            raise NotImplementedError()
        _LINALG_CONFIG.linalg = linalg
    return old


def geterr():
    """Return the current error modes."""
    return {"divide": None, "over": None, "under": None, "invalid": None, "linalg": _linalg_mode()}


def scatter_add(a, slices, value):
    """Add values at indices, with repeated indices adding up."""
    a._scatter_op(slices, value, "add")


def _special_getattr(name):
    if name in _SPECIAL_UFUNC_NAMES:
        return _Ufunc(getattr(scipy.special, name))
    raise AttributeError(f"module 'cupyx.scipy.special' has no attribute {name!r}")


_SPARSE = _module("cupyx.scipy.sparse", {"csr_matrix": csr_matrix, "issparse": issparse, "isspmatrix": issparse})
_INTERPOLATE = _module("cupyx.scipy.interpolate", {"BSpline": BSpline})
_SCIPY_LINALG = _module("cupyx.scipy.linalg", {"solve_triangular": solve_triangular})
_SPECIAL = _module("cupyx.scipy.special", getattr_function=_special_getattr)
_CUPYX_SCIPY = _module(
    "cupyx.scipy",
    {"sparse": _SPARSE, "interpolate": _INTERPOLATE, "linalg": _SCIPY_LINALG, "special": _SPECIAL},
)

cupyx = _module(
    "cupyx",
    {"errstate": errstate, "seterr": seterr, "geterr": geterr, "scatter_add": scatter_add, "scipy": _CUPYX_SCIPY},
)

MODULES = {
    "cupy": cupy,
    "cupy.linalg": _LINALG,
    "cupy.random": _RANDOM,
    "cupy.cuda": _CUDA,
    "cupy.cuda.runtime": _CUDA.runtime,
    "cupyx": cupyx,
    "cupyx.scipy": _CUPYX_SCIPY,
    "cupyx.scipy.sparse": _SPARSE,
    "cupyx.scipy.interpolate": _INTERPOLATE,
    "cupyx.scipy.linalg": _SCIPY_LINALG,
    "cupyx.scipy.special": _SPECIAL,
}
