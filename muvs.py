"""
An interface to VapourSynth

*** DO NOT PUBLISH MODULES THAT DEPEND ON THIS ***

objects:
    core (resembles vapoursynth.core)
    options

functions:
    pollute

functions for expression:
    Abs, Exp, Not, Log, Sqrt, 
    Min, Max, Conditional

"""

from collections import OrderedDict
import collections.abc
from contextlib import contextmanager
import functools
import inspect
import itertools
from math import exp, log, sqrt
import numbers
from typing import Union, Optional, Dict, List, Callable

import vapoursynth as vs
from vapoursynth import core as _vscore


class _Core:
    def __init__(self):
        self._registered_funcs = {} # type: Dict[str, Callable[..., '_VideoNode']]

    def __setattr__(self, name, value):
        if name in ["num_threads", "add_cache", "max_cache_size"]:
            setattr(_vscore, name, value)
        else:
            if callable(value):
                if name[0].isupper() and not hasattr(_vscore, name):
                    self._registered_funcs[name] = value
                else:
                    raise AttributeError("Attribute name should be capitalized")
            else:
                self.__dict__[name] = value

    def __getattr__(self, name):
        try:
            attr = getattr(_vscore, name)
        except AttributeError as e:
            if name in self._registered_funcs:
                return self._registered_funcs[name]
            else:
                raise e
        else:
            if isinstance(attr, vs.Plugin):
                return _Plugin(attr)
            else:
                return attr

    def __dir__(self) -> List[str]:
        return dir(_vscore) + sorted(list(self._registered_funcs.keys()))

    def register_functions(self, **kwargs: Dict[str, Callable[..., '_VideoNode']]):
        if all((name[0].isupper() and not hasattr(_vscore, name)) 
               for name in kwargs.keys()):

            self._registered_funcs.update(kwargs)
        else:
            raise ValueError("Registration error.")

core = _Core()


def pollute(*modules):
    class _FakeVS:
        def __init__(self):
            self.VideoNode = _VideoNode
            self.get_core = lambda : core

        def __getattr__(self, name):
            return getattr(vs, name)

    _vs = _FakeVS()

    # modify symbol table of each module
    if len(modules) == 0:
        import sys
        for name, module in sys.modules.items():
            if (name not in ("__vapoursynth__", "__main__") and
                getattr(module, "core", None) is not core and
                ((getattr(module, "vs", None) is vs) or (getattr(module, "core", None) is _vscore))
                ):
                module.core = core
                module.vs = _vs
    else:
        for module in modules:
            module.core = core
            module.vs = _vs


class _Options:
    def __init__(self):
        self._file = ""
        self._mode = 'a'
        self._open_kwargs = {}
        self._arithmetic_expr = False
        self._buffer = [] # type: List[str]
        self._record = False
        self._include_header = True

    @property
    def arithmetic_expr(self) -> bool:
        return self._arithmetic_expr

    def enable_arithmetic_expr(self):
        self._arithmetic_expr = True

    def disable_arithmetic_expr(self):
        self._arithmetic_expr = False

    @contextmanager
    def expr(self):
        prev_expr = self._arithmetic_expr

        self.enable_arithmetic_expr()

        try:
            yield None
        finally:
            self._arithmetic_expr = prev_expr

    @property
    def buffer(self) -> List[str]:
        return self._buffer

    @property
    def is_recording(self) -> bool:
        return self._record

    def start_recording(self, file, mode='a', **open_kwargs):
        self._file = file
        self._mode = mode
        self._open_kwargs = open_kwargs
        self._buffer.clear()
        self._record = True

        if self._include_header:
            self._buffer.append(
                "import vapoursynth as vs\n"
                "from vapoursynth import core\n"
                "\n"
                f"core.add_cache = {core.add_cache}\n"
                f"core.num_threads = {core.num_threads}\n"
                f"core.max_cache_size = {core.max_cache_size}\n"
                "\n")

            self._include_header = False

    def end_recording(self):
        if self._buffer:
            with open(file=self._file, mode=self._mode, **self._open_kwargs) as f:
                f.writelines(self._buffer)
                f.writelines(['\n'] * 5)

        self._file = ""
        self._mode = 'a'
        self._open_kwargs.clear()
        self._buffer.clear()
        self._record = False

    @contextmanager
    def record(self, file, mode='a', **open_kwargs):
        self.start_recording(file=file, mode=mode, **open_kwargs)

        try:
            yield None
        finally:
            self.end_recording()

options = _Options()


def _build_repr() -> Callable[..., str]:
    _clip_name_mapping = {} # type: Dict[vs.VideoNode, str]

    def closure(obj, default_prefix="unknown") -> str:
        if isinstance(obj, vs.VideoNode):
            return _clip_name_mapping.setdefault(obj, f"{default_prefix}{len(_clip_name_mapping)}")
        elif isinstance(obj, _VideoNode):
            return _clip_name_mapping.setdefault(obj._node, f"{default_prefix}{len(_clip_name_mapping)}")
        elif isinstance(obj, collections.abc.Sequence) and not isinstance(obj, (str, bytes, bytearray)):
            return '[' + ', '.join(closure(elem, default_prefix) for elem in obj) + ']'
        elif isinstance(obj, (vs.ColorFamily, vs.PresetFormat, vs.SampleType)):
            return f"vs.{obj!s}"
        elif isinstance(obj, vs.Format):
            return f"core.register_format({', '.join(k + '=' + closure(v) for k, v in obj._as_dict().items())})"
        else:
            return repr(obj)

    return closure

_repr = _build_repr()


class _Plugin:
    def __init__(self, plugin: vs.Plugin, injected_clip: Optional[vs.VideoNode] = None):
        if isinstance(plugin, vs.Plugin):
            self._plugin = plugin
        else:
            raise TypeError(f"{type(self).__name__!r}: Unknown plugin ({type(plugin)})")

        if injected_clip is None or isinstance(injected_clip, vs.VideoNode):
            self._injected_clip = injected_clip
        else:
            raise TypeError(f"{type(self).__name__!r}: Unknown injected clip ({type(injected_clip)})")

    def __getattr__(self, function_name):
        attr = getattr(self._plugin, function_name)

        if isinstance(attr, vs.Function):
            func = attr

            @functools.wraps(func)
            def closure(*args, **kwargs) -> '_VideoNode':
                if self._injected_clip is not None:
                    args = (self._injected_clip, ) + args

                def get_node(obj):
                    if isinstance(obj, vs.VideoNode):
                        return obj
                    elif isinstance(obj, _VideoNode):
                        return obj._node
                    elif isinstance(obj, _ArithmeticExpr):
                        return obj.compute()._node
                    elif isinstance(obj, collections.abc.Sequence) and not isinstance(obj, (str, bytes, bytearray)):
                        return type(obj)(get_node(item) for item in obj)
                    elif callable(obj):
                        class _remove_wrap:
                            """Fixes callables that returns VideoNode"""
                            def __init__(self, func):
                                self.func = func

                            def __call__(self, *args, **kwargs):
                                output = self.func(*args, **kwargs)
                                if isinstance(output, _VideoNode):
                                    output = output._node
                                return output

                            def __repr__(self):
                                return repr(self.func)

                        return _remove_wrap(obj)
                    else:
                        return obj

                def get_key(key):
                    if key.startswith('_'):
                        return key[1:]
                    else:
                        return key

                args = get_node(args)
                kwargs = dict((get_key(key), get_node(value)) for key, value in kwargs.items())

                # process
                output = func(*args, **kwargs) # type: vs.VideoNode
                _ = _repr(output, default_prefix="clip") # register output

                if options.is_recording:
                    options.buffer.append(self._get_str(func, args, kwargs, output) + '\n')

                return _VideoNode(output)

            return closure

        else:
            return attr

    def __hash__(self):
        return hash(self._plugin)

    def __dir__(self):
        return dir(self._plugin)

    @staticmethod
    def _get_str(func: vs.Function, args, kwargs, output):
        output_str = ""

        def diff_str(clip1: vs.VideoNode, clip2: vs.VideoNode):
            """Compare two clips and output a string of their difference"""
            res = []
            for attr in ["width", "height", "num_frames"]:
                if getattr(clip1, attr) != getattr(clip2, attr):
                    res.append(f"{attr}: {getattr(clip1, attr)} -> {getattr(clip2, attr)}")
            if clip1.format.name != clip2.format.name:
                res.append(f"format: {clip1.format.name} -> {clip2.format.name}")
            if clip1.fps != clip2.fps:
                res.append(f"fps: {clip1.fps_num}/{clip1.fps_den} -> {clip2.fps_num}/{clip2.fps_den}")
            return ', '.join(res)

        if len(args) > 0 and isinstance(args[0], vs.VideoNode):
            if diff_str(args[0], output) != "":
                output_str += f"# {diff_str(args[0], output)}\n"
        elif kwargs.get("clip", None):
            if diff_str(kwargs["clip"], output) != "":
                output_str += f"# {diff_str(kwargs['clip'], output)}\n"
        else:
            output_str += (f"# output: {output.width} x {output.height}, {output.format.name}, "
                           f"{output.num_frames} frames, {output.fps_num}/{output.fps_den} fps\n")

        args_dict = inspect.signature(func).bind(*args, **kwargs).arguments

        # replace clip in args_dict.values() with name of clip
        call_args = ', '.join(f"{k}={_repr(v)}" for k, v in args_dict.items() if v is not None)
        call_str = f"core.{func.plugin.namespace}.{func.name}({call_args})"

        output_str += f"{_repr(output, default_prefix='clip')} = {call_str}\n"

        return output_str


class _ArithmeticExpr:
    def __init__(self, obj): 
        if options.arithmetic_expr:
            if isinstance(obj, _VideoNode):
                self._expr = (obj,)
            elif isinstance(obj, type(self)):
                self._expr = obj._expr
            elif isinstance(obj, tuple):
                self._expr = obj
            else:
                raise TypeError(f"{type(self).__name__!r}: Unknown initializer ({type(obj)})")
        else:
            raise RuntimeError("Arithmetic expression is disabled.")

    def __getattr__(self, name):
        if hasattr(_vscore, name) or hasattr(self.clips[0], name):
            return getattr(self.compute(), name)
        else:
            raise AttributeError(f"{type(self).__name__!r} object has no attribute {name!r}")

    def __bool__(self):
        return NotImplemented

    def __repr__(self):
        if len(self._expr) > 1:
            return ' '.join(_repr(item) for item in self._expr)
        else:
            return ""

    def __hash__(self):
        return hash(self.clips + (self.expr,))

    @property
    def clips(self):
        """Ordered _VideoNode(s)"""
        return tuple(OrderedDict((obj, None) for obj in self._expr if isinstance(obj, _VideoNode)).keys())

    @property
    def expr(self) -> str:
        def _to_var_factory() -> Callable[..., str]:
            _clip_var_mapping = {} # type: Dict[_VideoNode, str]
            _vars = "xyzabcdefghijklmnopqrstuvw"
            def closure(obj):
                if isinstance(obj, _VideoNode):
                    if obj in _clip_var_mapping or len(_clip_var_mapping) < len(_vars):
                        return _clip_var_mapping.setdefault(obj, f"{_vars[len(_clip_var_mapping)]}")
                    else:
                        raise RuntimeError(f"{type(self).__name__!r}: Too many nodes")
                else:
                    return obj
            return closure

        _to_var = _to_var_factory()

        if len(self._expr) > 1:
            return ' '.join(_to_var(item) for item in self._expr).strip()
        else:
            return ""
    
    @property
    def lut_func(self) -> Callable[..., numbers.Integral]:
        clips = self.clips

        assert len(clips) in [1, 2]

        def rpn_parser(stack, next_val: str):
            if isinstance(next_val, str): # operators or constants
                if next_val in ["abs", "exp", "log", "sqrt"]:
                    return stack[:-1] + (f"{next_val}({stack[-1]})",)
                elif next_val == "not":
                    return stack[:-1] + (f"(not {stack[-1]})",)
                elif next_val == 'dup':
                    return stack + (stack[-1],)
                elif next_val in ['<', '<=', '>', '>=', '+', '-', '*', '/', 'and', 'or']:
                    return stack[:-2] + (f"({stack[-2]} {next_val} {stack[-1]})",)
                elif next_val == '=':
                    return stack[:-2] + (f"({stack[-2]} == {stack[-1]})",)
                elif next_val == '= not': # not triggered by current implementation
                    return stack[:-2] + (f"({stack[-2]} != {stack[-1]})",)
                elif next_val == 'pow':
                    return stack[:-2] + (f"({stack[-2]} ** {stack[-1]})",)
                elif next_val == 'xor':
                    return stack[:-2] + (f"(({stack[-2]} and not {stack[-1]}) or (not {stack[-2]} and {stack[-1]}))",)
                elif next_val in ['min', 'max']:
                    return stack[:-2] + (f"{next_val}({stack[-2]}, {stack[-1]})",)
                elif next_val == '?':
                    return stack[:-3] + (f"({stack[-2]} if {stack[-3]} else {stack[-1]})",)
                else: # vars, constants
                    return stack + (next_val,)
            else:
                raise TypeError(f"Unknown node in expr ({type(next_val)})")

        # postfix2infix
        func_expr = functools.reduce(rpn_parser, self.expr.split(), ())

        assert len(func_expr) == 1, str(func_exp)

        func_impl = func_expr[0]
        func_impl = f"min(max(int({func_impl} + 0.5), 0), {(2 ** clips[0].format.bits_per_sample) - 1})" # clamp

        if len(clips) == 1:
            lut_str = f"lambda x: {func_impl}"
        else: # len(clips) == 2
            lut_str = f"lambda x, y: {func_impl}"

        class _LambdaFunction:
            def __init__(self, func_str: str):
                self.func = eval(func_str)
                self.func_str = func_str
            
            def __call__(self, *args, **kwargs):
                return self.func(*args, **kwargs)
            
            def __repr__(self):
                return self.func_str

        return _LambdaFunction(lut_str)

    def _operate(self, op: str, *operands, position=0) -> '_ArithmeticExpr':
        # accepted operands: Union[numbers.Real, _VideoNode, _ArithmeticExpr]

        def to_element(obj):
            if isinstance(obj, numbers.Real):
                return (repr(float(obj)),)
            elif isinstance(obj, _VideoNode):
                return (obj,)
            elif isinstance(obj, _ArithmeticExpr):
                return obj._expr
            else:
                raise TypeError(f"{type(self).__name__!r}: Unknown input ({type(obj)})")

        _expr = [to_element(operand) for operand in operands]
        _expr.insert(position, self._expr)

        # "X X *" -> "X dup *"
        # _expr[i] == self._expr cannot be used since == is overloaded to return non-boolean value
        for i in range(position + 1, len(_expr)):
            if len(_expr[i]) == len(self._expr) and all((hash(x) == hash(y)) for x, y in zip(_expr[i], self._expr)):
                _expr[i] = ("dup",)

        _expr.append((op,))

        return type(self)(tuple(itertools.chain(*_expr)))

    def compute(self, planes=None, bits=None, use_lut=None) -> '_VideoNode':
        if options.arithmetic_expr:
            if self.expr in ['', 'x']: # empty expr
                return _VideoNode(clips[0]._node)
            else:
                clips = self.clips

                if bits is None:
                    not_equal_bits = lambda clip1, clip2: clip1.format.bits_per_sample != clip2.format.bits_per_sample

                    if len(clips) >= 2 and any(not_equal_bits(clips[0], clip) for clip in clips[1:]):
                        raise ValueError('"bits" must be specified.')
                    else:
                        bits = clips[0].format.bits_per_sample

                is_int = lambda clip: clip.format.sample_type == vs.INTEGER
                get_bits = lambda clip: clip.format.bits_per_sample
                lut_available = lambda clips: len(clips) <= 2 and all(map(is_int, clips)) and sum(map(get_bits, clips)) <= 20

                if use_lut is None:
                    use_lut = lut_available(clips) and len(self._expr) >= 15
                elif use_lut and not lut_available(clips):
                    raise ValueError("Lut computation is not available")

                # process
                if use_lut: # std.Lut() / std.Lut2()
                    if len(clips) == 1:
                        return core.std.Lut(clips[0], planes=planes, bits=bits, function=self.lut_func)
                    else: # len(clips) == 2
                        return core.std.Lut2(clips[0], clips[1], planes=planes, bits=bits, function=self.lut_func)

                else: # std.Expr()
                    if planes is None:
                        expr = self.expr
                    else:
                        if isinstance(planes, int):
                            planes = [planes]

                        expr = [
                            (self.expr if i in planes else "") 
                            for i in range(clips[0]._node.format.num_planes)]

                    in_format = clips[0]._node.format
                    out_format = core.register_format(
                        color_family=in_format.color_family, 
                        sample_type=in_format.sample_type, 
                        bits_per_sample=bits, 
                        subsampling_w=in_format.subsampling_w, 
                        subsampling_h=in_format.subsampling_h
                    )

                    return core.std.Expr(clips=clips, expr=expr, format=out_format)
                
        else:
            raise RuntimeError("Arithmetic expression is disabled.")

    # Arithmetic methods

    # unary operations
    def __neg__(self):
        return self._operate('-', 0, position=1)

    def __pos__(self):
        return type(self)(self)

    def __abs__(self):
        return self._operate('abs')

    # custom unary operations
    def __exp__(self):
        return self._operate('exp')

    def __log__(self):
        return self._operate('log')

    def __not__(self):
        return self._operate('not')

    def __sqrt__(self):
        return self._operate('sqrt')

    # binary operations
    def __lt__(self, other):
        return self._operate('<', other, position=0)

    def __le__(self, other):
        return self._operate('<=', other, position=0)

    def __eq__(self, other):
        return self._operate('=', other, position=0)

    def __ne__(self, other):
        return self._operate('= not', other, position=0)

    def __gt__(self, other):
        return self._operate('>', other, position=0)

    def __ge__(self, other):
        return self._operate('>=', other, position=0)

    def __add__(self, other):
        return self._operate('+', other, position=0)

    def __radd__(self, other):
        return self._operate('+', other, position=1)

    def __sub__(self, other):
        return self._operate('-', other, position=0)

    def __rsub__(self, other):
        return self._operate('-', other, position=1)

    def __mul__(self, other):
        return self._operate('*', other, position=0)

    def __rmul__(self, other):
        return self._operate('*', other, position=1)

    def __truediv__(self, other):
        return self._operate('/', other, position=0)

    def __rtruediv__(self, other):
        return self._operate('/', other, position=1)

    def __pow__(self, other, module=None):
        if module is None:
            if isinstance(other, numbers.Integral) and int(other) > 0:
                # exponentiation by squaring
                binary = format(int(other), 'b')
                _pre_expr = ""
                _post_expr = ""

                for b in reversed(binary[1:]):
                    if b == '0':
                        _pre_expr += "dup * "
                    else: # b == '1'
                        _pre_expr += "dup dup * "
                        _post_expr += "* "

                return self._operate((_pre_expr + _post_expr).strip())
            else:
                return self._operate('pow', other, position=0)
        else:
            return NotImplemented

    def __rpow__(self, other):
        return self._operate('pow', other, position=1)

    def __and__(self, other):
        return self._operate('and', other, position=0)

    def __rand__(self, other):
        return self._operate('and', other, position=1)

    def __or__(self, other):
        return self._operate('or', other, position=0)

    def __ror__(self, other):
        return self._operate('or', other, position=1)

    def __xor__(self, other):
        return self._operate('xor', other, position=0)

    def __rxor__(self, other):
        return self._operate('xor', other, position=1)

    # custom binary operations
    def __min__(self, other):
        return self._operate('min', other, position=0)

    def __rmin__(self, other):
        return self._operate('min', other, position=1)

    def __max__(self, other):
        return self._operate('max', other, position=0)

    def __rmax__(self, other):
        return self._operate('max', other, position=1)

    # custom ternary operation
    def __conditional__(self, other_true, other_false):
        return self._operate('?', other_true, other_false, position=0)

    def __rconditional__(self, other_condition, other_false):
        return self._operate('?', other_condition, other_false, position=1)

    def __rrconditional__(self, other_condition, other_true):
        return self._operate('?', other_condition, other_true, position=2)


def _build_VideoNode():
    _plane_idx_mapping = {
        vs.YUV: {'Y': 0, 'U': 1, 'V': 2}, 
        vs.RGB: {'R': 0, 'G': 1, 'B': 2}, 
        vs.GRAY: {'GRAY': 0, 'Y': 0}, 
        vs.YCOCG: {'Y': 0, 'CO': 1, 'CG': 2}
    }

    def __init__(self, node: vs.VideoNode):
        if not isinstance(node, vs.VideoNode):
            raise TypeError(f"{type(self).__name__!r}: Unknown input ({type(node)})")
        self._node = node

    def __getattr__(self, name):
        if name[0].isupper(): # non-standard attributes
            if (self.format.color_family in _plane_idx_mapping and
                name in _plane_idx_mapping[self.format.color_family]):

                idx = _plane_idx_mapping[self.format.color_family][name]
                return self.std.ShufflePlanes(planes=idx, colorfamily=vs.GRAY)

            elif hasattr(core, name):
                func = getattr(core, name)
                return functools.partial(func, self)
            else:
                raise AttributeError(f"{type(self).__name__!r} object has no attribute {name!r}")

        elif hasattr(_vscore, name) and isinstance(getattr(_vscore, name), vs.Plugin):
            plugin = getattr(_vscore, name)
            return _Plugin(plugin, self._node)
        else:
            attr = getattr(self._node, name)

            if callable(attr): # set_output(), etc
                @functools.wraps(attr)
                def closure(*args, **kwargs):
                    if options.is_recording:
                        args_str = ', '.join(args)
                        kwargs_str = ', '.join(f"{k}={_repr(v)}" for k, v in kwargs)
                        call_str = ', '.join(s for s in [args_str, kwargs_str] if s != '')
                        options.buffer.append(f"{_repr(self)}.{name}({call_str})\n")

                    return attr(*args, **kwargs)

                return closure

            else:
                return attr

    def __len__(self):
        return self.num_frames

    def __bool__(self):
        return NotImplemented

    def __dir__(self):
        return dir(self._node) + _plane_identifiers[self.format.color_family]

    def __hash__(self):
        return hash(self._node)

    def __iter__(self):
        return (type(self)(clip) for clip in iter(self._node))

    def __getitem__(self, val):
        if isinstance(val, slice):
            start, stop, step = val.indices(self.num_frames)

            if step > 0:
                stop -= 1
            else: # step < 0
                start, stop = stop + 1, start

            ret = self

            if start != 0 or stop != ret.num_frames - 1:
                ret = ret.std.Trim(first=start, last=stop)

            if step < 0:
                ret = ret.std.Reverse()

            if abs(step) != 1:
                ret = ret.std.SelectEvery(cycle=abs(step), offsets=[0])

            if ret is self: # shallow copy
                ret = type(self)(self._node)

            return ret

        elif isinstance(val, int):
            if val < 0:
                n = self.num_frames + val
            else:
                n = val

            if n < 0 or (self.num_frames > 0 and n >= self.num_frames):
                raise IndexError("index out of range")
            else:
                return self.std.Trim(first=n, length=1)
        else:
            raise TypeError(f"indices must be integers or slices, not {type(val)}")

    _dict = locals().copy()

    _create_method = (lambda name: 
                          lambda self, *args: 
                              getattr(_ArithmeticExpr, name)(_ArithmeticExpr(self), *args))

    _dict.update((attr, _create_method(attr)) for attr in ["__neg__", "__pos__", "__abs__", "__exp__", 
        "__log__", "__not__", "__sqrt__", "__lt__", "__le__", "__eq__", "__ne__", "__gt__", "__ge__", 
        "__add__", "__radd__", "__sub__", "__rsub__", "__mul__", "__rmul__", "__truediv__", "__rtruediv__", 
        "__pow__", "__rpow__", "__and__", "__rand__", "__xor__", "__rxor__", "__or__", "__ror__", 
        "__min__", "__rmin__", "__max__", "__rmax__", "__conditional__", "__rconditional__", "__rrconditional__"])

    return type("_VideoNode", (), _dict)

_VideoNode = _build_VideoNode()


# custom operations
Abs = abs


@functools.singledispatch
def Exp(x):
    return exp(x)

@Exp.register(_ArithmeticExpr)
@Exp.register(_VideoNode)
def _(x) -> _ArithmeticExpr:
    return x.__exp__()


@functools.singledispatch
def Not(x):
    return not x

@Not.register(_ArithmeticExpr)
@Not.register(_VideoNode)
def _(x) -> _ArithmeticExpr:
    return x.__not__()


@functools.singledispatch
def Log(x):
    return log(x)

@Log.register(_ArithmeticExpr)
@Log.register(_VideoNode)
def _(x) -> _ArithmeticExpr:
    return x.__log__()


@functools.singledispatch
def Sqrt(x):
    return sqrt(x)

@Sqrt.register(_ArithmeticExpr)
@Sqrt.register(_VideoNode)
def _(x) -> _ArithmeticExpr:
    return x.__sqrt__()


def Min(x, y):
    if isinstance(x, numbers.Real) and isinstance(y, numbers.Real):
        return min(x, y)
    elif hasattr(x, "__min__"):
        return x.__min__(y)
    elif hasattr(y, "__rmin__"):
        return y.__rmin__(x)
    else:
        raise TypeError(f"'Min': Unknown input ({type(x)}, {type(y)})")


def Max(x, y):
    if isinstance(x, numbers.Real) and isinstance(y, numbers.Real):
        return max(x, y)
    elif hasattr(x, "__max__"):
        return x.__max__(y)
    elif hasattr(y, "__rmax__"):
        return y.__rmax__(x)
    else:
        raise TypeError(f"'Max': Unknown input ({type(x)}, {type(y)})")


def Conditional(condition, condition_if_true, condition_if_false):
    try:
        return condition_if_true if bool(condition) else condition_if_false
    except TypeError:
        if hasattr(condition, "__conditional__"):
            return condition.__conditional__(condition_if_true, condition_if_false)
        elif hasattr(condition_if_true, "__rconditional__"):
            return condition_if_true.__rconditional__(condition, condition_if_false)
        elif hasattr(condition_if_false, "__rrconditional__"):
            return condition_if_false.__rrconditional__(condition, condition_if_true)
        else:
            raise TypeError(f"'Conditional': Unknown input ({type(condition)}, "
                            f"{type(condition_if_true)}, {type(condition_if_false)})")
