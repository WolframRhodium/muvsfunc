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
import math
import numbers
from typing import Union, Optional, Dict, List, Callable, MutableMapping, MutableSet
import weakref

import vapoursynth as vs
from vapoursynth import core as _vscore


__all__ = [
    "core", "expr", "pollute", "Expr", "record", "Recorder", 
    "Abs", "Exp", "Not", "And", "Or", "Xor", "Log", "Sqrt", 
    "Min", "Max", "Conditional"]


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
                vars(self)[name] = value

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
            self.core = core
            self.get_core = lambda : core

        def __getattr__(self, name):
            return getattr(vs, name)

    _vs = _FakeVS()

    # modify symbol table of each module
    if len(modules) == 0:
        import sys
        for name, module in sys.modules.items():
            if (
                name not in ("__vapoursynth__", "__main__") and
                getattr(module, "core", None) is not core and
                ((getattr(module, "vs", None) is vs) or 
                 (getattr(module, "core", None) is _vscore))
            ):
                module.core = core
                module.vs = _vs
    else:
        for module in modules:
            module.core = core
            module.vs = _vs


arithmetic_expr : bool = False

@contextmanager
def expr():
    global arithmetic_expr
    prev_expr = arithmetic_expr

    arithmetic_expr = True

    try:
        yield None
    finally:
        arithmetic_expr = prev_expr


class Recorder:
    _live_recorders : MutableSet["Recorder"] = weakref.WeakSet()

    def __init__(self):
        self.buffer : List[str] = []
        self.is_recording : bool = False
        Recorder._live_recorders.add(self)

    def start_recording(self, include_header=False):
        self.is_recording = True

        if include_header:
            self.buffer.append(
                "import vapoursynth as vs\n"
                "from vapoursynth import core\n"
                "\n"
                f"core.add_cache = {core.add_cache}\n"
                f"core.num_threads = {core.num_threads}\n"
                f"core.max_cache_size = {core.max_cache_size}\n"
                "\n")

    def end_recording(self, filename_or_stream, mode='a', **open_kwargs):
        self.is_recording = False

        if self.buffer:
            if isinstance(filename_or_stream, str):
                with open(filename_or_stream, mode=mode, **open_kwargs) as f:
                    f.writelines(self.buffer)
            else:
                stream = filename_or_stream
                stream.writelines(self.buffer)

            self.buffer.clear()

    def write(self, text):
        assert isinstance(text, str)
        self.buffer.append(text)


@contextmanager
def record(filename_or_stream, mode='a', include_header=False, **open_kwargs):
    recorder = Recorder()

    recorder.start_recording(include_header)

    try:
        yield recorder
    finally:
        recorder.end_recording(filename_or_stream=filename_or_stream, mode=mode, **open_kwargs)


def _build_repr() -> Callable[..., str]:
    _clip_name_mapping = weakref.WeakKeyDictionary() # type: MutableMapping[vs.VideoNode, str]
    counter = 0

    def closure(obj, default_prefix="unknown") -> str:
        if isinstance(obj, vs.VideoNode):
            if obj in _clip_name_mapping:
                return _clip_name_mapping[obj]

            else:
                nonlocal counter
                name = f"{default_prefix}{counter}"
                _clip_name_mapping[obj] = name
                counter += 1
                return name

        elif isinstance(obj, _VideoNode):
            return closure(obj._node, default_prefix)

        elif isinstance(obj, collections.abc.Sequence) and not isinstance(obj, (str, bytes, bytearray)):
            return f"[{', '.join(closure(elem, default_prefix) for elem in obj)}]"

        elif isinstance(obj, (vs.ColorFamily, vs.PresetFormat, vs.SampleType)):
            return f"vs.{obj!s}"

        elif isinstance(obj, vs.Format):
            arg_str = ', '.join(f"{k}={closure(v)}" for k, v in obj._as_dict().items())
            return f"core.register_format({arg_str})"

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
            def closure(*args, **kwargs):
                if self._injected_clip is not None:
                    args = (self._injected_clip, ) + args

                def get_node(obj):
                    if isinstance(obj, vs.VideoNode):
                        return obj
                    elif isinstance(obj, _VideoNode):
                        return obj._node
                    elif isinstance(obj, _ArithmeticExpr):
                        return obj.compute()._node
                    elif (
                        isinstance(obj, collections.abc.Sequence) and 
                        not isinstance(obj, (str, bytes, bytearray))
                    ):
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

                func_arg_names = (
                    key[:key.find(':')] 
                    for key in func.signature.split(';') 
                    if key != '')

                for _, arg_name in zip(args, func_arg_names):
                    if arg_name in kwargs:
                        raise TypeError(
                            f"{func.plugin.namespace}.{func.name}() "
                            f"got multiple values for argument \'{arg_name}\'")

                # process
                output = func(*args, **kwargs)

                if isinstance(output, vs.VideoNode):
                    _ = _repr(output, default_prefix="clip") # register output

                    for recorder in Recorder._live_recorders:
                        if recorder.is_recording:
                            recorder.buffer.append(self._get_str(func, args, kwargs, output) + '\n')

                    return _VideoNode(output)
                else:
                    return output

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


def _postfix2infix(expr: str) -> str:
    stack = []

    for next_val in expr.split():
        if next_val in ["abs", "exp", "log", "sqrt"]:
            stack[-1] = f"{next_val}({stack[-1]})"
        elif next_val == "not":
            stack[-1] = f"(not {stack[-1]})"
        elif next_val == 'dup':
            stack.append(stack[-1])
        elif next_val in ['<', '<=', '>', '>=', '+', '-', '*', '/', 'and', 'or']:
            stack = stack[:-2] + [f"({stack[-2]} {next_val} {stack[-1]})"]
        elif next_val == '=':
            stack = stack[:-2] + [f"({stack[-2]} == {stack[-1]})"]
        elif next_val == 'pow':
            stack = stack[:-2] + [f"({stack[-2]} ** {stack[-1]})"]
        elif next_val == 'xor':
            stack = stack[:-2] + [f"(({stack[-2]} and not {stack[-1]}) or "
                                  f"(not {stack[-2]} and {stack[-1]}))"]
        elif next_val in ['min', 'max']:
            stack = stack[:-2] + [f"{next_val}({stack[-2]}, {stack[-1]})"]
        elif next_val == '?':
            stack = stack[:-3] + [f"({stack[-2]} if {stack[-3]} else {stack[-1]})"]
        elif next_val == "swap":
            stack = stack[:-2] + [stack[-1], stack[-2]]
        elif next_val.startswith("dup") and next_val[3:].isdecimal():
            elem = stack[-int(next_val[3:]) - 1]
            stack.append(elem)
        elif next_val.startswith("swap") and next_val[3:].isdecimal():
            index = -int(next_val[3:]) - 1
            elem = stack[index]
            stack = stack[:index] + [stack[-1]] + stack[index+1:-1] + [elem]
        elif next_val.islower(): # vars
            stack.append(next_val)
        else:
            try:
                val = float(next_val)
            except ValueError:
                raise ValueError(f"Unknown node in expr ({type(next_val)})")
            else:
                if val in [float("nan"), float("inf")]:
                    raise ValueError(f"Invalid value ({val})")
                else:
                    stack.append(next_val)

    if len(stack) == 1:
        return stack[0]
    else:
        raise ValueError("Stack unbalanced at end of expression. "
                         "Need to have exactly one value on the stack to return.")


class _ArithmeticExpr:
    def __init__(self, obj): 
        if arithmetic_expr:
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

        self._cached_clip = None # type: Optional[_VideoNode]

    def __getattr__(self, name):
        if hasattr(_vscore, name) or hasattr(self.clips[0], name):
            return getattr(self.compute(), name)
        else:
            raise AttributeError(f"{type(self).__name__!r} object has no attribute {name!r}")

    def __bool__(self):
        raise RuntimeError("Impossible")

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

        # eliminate duplicate clips
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

    def get_expr(self, namer) -> str:
        return ' '.join(namer(item) for item in self._expr).strip()

    @property
    def lut_func(self) -> Callable[..., numbers.Integral]:
        clips = self.clips

        assert len(clips) in [1, 2]

        func_impl = _postfix2infix(self.expr)
        func_impl = f"min(max(int({func_impl} + 0.5), 0), {(2 ** clips[0].format.bits_per_sample) - 1})" # clamp

        if len(clips) == 1:
            lut_str = f"lambda x: {func_impl}"
        else: # len(clips) == 2
            lut_str = f"lambda x, y: {func_impl}"

        class _LambdaFunction:
            def __init__(self, func_str: str):
                self.func = eval(func_str, {"exp": math.exp, "log": math.log, "sqrt": math.sqrt})
                self.func_str = func_str

            def __call__(self, *args, **kwargs):
                return self.func(*args, **kwargs)

            def __repr__(self):
                return self.func_str

        return _LambdaFunction(lut_str)

    def _operate(self, op: str, *operands, position=0) -> '_ArithmeticExpr':
        # accepted operands: Union[numbers.Real, _VideoNode, _ArithmeticExpr]

        def expr_to_element(expr):
            if expr._cached_clip is not None:
                return (expr._cached_clip,)
            else:
                return expr._expr

        def to_element(obj):
            if isinstance(obj, numbers.Real):
                return (repr(float(obj)),)
            elif isinstance(obj, _VideoNode):
                return (obj,)
            elif isinstance(obj, _ArithmeticExpr):
                return expr_to_element(obj)
            else:
                raise TypeError(f"{type(self).__name__!r}: Unknown input ({type(obj)})")

        expr = [to_element(operand) for operand in operands]
        expr.insert(position, expr_to_element(self))

        # "X X *" -> "X dup *"
        # _expr[i] == self._expr cannot be used since == is overloaded to return non-boolean value
        for i in range(position + 1, len(expr)):
            if (
                len(expr[i]) == len(self._expr) and 
                all((hash(x) == hash(y)) for x, y in zip(expr[i], self._expr))
            ):
                expr[i] = ("dup",)
            else:
                break

        expr.append((op,))

        return type(self)(tuple(itertools.chain(*expr)))

    def compute(self, planes=None, bits=None, use_lut=None) -> '_VideoNode':
        if arithmetic_expr:
            cacheable = planes is None and bits is None and use_lut is None

            if cacheable and self._cached_clip is not None:
                return self._cached_clip

            if self.expr in ['', 'x']: # empty expr
                return _VideoNode(self.clips[0]._node)
            else:
                clips = self.clips

                if bits is None:
                    not_equal_bits = (
                        lambda clip1, clip2: 
                            clip1.format.bits_per_sample != clip2.format.bits_per_sample)

                    if len(clips) >= 2 and any(not_equal_bits(clips[0], clip) for clip in clips[1:]):
                        raise ValueError('"bits" must be specified.')
                    else:
                        bits = clips[0].format.bits_per_sample

                is_int = lambda clip: clip.format.sample_type == vs.INTEGER
                get_bits = lambda clip: clip.format.bits_per_sample
                lut_available = (
                    lambda clips: 
                        len(clips) <= 2 and all(map(is_int, clips)) and sum(map(get_bits, clips)) <= 20)

                if use_lut is None:
                    use_lut = lut_available(clips) and len(self._expr) >= 15
                elif use_lut and not lut_available(clips):
                    raise ValueError("Lut computation is not available")

                # process
                if use_lut: # std.Lut() / std.Lut2()
                    if len(clips) == 1:
                        return core.std.Lut(clips[0], planes=planes, bits=bits, function=self.lut_func)
                    else: # len(clips) == 2
                        return core.std.Lut2(
                            clips[0], clips[1], planes=planes, bits=bits, function=self.lut_func)

                else: # std.Expr()
                    if planes is None:
                        expr = self.expr
                    else:
                        if isinstance(planes, int):
                            planes = [planes]

                        expr = [
                            (self.expr if i in planes else "") 
                            for i in range(clips[0].format.num_planes)]

                    in_format = clips[0].format

                    if bits == in_format.bits_per_sample:
                        out_format = None
                    else:
                        out_format = core.register_format(
                            color_family=in_format.color_family, 
                            sample_type=vs.INTEGER if bits <= 16 else vs.FLOAT, 
                            bits_per_sample=bits, 
                            subsampling_w=in_format.subsampling_w, 
                            subsampling_h=in_format.subsampling_h
                        )

                    res = core.std.Expr(clips=clips, expr=expr, format=out_format)

                    if cacheable:
                        self._cached_clip = res

                    return res

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

    def __invert__(self):
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
                    for recorder in Recorder._live_recorders:
                        if recorder.is_recording:
                            args_str = ', '.join(args)
                            kwargs_str = ', '.join(f"{k}={_repr(v)}" for k, v in kwargs)
                            call_str = ', '.join(s for s in [args_str, kwargs_str] if s != '')
                            recorder.buffer.append(f"{_repr(self)}.{name}({call_str})\n")

                    return attr(*args, **kwargs)

                return closure

            else:
                return attr

    def __len__(self):
        return self.num_frames

    def __str__(self):
        return "muvs " + str(self._node)

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

    methods = locals().copy()

    create_method = (lambda name: 
                          lambda self, *args: 
                              getattr(_ArithmeticExpr(self), name)(*args))

    magic_methods = [
        "__neg__", "__pos__", "__abs__", "__exp__", "__log__", "__invert__", "__sqrt__", "__lt__", 
        "__le__", "__eq__", "__ne__", "__gt__", "__ge__", "__add__", "__radd__", "__sub__", 
        "__rsub__", "__mul__", "__rmul__", "__truediv__", "__rtruediv__", "__pow__", "__rpow__", 
        "__and__", "__rand__", "__xor__", "__rxor__", "__or__", "__ror__", "__min__", "__rmin__", 
        "__max__", "__rmax__", "__conditional__", "__rconditional__", "__rrconditional__"
    ]

    methods.update((name, create_method(name)) for name in magic_methods)

    return type("_VideoNode", (), methods)

_VideoNode = _build_VideoNode()


def Expr(exprs, format=None) -> '_VideoNode':
    if isinstance(exprs, _VideoNode):
        exprs = [_ArithmeticExpr(exprs)]
    elif isinstance(exprs, _ArithmeticExpr):
        exprs = [exprs]
    elif isinstance(exprs, collections.abc.Sequence):
        if len(exprs) == 0:
            raise ValueError("Empty expression")

        for i in range(len(exprs)):
            if isinstance(exprs[i], _VideoNode):
                exprs[i] = _ArithmeticExpr(exprs[i])
            elif exprs[i] is not None and not isinstance(exprs[i], (_ArithmeticExpr, numbers.Real)):
                raise TypeError(f"Invalid type ({type(exprs[i])})")

    for expr in exprs:
        if isinstance(expr, _ArithmeticExpr):
            num_planes = expr.clips[0].format.num_planes

            for i in range(len(exprs), num_planes):
                exprs.append(exprs[-1])

            break
    else:
        raise ValueError("No clip is given")

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

    expr_strs = []
    for i in range(num_planes):
        if exprs[i] is None:
            expr_strs.append("")
        elif isinstance(exprs[i], numbers.Real):
            expr_strs.append(str(exprs[i]))
        else:
            expr_str = exprs[i].get_expr(_to_var)

            if expr_str == 'x':
                expr_strs.append('')
            else:
                expr_strs.append(expr_str)

    clips = (
        tuple(OrderedDict((obj, None) for obj in itertools.chain.from_iterable(
            expr.clips for expr in exprs 
            if isinstance(expr, _ArithmeticExpr)
        )).keys()))

    return core.std.Expr(clips, expr_strs, format)



# custom operations
Abs = abs

def Exp(x):
    if isinstance(x, (_ArithmeticExpr, _VideoNode)):
        return x.__exp__()
    else:
        return math.exp(x)


def Not(x):
    if isinstance(x, (_ArithmeticExpr, _VideoNode)):
        return x.__invert__()
    else:
        return not x


def And(x, y):
    if isinstance(x, (_ArithmeticExpr, _VideoNode)):
        return x.__and__(y)
    elif isinstance(y, (_ArithmeticExpr, _VideoNode)):
        return y.__rand__(x)
    else:
        return x and y


def Or(x, y):
    if isinstance(x, (_ArithmeticExpr, _VideoNode)):
        return x.__or__(y)
    elif isinstance(y, (_ArithmeticExpr, _VideoNode)):
        return y.__ror__(x)
    else:
        return x or y


def Xor(x, y):
    if isinstance(x, (_ArithmeticExpr, _VideoNode)):
        return x.__xor__(y)
    elif isinstance(y, (_ArithmeticExpr, _VideoNode)):
        return y.__rxor__(x)
    else:
        return (x and not y) or (not x and y)


def Log(x):
    if isinstance(x, (_ArithmeticExpr, _VideoNode)):
        return x.__log__()
    else:
        return math.log(x)


def Sqrt(x):
    if isinstance(x, (_ArithmeticExpr, _VideoNode)):
        return x.__sqrt__()
    else:
        return math.sqrt(x)


def Min(x, y):
    if isinstance(x, (_ArithmeticExpr, _VideoNode)):
        return x.__min__(y)
    elif isinstance(y, (_ArithmeticExpr, _VideoNode)):
        return y.__rmin__(x)
    else:
        return min(x, y)


def Max(x, y):
    if isinstance(x, (_ArithmeticExpr, _VideoNode)):
        return x.__max__(y)
    elif isinstance(y, (_ArithmeticExpr, _VideoNode)):
        return y.__rmax__(x)
    else:
        return max(x, y)


def Conditional(condition, condition_if_true, condition_if_false):
    try:
        return condition_if_true if condition else condition_if_false
    except TypeError:
        if isinstance(condition, (_ArithmeticExpr, _VideoNode)):
            return condition.__conditional__(condition_if_true, condition_if_false)
        elif isinstance(condition_if_true, (_ArithmeticExpr, _VideoNode)):
            return condition_if_true.__rconditional__(condition, condition_if_false)
        elif isinstance(condition_if_false, (_ArithmeticExpr, _VideoNode)):
            return condition_if_false.__rrconditional__(condition, condition_if_true)
        else:
            raise TypeError(f"'Conditional': Unknown input ({type(condition)}, "
                            f"{type(condition_if_true)}, {type(condition_if_false)})")
