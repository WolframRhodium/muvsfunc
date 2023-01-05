"""
An interface to VapourSynth

*** DO NOT PUBLISH MODULES THAT DEPEND ON THIS ***

objects:
    core (resembles vapoursynth.core)

functions:
    pollute (poisons for foreign modules)
    expr (switch for arithmetic expression)
    Expr (resembles core.std.Expr(), but with infix expression)
    record (computational graph recorder, resembles open())
    Recorder (base class for recorder)

functions for arithmetic expression:
    Abs, Exp, Not, And, Or, Xor, Log, 
    Sqrt, Min, Max, Conditional

"""

from abc import ABC, abstractmethod, abstractstaticmethod
from collections import OrderedDict
import collections.abc
from contextlib import contextmanager
import functools
import inspect
import itertools
import math
import numbers
import operator as op
from typing import Callable, Dict, List, MutableMapping, MutableSet
from typing import Optional, Sequence, Union
import weakref

import vapoursynth as vs
from vapoursynth import core as _vscore


__all__ = [
    "core", "expr", "pollute", "Expr", "record", "Recorder", 
    "Abs", "Exp", "Not", "And", "Or", "Xor", "Log", "Sqrt", 
    "Min", "Max", "Conditional"]


_is_api4: bool = hasattr(vs, "__api_version__") and vs.__api_version__.api_major == 4

class _Core:
    def __init__(self):
        self._registered_funcs = {} # type: Dict[str, Callable[..., '_VideoNode']]

    def __setattr__(self, name, value):
        if name in ["num_threads", "max_cache_size"]:
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

        elif isinstance(obj, (vs.VideoFormat if _is_api4 else vs.Format)):
            arg_str = ', '.join(f"{k}={closure(v)}" for k, v in obj._as_dict().items())
            return f"core.query_video_format({arg_str})" if _is_api4 else f"core.register_format({arg_str})"

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
                elif isinstance(output, list) and len(output) > 0 and isinstance(output[0], vs.VideoNode):
                    for item in output:
                        _ = _repr(item, default_prefix="clip") # register output
                    
                    for recorder in Recorder._live_recorders:
                        if recorder.is_recording:
                            recorder.buffer.append(self._get_str(func, args, kwargs, output, check_output=False) + '\n')

                    return list(_VideoNode(item) for item in output)
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
    def _get_str(func: vs.Function, args, kwargs, output, check_output=True):
        output_str = ""

        if check_output:
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


########################## Expr IR Start ##########################
class ExprIR(ABC):
    """ AST-style expression """

    @abstractmethod
    def __eq__(self, other):
        pass

    @abstractmethod
    def __repr__(self):
        """ Infix and function call style """
        pass

    @abstractmethod
    def __str__(self):
        """ Postfix style """
        pass

class DupN(ExprIR):
    def __eq__(self, other):
        return isinstance(other, DupN)
        
    def __repr__(self):
        return "DupN()"

    def __str__(self):
        return "dup"
dup = DupN()

class UnaryBaseOp(ExprIR):
    @abstractstaticmethod
    def cast(x):
        pass

    def __init__(self, x):
        self.x = self.cast(x)

    def __eq__(self, other):
        return isinstance(other, type(self)) and self.x == other.x

    def __repr__(self):
        return f"{type(self).__name__}({self.x!r})"

    def __str__(self):
        return f"{self.x!s} {type(self).__name__.lower()}"

class ConstantN(UnaryBaseOp):
    def __str__(self):
        return f"{self.x!s}"

    @staticmethod
    def cast(x):
        assert isinstance(x, numbers.Real)
        return x
ConstantN_0 = ConstantN(0)
ConstantN_1 = ConstantN(1)

class VarN(UnaryBaseOp):
    def __eq__(self, other):
        return isinstance(other, VarN) and hash(self.x) == hash(other.x)

    def __str__(self):
        return f"{self.x!s}"

    @staticmethod
    def cast(x):
        assert isinstance(x, _VideoNode)
        return x

def Cast(x):
    if isinstance(x, ExprIR):
        return x
    elif isinstance(x, numbers.Real):
        return ConstantN(x)
    elif isinstance(x, _VideoNode):
        return VarN(x)
    elif isinstance(x, vs.VideoNode):
        return VarN(_VideoNode(x))
    else:
        raise TypeError(f"Unkonwn input ({type(x)})")

class UnaryOp(UnaryBaseOp):
    @abstractstaticmethod
    def compute(x):
        pass

    def __str__(self):
        return f"{self.x!s} {self.op_name}"

    @staticmethod
    def cast(x):
        return Cast(x)

class NotN(UnaryOp):
    op_name = "not"
    compute = op.not_

class AbsN(UnaryOp):
    op_name = "abs"
    compute = abs

class SqrtN(UnaryOp):
    op_name = "sqrt"
    compute = math.sqrt

class LogN(UnaryOp):
    op_name = "log"
    compute = math.log

class ExpN(UnaryOp):
    op_name = "exp"
    compute = math.exp

class BinaryOp(ExprIR):
    @abstractstaticmethod
    def compute(x, y):
        pass

    def __init__(self, x, y):
        self.x, self.y = self.cast(x, y)

    def __eq__(self, other):
        return (
            isinstance(other, type(self)) and 
            self.x == other.x and 
            self.y == other.y
        )

    def __repr__(self):
        return f"{type(self).__name__}({self.x!r}, {self.y!r})"

    def __str__(self):
        return f"{self.x!s} {self.y!s} {self.op_name}"

    @staticmethod
    def cast(x, y):
        return Cast(x), Cast(y)

class AddN(BinaryOp):
    op_name = "+"
    compute = op.add

class SubN(BinaryOp):
    op_name = "-"
    compute = op.sub

class MulN(BinaryOp):
    op_name = "*"
    compute = op.mul

class DivN(BinaryOp):
    op_name = "/"
    compute = op.truediv

class PowN(BinaryOp):
    op_name = "pow"
    compute = op.pow

class AndN(BinaryOp):
    op_name = "and"
    compute = op.and_

class OrN(BinaryOp):
    op_name = "or"
    compute = op.or_

class XorN(BinaryOp):
    op_name = "xor"
    compute = op.xor

class LtN(BinaryOp):
    op_name = "<"
    compute = op.lt

class LeN(BinaryOp):
    op_name = "<="
    compute = op.le

class EqN(BinaryOp):
    op_name = "="
    compute = op.eq

class NeN(BinaryOp):
    op_name = "= not"
    compute = op.ne

class GeN(BinaryOp):
    op_name = ">="
    compute = op.ge

class GtN(BinaryOp):
    op_name = ">"
    compute = op.gt

class MaxN(BinaryOp):
    op_name = "max"
    compute = max

class MinN(BinaryOp):
    op_name = "min"
    compute = min

class ConditionalN(ExprIR):
    def __init__(self, x, y, z):
        self.x, self.y, self.z = self.cast(x, y, z)

    def __eq__(self, other):
        return (
            isinstance(other, ConditionalN) and 
            self.x == other.x and 
            self.y == other.y and 
            self.z == other.z
        )

    def __repr__(self):
        return f"ConditionalN({self.x!r}, {self.y!r}, {self.z!r})"

    def __str__(self):
        return f"{self.x!s} {self.y!s} {self.z!s} ?"

    @staticmethod
    def cast(x, y, z):
        return Cast(x), Cast(y), Cast(z)

    @staticmethod
    def compute(x, y, z):
        return y if x else z

def _simplify(expr: ExprIR) -> ExprIR:
    assert isinstance(expr, ExprIR)

    while True:
        prev_expr = expr

        # early skipping
        if isinstance(expr, (DupN, ConstantN, VarN)):
            return expr
        # constant foldings and universal eliminations
        elif isinstance(expr, UnaryOp) and isinstance(expr.x, ConstantN):
            # num op -> op(num)
            return ConstantN(expr.compute(expr.x.x))
        elif isinstance(expr, BinaryOp):
            if isinstance(expr.x, ConstantN):
                if isinstance(expr.y, ConstantN):
                    # num1 num2 op -> op(num1, num2)
                    return ConstantN(expr.compute(expr.x.x, expr.y.x))
                elif expr.y == DupN:
                    # num dup op -> op(num, num)
                    return ConstantN(expr.compute(expr.x.x, expr.x.x))
            elif expr.x == expr.y:
                # x x op -> x dup op
                expr = type(expr)(expr.x, dup)

        # operator specific simplifications
        if isinstance(expr, SqrtN):
            x = _simplify(expr.x)
            if isinstance(x, MulN) and isinstance(x.y, DupN):
                # x dup * sqrt -> x abs
                expr = AbsN(x.x)
            else:
                expr = SqrtN(x)
        elif isinstance(expr, LogN):
            x = _simplify(expr.x)
            if isinstance(x, ExpN):
                # x exp log -> x
                expr = x.x
            else:
                expr = LogN(x)
        elif isinstance(expr, ExpN):
            x = _simplify(expr.x)
            if isinstance(x, LogN):
                # x log exp -> x
                expr = x.x
            else:
                expr = ExpN(x)
        elif isinstance(expr, AddN):
            if expr.x == ConstantN_0:
                # 0 x + -> x
                expr = expr.y
            elif expr.y == ConstantN_0:
                # x 0 + -> x
                expr = expr.x
        elif isinstance(expr, SubN):
            if isinstance(expr.y, DupN):
                # x dup - -> 0
                return ConstantN_0
            elif expr.y == ConstantN_0:
                # x 0 - -> x
                expr = expr.x
        elif isinstance(expr, MulN):
            if expr.x == ConstantN_1:
                # 1 x * -> x
                expr = expr.y
            elif expr.y == ConstantN_1:
                # x 1 * -> x
                expr = expr.x
        elif isinstance(expr, DivN):
            if isinstance(expr.y, DupN):
                # x dup / -> 1
                return ConstantN_1
            elif expr.y == ConstantN_1:
                # x 1 / -> x
                expr = expr.x
        elif isinstance(expr, PowN):
            if isinstance(expr.x, ConstantN):
                if expr.x == ConstantN_0:
                    # 0 x pow -> 0
                    expr = ConstantN_0
                elif expr.x == ConstantN_1:
                    # 1 x pow -> 1
                    expr = ConstantN_1
                elif expr.x == ConstantN(math.e):
                    # math.e x pow -> x exp
                    expr = ExpN(expr.y)
            elif isinstance(expr.y, ConstantN):
                if expr.y == ConstantN_0:
                    # x 0 pow -> 1
                    expr = ConstantN_1
                elif expr.y == ConstantN_1:
                    # x 1 pow -> x
                    expr = expr.x
                elif expr.y == ConstantN(2):
                    # x 2 pow -> x dup *
                    expr = MulN(expr.x, dup)
                elif expr.y == ConstantN(0.5):
                    # x 0.5 pow -> x sqrt
                    expr = SqrtN(expr.x)
                elif expr.y == ConstantN(-0.5):
                    # x -0.5 pow -> x dup sqrt /
                    expr = DivN(expr.x, SqrtN(dup))
        elif isinstance(expr, (MaxN, MinN)) and isinstance(expr.y, DupN):
            # x dup {max/min} -> x
            expr = expr.x
        elif isinstance(expr, ConditionalN):
            if isinstance(expr.x, ConstantN):
                # num x y ? -> (num ? x : y)
                expr = ConditionalN.compute(expr.x, expr.y, expr.z)
            elif expr.y == expr.z:
                # _ x x ? -> x
                expr = expr.y

        # non-local simplification of binary operations
        if isinstance(expr, BinaryOp):
            expr = type(expr)(_simplify(expr.x), _simplify(expr.y))

            if isinstance(expr.x, ConstantN):
                if isinstance(expr.y, UnaryOp):
                    if isinstance(expr.y.x, DupN):
                        # num dup op1 op2 -> num num op1 op2
                        expr = type(expr)(expr.x, type(expr.y)(expr.x))
                elif isinstance(expr.y, BinaryOp):
                    if isinstance(expr.y.x, DupN):
                        # num dup x op1 op2 -> num num1 x op1 op2
                        expr = type(expr)(expr.x, type(expr.y)(expr.x, expr.y.y))
            elif isinstance(expr.y, BinaryOp) and expr.x == expr.y.x:
                # x x y op1 op2 -> x dup y op1 op2
                expr = type(expr)(expr.x, type(expr.y)(dup, expr.y.y))

        if expr == prev_expr:
            # no progress
            return expr
        else:
            prev_expr = expr
            # continue

def postfix(expr: ExprIR, namer: Optional[Callable[[VarN], str]] = None) -> str:
    assert isinstance(expr, ExprIR)

    if isinstance(expr, ConstantN):
        return str(expr)
    elif isinstance(expr, VarN):
        if namer is None:
            return str(expr)
        else:
            return namer(expr)
    elif isinstance(expr, DupN):
        return "dup"
    elif isinstance(expr, UnaryOp):
        return f"{postfix(expr.x, namer)} {expr.op_name}"
    elif isinstance(expr, BinaryOp):
        first = postfix(expr.x, namer)
        return f"{first} {postfix(expr.y, namer)} {expr.op_name}"
    elif isinstance(expr, ConditionalN):
        first = postfix(expr.x, namer)
        second = postfix(expr.y, namer)
        return f"{first} {second} {postfix(expr.z, namer)} ?"
    else:
        raise TypeError(f"Unknwon type {type(expr)}")


def infix(expr: ExprIR, namer: Optional[Callable[[VarN], str]] = None, 
    top: Optional[str] = None
) -> str:
    assert isinstance(expr, ExprIR)

    if isinstance(expr, ConstantN):
        return str(expr)
    elif isinstance(expr, VarN):
        if namer is None:
            return str(expr)
        else:
            return namer(expr)
    elif isinstance(expr, DupN):
        if top:
            return top
        else:
            raise ValueError("Empty dup node")
    elif isinstance(expr, UnaryOp):
        return f"{expr.op_name}({infix(expr.x, namer, top=top)})"
    elif isinstance(expr, BinaryOp):
        first = infix(expr.x, namer, top=top)
        return f"({first} {expr.op_name} {infix(expr.y, namer, top=first)})"
    elif isinstance(expr, ConditionalN):
        first = infix(expr.x, namer, top=top)
        second = infix(expr.y, namer, top=first)
        return f"({second} if {first} else {infix(expr.z, namer, top=second)})"
    else:
        raise TypeError(f"Unknwon type {type(expr)}")

########################## Expr IR End ##########################

def namer_factory():
    alphabet = "xyzabcdefghijklmnopqrstuvw"
    mapping = OrderedDict() # type: MutableMapping[_VideoNode, str]

    def namer(obj: VarN) -> str:
        x = obj.x
        if x in mapping or len(mapping) < len(alphabet):
            return mapping.setdefault(x, f"{alphabet[len(mapping)]}")
        else:
            raise RuntimeError(f"{type(self).__name__!r}: Too many nodes")

    return namer


class _Fake_VideoNode:
    """ Fake VideoNode used to bypass instance check in other scripts """
    pass


class _ArithmeticExpr(_Fake_VideoNode):
    def __init__(self, obj): 
        self._expr = Cast(obj) # type: ExprIR
        self._cached_clip = None # type: Optional[_VideoNode]

    def __getattr__(self, name):
        if hasattr(_vscore, name) or hasattr(self.clips[0], name):
            return getattr(self.compute(), name)
        else:
            raise AttributeError(f"{type(self).__name__!r} object has no attribute {name!r}")

    def __bool__(self):
        raise RuntimeError("Impossible")

    def __hash__(self):
        return hash(self.clips + (self.expr,))

    def __str__(self):
        def namer(x: VarN):
            return _repr(x.x)
        return infix(self._expr, namer=namer).strip("()")

    @property
    def clips(self):
        from collections import OrderedDict

        clips_dict = OrderedDict()
        exprs = [self._expr]

        while exprs:
            expr = exprs.pop()
            if isinstance(expr, VarN):
                clips_dict.setdefault(expr.x, None)
            elif isinstance(expr, UnaryOp):
                exprs.append(expr.x)
            elif isinstance(expr, BinaryOp):
                exprs.extend([expr.y, expr.x])
            elif isinstance(expr, ConditionalN):
                exprs.extend([expr.z, expr.y, expr.x])

        return tuple(clips_dict.keys())

    def get_expr(self, namer) -> str:
        return postfix(self._expr, namer=namer)

    @property
    def expr(self) -> str:
        return self.get_expr(namer=namer_factory())

    @property
    def lut_func(self) -> Callable[..., numbers.Integral]:
        clips = self.clips

        assert len(clips) in [1, 2]

        func_impl = infix(self._expr, namer=namer_factory())
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

    def compute(self, planes=None, bits=None, use_lut=None, 
        simplify: Union[bool, Callable[[ExprIR], ExprIR]] = True
    ) -> '_VideoNode':

        if arithmetic_expr:
            cacheable = planes is None and bits is None and use_lut is None

            if cacheable and self._cached_clip is not None:
                return self._cached_clip

            if simplify:
                if callable(simplify):
                    self._expr = simplify(self._expr)
                else:
                    self._expr = _simplify(self._expr)

                if len(self.clips) == 0:
                    raise ValueError("ArithmeticExpr becomes empty")

            if self.expr in ['', 'x']: # empty expr
                return _VideoNode(self.clips[0]._node)
            else:
                clips = self.clips
                if len(clips) > 26:
                    raise RuntimeError("Too many clips.")

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
                    use_lut = lut_available(clips) and len(self.expr.split()) >= 15
                elif use_lut and not lut_available(clips):
                    raise ValueError("Lut computation is not available")

                # process
                if use_lut: # std.Lut() / std.Lut2()
                    if len(clips) == 1:
                        res = core.std.Lut(clips[0], planes=planes, bits=bits, function=self.lut_func)
                    else: # len(clips) == 2
                        res = core.std.Lut2(
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
                        query_video_format = core.query_video_format if _is_api4 else core.register_format
                        out_format = query_video_format(
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
    def _operate(self, 
        op: Union[UnaryOp, BinaryOp, ConditionalN], 
        *operands: Sequence[Union[numbers.Real, vs.VideoNode, "_VideoNode", ExprIR]]
    ) -> "_ArithmeticExpr":
        unwrap = lambda x: x._expr if isinstance(x, type(self)) else x
        result = op(*map(unwrap, operands))
        return type(self)(result)

    # unary operations
    def __neg__(self):
        return self._operate(SubN, 0, self)

    def __pos__(self):
        return self

    def __abs__(self):
        return self._operate(AbsN, self)

    def __invert__(self):
        return self._operate(NotN, self)

    def __exp__(self):
        return self._operate(ExpN, self)

    def __log__(self):
        return self._operate(LogN, self)

    def __sqrt__(self):
        return self._operate(SqrtN, self)

    # binary operations
    def __lt__(self, other):
        return self._operate(LtN, self, other)

    def __le__(self, other):
        return self._operate(LeN, self, other)

    def __eq__(self, other):
        return self._operate(EqN, self, other)

    def __ne__(self, other):
        return self._operate(NeN, self, other)

    def __gt__(self, other):
        return self._operate(GtN, self, other)

    def __ge__(self, other):
        return self._operate(GeN, self, other)

    def __add__(self, other):
        return self._operate(AddN, self, other)

    def __radd__(self, other):
        return self._operate(AddN, other, self)

    def __sub__(self, other):
        return self._operate(SubN, self, other)

    def __rsub__(self, other):
        return self._operate(SubN, other, self)

    def __mul__(self, other):
        return self._operate(MulN, self, other)

    def __rmul__(self, other):
        return self._operate(MulN, other, self)

    def __truediv__(self, other):
        return self._operate(DivN, self, other)

    def __rtruediv__(self, other):
        return self._operate(DivN, other, self)

    def __pow__(self, other, module=None):
        if module is None:
            return self._operate(PowN, self, other)
        else:
            raise NotImplemented

    def __rpow__(self, other):
        return self._operate(PowN, other, self)

    def __and__(self, other):
        return self._operate(AndN, self, other)

    def __rand__(self, other):
        return self._operate(AndN, other, self)

    def __or__(self, other):
        return self._operate(OrN, self, other)

    def __ror__(self, other):
        return self._operate(OrN, other, self)

    def __xor__(self, other):
        return self._operate(XorN, self, other)

    def __rxor__(self, other):
        return self._operate(XorN, other, self)

    # custom binary operations
    def __max__(self, other):
        return self._operate(MaxN, self, other)

    def __rmax__(self, other):
        return self._operate(MaxN, other, self)

    def __min__(self, other):
        return self._operate(MinN, self, other)

    def __rmin__(self, other):
        return self._operate(MinN, other, self)

    # custom ternary operation
    def __conditional__(self, other_true, other_false):
        return self._operate(ConditionalN, self, other_true, other_false)

    def __rconditional__(self, other_condition, other_false):
        return self._operate(ConditionalN, other_true, self, other_false)

    def __rrconditional__(self, other_condition, other_true):
        return self._operate(ConditionalN, other_true, other_false, self)


def _build_VideoNode(fake_vn=None):
    _plane_idx_mapping = {
        vs.YUV: {'Y': 0, 'U': 1, 'V': 2}, 
        vs.RGB: {'R': 0, 'G': 1, 'B': 2}, 
        vs.GRAY: {'GRAY': 0, 'Y': 0}
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
                            args_str = ', '.join(map(_repr, args))
                            kwargs_str = ', '.join(f"{k}={_repr(v)}" for k, v in kwargs.items())
                            call_str = ', '.join(s for s in [args_str, kwargs_str] if s != '')
                            recorder.buffer.append(f"{_repr(self)}.{name}({call_str})\n")

                    return attr(*args, **kwargs)

                return closure

            else:
                return attr

    def __len__(self):
        return self.num_frames

    def __str__(self):
        return f"muvs {self._node!s}"

    def __bool__(self):
        raise RuntimeError("Impossible")

    def __dir__(self):
        return dir(self._node) + list(_plane_idx_mapping[self.format.color_family].keys())

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

    return type("_VideoNode", (fake_vn,) if fake_vn is not None else (), methods)

_VideoNode = _build_VideoNode(_Fake_VideoNode)


def Expr(exprs, format=None, 
    simplify: Union[bool, Callable[[ExprIR], ExprIR]] = True
) -> '_VideoNode':
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

    if simplify:
        for i in range(len(exprs)):
            if isinstance(exprs[i], _ArithmeticExpr):
                if callable(simplify):
                    exprs[i] = _ArithmeticExpr(simplify(exprs[i]._expr))
                else:
                    exprs[i] = _ArithmeticExpr(_simplify(exprs[i]._expr))

    for expr in exprs:
        if isinstance(expr, _ArithmeticExpr):
            num_planes = expr.clips[0].format.num_planes

            for i in range(len(exprs), num_planes):
                exprs.append(exprs[-1])

            break
    else:
        raise ValueError("No clip is given")

    namer = namer_factory()

    expr_strs = []
    for i in range(num_planes):
        if exprs[i] is None:
            expr_strs.append("")
        elif isinstance(exprs[i], numbers.Real):
            expr_strs.append(str(exprs[i]))
        else:
            expr_str = exprs[i].get_expr(namer=namer)

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
    except RuntimeError:
        if isinstance(condition, (_ArithmeticExpr, _VideoNode)):
            return condition.__conditional__(condition_if_true, condition_if_false)
        elif isinstance(condition_if_true, (_ArithmeticExpr, _VideoNode)):
            return condition_if_true.__rconditional__(condition, condition_if_false)
        elif isinstance(condition_if_false, (_ArithmeticExpr, _VideoNode)):
            return condition_if_false.__rrconditional__(condition, condition_if_true)
        else:
            raise TypeError(f"'Conditional': Unknown input ({type(condition)}, "
                            f"{type(condition_if_true)}, {type(condition_if_false)})")


def pollute(*modules):
    class _FakeVS:
        def __init__(self):
            self.VideoNode = _Fake_VideoNode
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
