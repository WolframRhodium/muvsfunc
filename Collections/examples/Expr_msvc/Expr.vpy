"""
this project is inspired by https://github.com/Endilll/exprcpp
it dynamically generates code for a plugin (Expr in this case)

** the code here is badly written, don't use it**
"""


import os
import tempfile
from string import Template

import vapoursynth as vs
from vapoursynth import core


def compile_plugin(filenames, vs_include_dir):
    from distutils.msvccompiler import MSVCCompiler
    msvc = MSVCCompiler()

    msvc.add_include_dir(vs_include_dir)
    
    tempdir=tempfile.gettempdir()

    msvc.compile(filenames, extra_postargs=["/O2", "/EHsc"], output_dir=tempdir)
    msvc.link_shared_lib([os.path.join(tempdir, "Source.obj")], output_libname="expr", output_dir=tempdir)
    core.std.LoadPlugin(os.path.join(tempdir, "expr.dll"))


def Expr(clips, func_impl, func_name, planes=None, vs_include_dir=r"D:\VapourSynth\sdk\include\vapoursynth"):
    # assertions
    for clip in clips[1:]:
        assert (
            clip.format.id == clips[0].format.id and 
            clip.width == clips[0].width and 
            clip.height == clips[0].height and 
            clip.num_frames == clips[0].num_frames
        )

    # completes code from template
    num_inputs = len(clips)

    clip = clips[0]
    if clip.format.sample_type == vs.FLOAT:
        t = "float"
    elif clip.format.bits_per_sample == 8:
        t = "uint8_t"
    else:
        t = "uint16_t"
    inputs = ", ".join(f"srcp[{i}][x]" for i in range(num_inputs))

    if planes is None:
        planes = list(range(clip.format.num_planes))
    planes = "{" + ", ".join(("1" if i in planes else "0") for i in range(3)) + "}"
    
    with open("Source_template.cpp", 'r') as f:
        code_template = f.read()

    code = (
        Template(code_template)
        .substitute(
            num_inputs=num_inputs, t=t, inputs=inputs, func_name=func_name, 
            planes=planes, func_impl=func_impl)
    )

    # generates final source code
    with open("Source.cpp", 'w') as f:
        f.write(code)

    compile_plugin(("Source.cpp",), vs_include_dir)

    return core.expr.Expr(clips)


# test clips
src1 = core.std.BlankClip(format=vs.YUV420P8, color=[0, 13, 29])
src2 = core.std.BlankClip(format=vs.YUV420P8, color=[93, 128, 247])


# usage
func_impl = """
T add(T x, T y) {
    return x + y - (T) 128;
}
"""

res = Expr([src1, src2], func_impl, func_name="add", planes=[0, 2])

res.set_output()
