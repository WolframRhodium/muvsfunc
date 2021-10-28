# muvsfunc
Muonium's VapourSynth functions

## Dependencies
[VapourSynth](https://github.com/vapoursynth/vapoursynth) R39-R57

### Scripts
- [havsfunc](https://github.com/HomeOfVapourSynthEvolution/havsfunc/blob/master/havsfunc.py)

- [mvsfunc](https://github.com/AmusementClub/mvsfunc/blob/mod/mvsfunc.py)

- [nnedi3_resample](https://github.com/AmusementClub/nnedi3_resample)

and the dependencies of them.

### Plugins
- [AWarpSharp2](https://github.com/dubhater/vapoursynth-awarpsharp2)

- [Bilateral](https://github.com/HomeOfVapourSynthEvolution/VapourSynth-Bilateral)

- [CAS](https://github.com/HomeOfVapourSynthEvolution/VapourSynth-CAS)

- [CTMF](https://github.com/HomeOfVapourSynthEvolution/VapourSynth-CTMF)

- [descale](https://github.com/Irrational-Encoding-Wizardry/descale)

- [DFTTest](https://github.com/HomeOfVapourSynthEvolution/VapourSynth-DFTTest)

- [EEDI2](https://github.com/HomeOfVapourSynthEvolution/VapourSynth-EEDI2)

- [fmtconv](https://github.com/EleonoreMizo/fmtconv)

- [misc](https://github.com/vapoursynth/vs-miscfilters-obsolete) (required by VS R55 and later)

- [MVTools](https://github.com/dubhater/vapoursynth-mvtools)

- [nnedi3](https://github.com/dubhater/vapoursynth-nnedi3)

- [RemoveGrain](https://github.com/vapoursynth/vs-removegrain) (required by VS R55 and later)

- [SangNom](https://bitbucket.org/James1201/vapoursynth-sangnom)

- [TCanny](https://github.com/HomeOfVapourSynthEvolution/VapourSynth-TCanny)

- [TemporalMedian](https://github.com/dubhater/vapoursynth-temporalmedian)

- [VSFilter](https://github.com/HomeOfVapourSynthEvolution/VSFilter) (only required by `TextSub16()`)

- [VSFilterMod](https://github.com/sorayuki/VSFilterMod) (only required by `TextSub16()`)

- [vs_mxnet](https://github.com/kice/vs_mxnet) (only required by `super_resolution()`)

### Python Packages
- [matplotlib](https://github.com/matplotlib/matplotlib) (only required by `getnative()`)

- [MXNet](https://github.com/apache/incubator-mxnet) (only required by `super_resolution()`)

### Optional dependencies
- [Akarin's Expr](https://github.com/AkarinVS/vapoursynth-plugin) (performance optimizations)


## Files
`muvsfunc.py` is the main script. It contains some algorithms like `GradFun3`, `GuidedFilter`, `TextSub16`, some helper functions like `MergeChroma`, and some ideas that I develop like `LDMerge`, `AnimeMask`.

`muvsfunc_misc.py` is a complement of the previous script, containing some out-dated algorithms like `SSR`(Single-scale Retinex), some helper functions like `gauss`, `band_merge`, and also one of my idea named `detail_enhancement`. It may or may not be merged to the main script some day.

`muvsfunc_numpy.py` contains algorithms that are processed in `numpy.ndarray` rather than C/C++. Due to the low performance, they are mainly for research. Here is my current interest.

`LUM.py` and `SuperRes.py`(it's not the SuperRes in madVR or MPDN) are the dross of history. You won't need to use them.

## Resources

#### **_[OpenCV for VapourSynth](https://github.com/WolframRhodium/muvsfunc/wiki/OpenCV-Python-for-VapourSynth)_**

#### [muvs tutorial](https://github.com/WolframRhodium/muvsfunc/wiki/muvs-tutorial)
