# muvsfunc
Muonium's VapourSynth functions

### Files
`muvsfunc.py` is the main script. It contains some algorithms like `GradFun3`, `GuidedFilter`, `TextSub16`, some helper functions like `MergeChroma`, and some ideas that I develop like `LDMerge`, `AnimeMask`.

`muvsfunc_misc.py` is a complement of the previous script, containing some out-dated algorithms like `SSR`(Single-scale Retinex), some helper functions like `gauss`, `band_merge`, and also one of my idea named `detail_enhancement`. It may or may not be merged to the main script some day.

`muvsfunc_numpy.py` contains algorithms that are processed in `numpy.ndarray` rather than C/C++. Due to the low performance, they are mainly for research. Here is my current interest.

`LUM.py` and `SuperRes.py`(it's not the SuperRes in madVR or MPDN) are the dross of history. You won't need to use them.

*Most functions are documented. But since I'm not a native English speaker, there might be some grammar errors. I'm happy to hear others point out mistakes to me.*
