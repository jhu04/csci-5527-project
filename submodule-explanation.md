### Setup
The submodule at `PyGRANSO/` points to the main PyGRANSO repo. 
After cloning this repo/checkout out this branch for the first time, run
```
git submodule update --init --recursive
```
to set up the submodule.

Additionally, set the git option
```
git config submodule.recurse true
```
to avoid possible weirdness when switching between branches with different versions of the submodule (or that don't have it at all).
(otherwise you may need to provide the `--recurse-submodules` flag to `git checkout` and `git switch`)

### Migration
Add the `PyGRANSO` directory to your python interpreter path at the top of your scripts:
```python
import sys
import pathlib
pygranso_path = pathlib.Path("./PyGRANSO/").resolve()
sys.path.insert(0, str(pygranso_path))
```

