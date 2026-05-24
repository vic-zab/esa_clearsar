# Python Environment
This project uses conda environment `esa_sar` on Windows with Miniconda.

**IMPORTANT**: The shell Claude uses cannot call `conda` — condabin only contains `.bat` files that don't work in bash. Always use the environment's Python executable directly:

  /c/Users/Victor/.conda/envs/esa_sar/python.exe script.py <br>
  /c/Users/Victor/.conda/envs/esa_sar/python.exe -m pip install \<package\>

Never use `conda activate`, never call `python` or `pip` directly.