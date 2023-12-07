# Ponderosa-beta

### Setting up the virtual environment and installing phasedibd

*I have tested this with Python 3.10.9. Earlier versions of Python may work, but phasedibd needs at least Python 3.8 to install.*

1. Create the virtual environment: `python -m venv ponderosa_venv`
2. Activate the virtual environment: `source ponderosa_venv/bin/activate`
3. Install required packages: `python -m pip install -r requirements.txt` (`requirements.txt` is found in this directory)
4. Clone `phasedibd`: `git clone https://github.com/23andMe/phasedibd`
5. `cd phasedibd`
6. `make`**
7. `python setup.py install`
8. `python tests/unit_tests.py`

**On the HPC that I use, I need to replace the third line of the phasedibd Makefile with `python -m pip install cython` (currently it reads `pip install cython --user`, which may cause permissions issues).

### Plotting IBD segments

```
from pedigree_tools import Karyogram

# either a list of .map files or a single map file that contains all chromosomes
map_files = [f"chr_{chrom} for chrom in range(1, 23)]

# initialize the object, set cm to True if you're plotting centimogran positions, cm = False plots Mb positions
kgram = Karyogram(map_files, cm = True)

# segments is a list of segments with the following info: [chromosome, start, stop, haplotype index (0 or 1)]
segments = [[1, 32.1, 45.6, 0], [2, 45.5, 123.4, 1]]

# plot segments; optional keyword arguments include
# file_name [default: "karyogram"]: writes the output as [file_name].png
# hap0_color [default: "cornflowerblue"]: the color of IBD segments on the 0 haplotype
# hap1_color [default: "tomato"]: the color of IBD segments on the 1 haplotype
# dpi [default: 500]: dpi of the plot

kgram.plot_segments(segments, file_name = "my_karyogram", hap0_color = "skyblue")
```
