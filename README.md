# Ponderosa-beta

### Plotting IBD segments

```
from pedigree_tools import Karyogram

map_files = [f"chr_{chrom} for chrom in range(1, 23)]

kgram = Karyogram(map_files, cm = True)

segments = [[1, 32.1, 45.6, 0], [2, 45.5, 123.4, 1]]

kgram.plot_segments(segments, file_name = "my_karyogram")
```
