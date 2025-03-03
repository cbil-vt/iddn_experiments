# 5 ChIP-seq data for monocytes, and 4 for T cells, from Cistrome

```
41287	STAT1	classical monocytes
41301	IRF1	classical monocytes
45444	CTCF	classical monocytes
81223	RUNX1	classical monocytes  # only one in both cell types
85986	SPI1	classical monocytes  # quite weak scores, all < 3
44092	ETS1	naive CD4 T cells
44097	RUNX1	naive CD4 T cells
44098	FOXP3	naive CD4 T cells
47435	REST	naive CD4 T cells
```

# get network from sparse output

```python
import numpy as np
from scipy.sparse import coo_array

net_out = np.zeros((2, n_gene, n_gene), dtype=np.int8)
for n in range(n_gene):
    for k in range(2):
        x: coo_array = out_iddn[n][k]
        idx = x.coords[0]
        dat = x.data
        idx = idx[np.abs(dat)>1e-4]
        net_out[k, idx, n] = 1
        net_out[k, n, idx] = 1
```

# ATAC

```python
atac = mu.read("../../iddn_data/muon/data/pbmc10k.h5mu/atac")
```

