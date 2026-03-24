# ST-DiODE

PyTorch implementation of **ST-DiODE** for spatiotemporal sequence prediction.

This repository contains a cleaned and training-ready version of the project code, including:

- **ST-DiODE backbone**
- **training / validation / testing pipeline**
- **custom `.npy` sequence dataloader**
- **logging, checkpoint saving, and metric evaluation**

The current pipeline is directly adapted to the polished codebase and is especially convenient for **custom sequence data in `.npy` format** with shape:

```text
[N, T, C, H, W]
