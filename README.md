# Repository Structure
- `data/` contains `sts-other.tsv` and new versions of the dataset are collected there.
- `models/` for any trained models to be saved in. You can download the all finetuned models [here](https://drive.google.com/drive/folders/1-REnJOs-XBVV1-e8vnK_u34z79mlvznW?usp=share_link).
- `readme.txt` information about the dataset provided by the creators.
- `analysis.ipynb` contains all code for the required data analysis.
  - Please be aware that the package `lazypredict` seems to have problems on MAC, so we'd recommend running the code on either colab or on a linux/ windows machine. Alternatively, we marked parts in the code where different models could be used as a workaround if ran on a MAC.