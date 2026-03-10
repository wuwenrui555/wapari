# WapaRi

## Installation

```bash
env_name=wapari

# Remove existing environment if exists
conda deactivate && conda env remove -y -n $env_name

# Create and activate a new conda environment
mamba create -y -n $env_name python=3.11
mamba activate $env_name

pip install git+https://github.com/wuwenrui555/wapari.git@dev
```

## References

- [napari training course](https://github.com/sofroniewn/napari-training-course/tree/master/lessons)
