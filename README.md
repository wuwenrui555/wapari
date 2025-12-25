# WapaRi

## Installation

```bash
env_name=wapari

# Remove existing environment if exists
conda deactivate && conda env remove -y -n $env_name

# Create and activate a new conda environment
mamba create -y -n $env_name python=3.10
mamba activate $env_name
```
