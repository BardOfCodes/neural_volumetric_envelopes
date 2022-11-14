# Installation

1. Install [conda](https://conda.io/projects/conda/en/latest/user-guide/install/index.html).

2. create new environment.

    ```bash
    source install_conda_env.sh
    ```

3. Install `nve_dev`.

    ```bash
    conda activate nve_env
    python setup.py install
    ```

4. Please add instructions for any other package you install here.

# To update packages

Install what you need.

For pip installed packages, after installation use:

```bash
pip freeze > requirements.txt
```

For conda installed packages, add the command to `install_conda_env.sh` or lets figure out a better way.

For pip update:

```bash
pip install -r requirements.txt
```