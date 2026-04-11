# bnd : BeNeuroLab Data Organization

A **lightweight** collection of functions for managing the experimental data recorded in the 
BeNeuro Lab, and a CLI tool called `bnd` for easy access to this functionality.

Play around with it and raise Github issues if anything fails

# Setting up

## 1. Install `bnd`

### Option A — pipx (recommended)

[pipx](https://pipx.pypa.io) installs `bnd` in an isolated environment and makes the CLI available system-wide.

1. Install pipx if you don't have it:
   ```shell
   # Windows (requires Python ≥ 3.10)
   pip install pipx
   pipx ensurepath   # restart your terminal after this

   # Linux
   sudo apt install pipx
   pipx ensurepath
   ```

2. Install `bnd`:
   ```shell
   # Lightweight (upload, download, config only — fast install):
   pipx install "bnd @ git+https://github.com/BeNeuroLab/bnd.git"

   # Full install with processing dependencies (NWB, kilosort, pyaldata):
   pipx install "bnd[processing] @ git+https://github.com/BeNeuroLab/bnd.git"
   ```
   To install a specific branch (e.g. for testing):
   ```shell
   pipx install "bnd[processing] @ git+https://github.com/BeNeuroLab/bnd.git@seperate-ks-env"
   ```

3. Verify:
   ```shell
   bnd --help
   ```

To **update** to the latest commits:
```shell
pipx install --force "bnd[processing] @ git+https://github.com/BeNeuroLab/bnd.git"
```

### Option B — conda

1. Install [Miniconda](https://docs.anaconda.com/miniconda/install/#quick-command-line-install) or [Miniforge](https://github.com/conda-forge/miniforge).
2. Clone the repo and create the environment:
   ```shell
   git clone git@github.com:BeNeuroLab/bnd.git
   cd ./bnd
   conda env create --file=processing_env.yml   # includes scientific dependencies
   conda activate bnd
   pip install -e .
   ```
   To update later:
   ```shell
   conda env update --file=processing_env.yml
   ```

## 2. Set up Kilosort (separate conda env)

Kilosort runs in its own conda environment — `bnd` invokes it via `conda run -n kilosort ...`.

1. Create and activate the env:
   ```shell
   conda create -n kilosort python=3.10 pip
   conda activate kilosort
   ```
2. Install Kilosort following the [official instructions](https://github.com/MouseLand/Kilosort):
   ```shell
   python -m pip install "kilosort[gui]"
   ```
   Or minimal (no GUI):
   ```shell
   python -m pip install kilosort
   ```
3. Install GPU-enabled PyTorch (example for CUDA 11.8):
   ```shell
   conda install pytorch pytorch-cuda=11.8 -c pytorch -c nvidia
   ```

> **Note:** If your env is not named `kilosort`, set the environment variable `BND_KILOSORT_ENV` to
> the env name before running `bnd`.

## 3. Configure `bnd`

```shell
bnd init    # Provide the path to local and remote data storage
bnd --help  # Start reading about the functions!
```

# Example usage

Complete your experimental session on animal M099. Then:

```shell
bnd up M099
```

Now, you want to process your data into a pyaldata format. Its a good idea to do this on one of the lab workstations:

```shell
bnd dl M099_2025_01_01_10_00 -v  # Downloads everything
bnd to-pyal M099_2025_01_01_10_00  # Run kilosort, nwb conversion, and pyaldata conversion
bnd up M099_2025_01_01_10_00  # Uploads new files to server
```

If you want specific things during your pipeline (e.g., dont run kilosort, use a custom channel map) read the API below.

# API

## Config

### `bnd init`

Create a .env file (if there isnt one) to store the paths to the local and remote data storage.

### `bnd show-config`

Show the contents of the config file.

## Updating

### `bnd check-updates`

Check if there are any new commits on the repo's main branch.

### `bnd self-update`

Update the bnd tool by pulling the latest commits from the repo's main branch.

## Data Transfer

### `bnd up <session_or_animal_name>`

Upload data from session or animal name to the server. If the file exists on the server, it won't be replaced. Every file in the session folder will get uploaded.

Example usage to upload everything of a given session:

```shell
bnd up M017_2024_03_12_18_45
bnd up M017
```

### `bnd dl <session>`

Download experimental data from a given session from the remote server.

Example usage to download everything:

```shell
bnd dl M017_2024_03_12_18_45 -v  # will download everything, including videos
bnd dl M017_2024_03_12_18_45  # will download everything, except videos
bnd dl M017_2024_03_12_18_45 --max-size=50  # will download files smaller than 50MB
```

## Pipeline

### `bnd to-pyal <session>`

Convert session data into a pyaldata dataframe and saves it as a .mat

If no .nwb file is present it will automatically generate one and if a nwb file is present it will skip it. If you want to generate a new one run `bnd to-nwb`

If no kilosorted data is available it will not kilosort by default. If you want to kilosort add the flag `-k`

Example usage:

```shell
bnd to-pyal M037_2024_01_01_10_00  # Kilosorts data, runs nwb and converts to pyaldata
bnd to-pyal M037_2024_01_01_10_00 -K  # converts to pyaldata without kilosorting (if no .nwb file is present)
bnd to-pyal M037_2024_01_01_10_00 -c  # Use custom mapping during nwb conversion if custom_map.json is available (see template in repo). -C uses available default mapping
```

### `bnd to-nwb <session>`

Convert session data into a nwb file and saves it as a .nwb

If no kilosorted data is available it will not kilosort by default. If you want to kilosort add the flag `-k`

Example usage:

```shell
bnd to-nwb M037_2024_01_01_10_00  # Kilosorts data and run nwb
bnd to-nwb M037_2024_01_01_10_00 -K  # converts to nwb without kilosorting (if no .nwb file is present)
bnd to-nwb M037_2024_01_01_10_00 -c  # Use custom mapping during conversion if custom_map.json is available (see template in repo). Option `-C` uses available default mapping
```

### `bnd ksort <session>`

Kilosorts data from a single session on all available probes and recordings

Example usage:

```shell
bnd ksort M037_2024_01_01_10_00
```

# TODOs:

- Add `AniposeInterface` in nwb conversion
- Implement Npx2.0 functionality
