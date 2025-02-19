# bnd : BeNeuroLab Data Organization

A **lightweight** collection of functions for managing the experimental data recorded in the 
BeNeuro Lab, and a CLI tool called `bnd` for easy access to this functionality.

Play around with it and raise Github issues if anything fails

# Setting up

1. Install `conda`
   - You can use either [Miniconda](https://docs.anaconda.com/miniconda/install/#quick-command-line-install) or [Miniforge](https://github.com/conda-forge/miniforge)
2. Clone repo
   ```shell
   git clone git@github.com:BeNeuroLab/bnd.git
   cd ./beneuro_experimental_data_organization
   ```
3. Open either Miniconda prompt or Miniforge promt and run the following command. This 
   may take some time: 
   ```shell
   conda env create --file=env.yml
   ```
   or if you want the processing depedencies:
   ```shell
   conda env create --file=processing_env.yml
   ```

   For kilosorting you will also need:
   1. Install kilosort and the GUI, run `python -m pip install kilosort[gui]`. If you're on a zsh server, you may need to use `python -m pip install "kilosort[gui]"` 
   2. You can also just install the minimal version of kilosort with python -m pip install kilosort.
   3. Next, if the CPU version of pytorch was installed (will happen on Windows), remove it with `pip uninstall torch`
   4. Then install the GPU version of pytorch `conda install pytorch pytorch-cuda=11.8 -c pytorch -c nvidia`

   If you installed the base environment and want to update later on:
   ```shell
   conda env update --file=processing_env.yml
   ```
   And then do the kilosort step
4. Create your configuration file:
   ```shell
   bnd init  # Provide the path to local and remote data storage
   bnd --help # Start reading about the functions!

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
