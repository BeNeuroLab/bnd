# bnd : BeNeuroLab Data Organization

A collection of functions for managing the experimental data recorded in the 
BeNeuro Lab, and a CLI tool called `bnd` for easy access to this functionality.

# TODOs:
   - Create `up` and `down` functions
   - Create `to-pyal` pipeline
     - Create `to-nwb` pipeline
     - Create `ksort` pipeline without docker
   - Set up github action for environment
   - Begin building some tests


# Setting up
## Installation

1. Install `conda`
   - You can use either [Miniconda](https://docs.anaconda.com/miniconda/install/#quick-command-line-install) or [Miniforge](https://github.com/conda-forge/miniforge)
2. Clone repo
   ```shell
   git clone https://github.com/BeNeuroLab/beneuro_experimental_data_organization.git
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

   If you installed the base environment and want to update later on:
   ```shell
   conda env update -n bnd --file=processing_env.yml
   ```
4. Create your configuration file:
   ```shell
   bnd init  # Provide the path to local and remote data storage
   bnd --help # Start reading about the functions!