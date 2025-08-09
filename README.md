# An Study of Offensive Lineman Run Blocking Abilites


## Setting up the Env to run the notebook:

### Step 1: Create the Conda Environment

Make sure you have [Conda](https://docs.conda.io/en/latest/miniconda.html) installed.

```bash
conda env create -f environment.yml
```

### Step 2: Activate the Environment

```bash
conda activate nfl-oline
```

### Step 3: Register the Environment as a Jupyter Kernel

```bash
python -m ipykernel install --user --name=nfl-oline --display-name="nfl-oline"
```

### Step 4: Launch Jupyter Notebook

```bash
jupyter notebook
```

Then, in the notebook interface, select the kernel named **nfl-oline**.

### Step 5: Add the "data" folder
Place the unedited 2025 NFL Big Data Bowl competition data in a folder named "data" at the root level of the project.

---

You’re now ready to run the notebook using the `nfl-oline` environment.
