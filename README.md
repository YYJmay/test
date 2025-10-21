# Remote Server Scientific Computing Environment Setup Tutorial

**Target Audience**: Students who need to use remote servers for scientific computing, machine learning, and deep learning  
**Learning Objectives**: Configure a professional remote development environment from scratch, master remote connection, Conda, Git, tmux and other core tools  


---

## Tutorial Structure

| Part | Content | 
|------|---------|
| **Part 1** | VS Code Installation and Remote SSH Connection | 
| **Part 2** | Python Scientific Computing Environment | 
| **Part 3** | Project Structure and Example Code | 
| **Part 4** | Git Version Control and GitHub | 
| **Part 5** | Running Long-Running Tasks with tmux | 

---

# Part 1. VS Code Installation and Remote Connection

## Learning Objectives

- Install and configure VS Code on local computer
- Connect to remote server using username and password
- Edit remote files directly in VS Code
- Install Miniconda on remote server (if needed)

---

## 1.1 Install Visual Studio Code (Local Computer)

### Why VS Code?

| Feature | Benefit |
|---------|---------|
| **Code Editing** | IntelliSense, syntax highlighting, code navigation |
| **Integrated Terminal** | No need to switch windows to run commands |
| **Jupyter Support** | Run `.ipynb` files directly in editor |
| **Remote Development** | Write and run code on remote GPU servers via SSH |
| **Git Integration** | Seamless version control |

### Download and Install

1. Visit official website: https://code.visualstudio.com/
2. Download version for your OS (Windows / macOS / Linux)
3. Run installer, follow default options

**Teaching Note**: Demonstrate download, installation, and first launch

---

## 1.2 Install Required VS Code Extensions

### How to Install Extensions

1. Click **Extensions** icon in left sidebar (four squares)
2. Or press: `Ctrl + Shift + X` (Windows/Linux) / `Cmd + Shift + X` (macOS)
3. Search extension name
4. Click **Install**

### Required Extensions

| Extension Name | Purpose | Priority |
|---------------|---------|----------|
| **Python** | Python code execution, debugging | Must have |
| **Jupyter** | Run `.ipynb` notebooks | Must have |
| **Pylance** | Python IntelliSense and type checking | Must have |
| **Remote - SSH** | **Core plugin for connecting to remote servers** | Must have |
| **GitLens** | Git history, code authors | Recommended |

**Teaching Note**: 
- Open Extensions panel
- Search and install above extensions
- Emphasize importance of **Remote - SSH**

---

## 1.3 Connect to Remote Server Using VS Code

### Benefits of Remote Development

| Benefit | Description |
|---------|-------------|
| Direct file editing | No manual upload/download |
| Use server resources | GPU, large memory, multi-core CPU |
| Integrated terminal | Execute commands directly on server |
| Environment isolation | Python packages stay on server |

### Connection Steps (Detailed Demonstration)

#### Step 1: Open Command Palette

In VS Code, press:
- `Ctrl + Shift + P` (Windows/Linux)
- `Cmd + Shift + P` (macOS)

#### Step 2: Enter Connection Command

In command palette, type:
```
Remote-SSH: Connect to Host
```

#### Step 3: Enter Server Address

Select **"Add New SSH Host"**, then enter:

```bash
ssh username@server.address.com
```

**Example**:
```bash
ssh student01@gpu.hkbu.edu.hk
```

**Parameter Description**:
- `username`: Your username on the server
- `server.address.com`: Server address (IP or domain name)

#### Step 4: Select Configuration File

Choose location to save SSH config, recommended:
```
~/.ssh/config  (Linux/macOS)
C:\Users\YourName\.ssh\config  (Windows)
```

#### Step 5: Connect to Server

1. Open command palette again (`Ctrl+Shift+P`)
2. Select `Remote-SSH: Connect to Host`
3. Select the server you just added (e.g., `gpu.hkbu.edu.hk`)
4. **Select OS type**: Usually select **Linux**
5. **Enter password**: Input your server login password

**Note for First Connection**: VS Code will automatically install necessary components on server (takes 1-3 minutes)

#### Step 6: Verify Successful Connection

Connection success indicators:

| Location | Display |
|----------|---------|
| **Bottom-left corner** | Green\Blue label like `SSH: gpu.hkbu.edu.hk` |
| **Terminal** | Open terminal shows server shell |

#### Step 7: Open Remote Folder

1. Click **Explorer** icon on left
2. Click **Open Folder**
3. Enter path: `/home/student01` (or your username)
4. Click **OK**
5. **Enter password again**

Now all files you see are on the remote server.

### Test Remote Execution

Open integrated terminal (`` Ctrl + ` ``) and run:

```bash
# Check current location
pwd

ls

ssh gpu12-25

# Check hostname
hostname

# If GPU available, check info
nvidia-smi
```

If commands show server information, connection successful.


---

## 1.4 Install Miniconda on Server (If Needed)

### When Do You Need This?

**Need to install**:
- Server does not have Anaconda/Miniconda pre-installed
- You need to manage your own Python environments

**Do not need**:
- Server already has conda

### Check if Server Has Conda

```bash
conda --version
```

If shows version number, already installed; if shows "command not found", need to install.

---

### Miniconda Installation Steps

Execute in remote server terminal:

#### Step 1: Download Installation Script

```bash
cd ~
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh

# If wget not available, use curl
# curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
```

#### Step 2: Run Installation

```bash
bash Miniconda3-latest-Linux-x86_64.sh
```

**Interactive Process**:
1. Read license, type `yes`
2. Installation location, press `Enter` (use default)
3. Initialize conda, type `yes`

#### Step 3: Activate Conda

```bash
source ~/.bashrc
conda --version
```

#### Step 4: Configure (Optional)

```bash
# Disable auto-activation of base environment
conda config --set auto_activate_base false
```

#### Step 5: Clean Up Installation File

```bash
rm ~/Miniconda3-latest-Linux-x86_64.sh
```

Installation complete.

---

## Part 1 Completion Checklist

- [ ] VS Code installed
- [ ] Python, Jupyter, Remote-SSH extensions installed
- [ ] Can connect to remote server with password
- [ ] Remote server has available conda

---

# Part 2. Configure Python Scientific Computing Environment

## Learning Objectives

- Create isolated Conda virtual environment
- Install scientific computing stack (NumPy/SciPy/Pandas/Matplotlib/scikit-learn)
- Configure PyTorch (GPU support)
- Register Jupyter kernel

---

## 2.1 Understanding the Scientific Python Ecosystem

Before installation, let's understand the role of each package in the scientific computing workflow:

### The Scientific Computing Stack (Layer by Layer)

```
Application Layer
    |
    |-- Machine Learning: scikit-learn (classification, regression, clustering)
    |-- Deep Learning: PyTorch (neural networks, GPU acceleration)
    |
Data Processing Layer
    |
    |-- Tables/DataFrames: Pandas (CSV, SQL-like operations)
    |-- Visualization: Matplotlib, Seaborn (plots, charts)
    |
Numerical Foundation Layer
    |
    |-- Arrays/Matrices: NumPy (n-dimensional arrays, broadcasting)
    |-- Scientific Functions: SciPy (optimization, integration, linear algebra)
    |
Hardware Abstraction Layer
    |
    |-- CPU: Default NumPy backend
    |-- GPU: PyTorch with CUDA support
```

### Package Roles and Relationships

**NumPy (Numerical Python)**:
- Core: Multi-dimensional arrays (ndarray)
- Purpose: Fast vectorized operations on arrays
- Use case: Any numerical computation (foundation for almost all other packages)
- Key concept: Broadcasting (element-wise operations without loops)

**SciPy (Scientific Python)**:
- Core: Built on top of NumPy
- Purpose: Advanced mathematical algorithms
- Use case: Optimization, integration, interpolation, signal processing, linear algebra
- Key concept: Extends NumPy with specialized scientific functions

**Pandas**:
- Core: DataFrame (2D labeled table) and Series (1D labeled array)
- Purpose: Data manipulation and analysis
- Use case: Loading CSV/Excel, data cleaning, grouping, merging, time series
- Key concept: Like SQL for Python, Excel on steroids

**Matplotlib**:
- Core: MATLAB-like plotting library
- Purpose: Create static, animated, and interactive visualizations
- Use case: Line plots, scatter plots, histograms, heatmaps
- Key concept: Object-oriented plotting with figures and axes

**Seaborn**:
- Core: Built on Matplotlib
- Purpose: Statistical data visualization with beautiful defaults
- Use case: Distribution plots, regression plots, categorical plots
- Key concept: High-level interface for complex statistical plots

**scikit-learn**:
- Core: Classical machine learning algorithms
- Purpose: Predictive data analysis
- Use case: Classification, regression, clustering, dimensionality reduction
- Key concept: Consistent API (fit, predict, transform)

**PyTorch**:
- Core: Tensor library with GPU acceleration
- Purpose: Deep learning and neural networks
- Use case: CNN, RNN, Transformers, any gradient-based optimization
- Key concept: Automatic differentiation, GPU tensors

---

## 2.2 Create Isolated Conda Environment

### Why Virtual Environments?

| Problem | Virtual Environment Solution |
|---------|----------------------------|
| Package version conflicts | Each project has independent environment |
| Difficult to reproduce | Export config, others can rebuild exactly |
| System pollution | Does not affect system Python |

### Create Environment

Execute in remote server terminal:

```bash
# Ensure conda is available
source ~/.bashrc

# Create environment named "sci" with Python 3.10
conda create -n sci python=3.10 -y

# Activate environment
conda activate sci
```

Prompt changes: `(base)` -> `(sci)`

### Verify Environment

```bash
# Check Python location
which python

# Check version
python --version
```

Path should contain `envs/sci/`

---

## 2.3 Install Scientific Computing Libraries

### Packages to Install

| Category | Packages | Purpose |
|----------|----------|---------|
| **Numerical** | numpy, scipy | Arrays, linear algebra, optimization |
| **Data** | pandas | Tabular data processing |
| **Visualization** | matplotlib, seaborn | Plotting |
| **Machine Learning** | scikit-learn | Classical ML algorithms |
| **Jupyter** | jupyterlab, ipykernel | Notebook support |

### Installation Steps

Ensure `sci` environment is activated:

```bash
# Upgrade pip
pip install --upgrade pip

# Install core libraries
pip install numpy scipy pandas matplotlib seaborn scikit-learn

# Install Jupyter support
pip install jupyterlab ipykernel

# Install helper tools
pip install tqdm
```

Installation time: 3-10 minutes

### Verify Installation

```bash
python -c "import numpy; print('NumPy:', numpy.__version__)"
python -c "import pandas; print('Pandas:', pandas.__version__)"
python -c "import sklearn; print('scikit-learn:', sklearn.__version__)"
```




---

## 2.4 Understanding GPU Computing and CUDA

### What is CUDA?

**CUDA (Compute Unified Device Architecture)**:
- NVIDIA's parallel computing platform
- Allows programs to use GPU for general computation (not just graphics)
- Essential for deep learning acceleration

### GPU Computing Hierarchy

```
Your Program (Python + PyTorch)
        |
    PyTorch CUDA Interface
        |
    CUDA Toolkit (nvcc, cuBLAS, cuDNN)
        |
    CUDA Driver (on server)
        |
    GPU Hardware (RTX 4090, A100, etc.)
```

### Key Terms

| Term | Description |
|------|-------------|
| **GPU** | Graphics card hardware (e.g., NVIDIA RTX 4090) |
| **CUDA Driver** | Low-level driver, determines maximum CUDA version supported |
| **CUDA Toolkit** | Development tools and libraries (nvcc compiler, cuBLAS, cuDNN) |
| **PyTorch CUDA** | PyTorch compiled with specific CUDA version |

### Why Version Matching Matters

PyTorch must be compiled with CUDA version **equal to or lower than** driver version.

Example:
- Driver supports CUDA 12.1 -> Can use PyTorch with CUDA 12.1, 11.8, or CPU
- Driver supports CUDA 11.8 -> Can use PyTorch with CUDA 11.8 or CPU (NOT 12.1)

---

## 2.5 Check GPU and Install PyTorch

### Step 1: Check GPU Information

```bash
nvidia-smi
```

**Look for key information in output**:
- **CUDA Version**: e.g., `12.0` or `11.8` (driver's maximum supported version)
- **GPU Model**: e.g., `NVIDIA RTX A6000`
- **Memory**: e.g., `48GB`

If command not found, server has no GPU; install CPU-only PyTorch.

### Step 2: Install PyTorch

Choose based on CUDA version:

#### GPU Available (CUDA 12.x)

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

#### GPU Available (CUDA 11.8)

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

#### CPU Only Version

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

Installation time: 5-15 minutes

### Step 3: Verify PyTorch Installation

Create test file:

```bash
cat > ~/check_torch.py << 'EOF'
import torch

print("=" * 60)
print("PyTorch Environment Check")
print("=" * 60)
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU count: {torch.cuda.device_count()}")
    print(f"GPU 0: {torch.cuda.get_device_name(0)}")
    
    # Simple test
    x = torch.rand(1000, 1000, device='cuda')
    y = torch.mm(x, x)
    print(f"GPU compute test: Success")
else:
    print("Running in CPU mode")
print("=" * 60)
EOF

python ~/check_torch.py
```

If see `CUDA available: True`, GPU configuration successful.

---

## Part 2 Completion Checklist

- [ ] Created `sci` Conda environment
- [ ] Installed NumPy, SciPy, Pandas, Matplotlib, scikit-learn
- [ ] Installed JupyterLab and ipykernel
- [ ] Installed PyTorch (GPU or CPU)
- [ ] Verified PyTorch and CUDA

---

# Part 3. Project Structure and Example Code

## Learning Objectives

- Create standard project structure
- Understand when to use each library
- Write runnable example code

---

## 3.1 Standard Project Structure

### Create Folders

```bash
cd ~
mkdir -p sci-workshop/{src,notebooks,data,results,figures}
cd sci-workshop
```

### Project Structure

```
sci-workshop/
├── src/                    # Python scripts (.py files)
├── notebooks/              # Jupyter notebooks (.ipynb)
├── data/                   # Data files
├── results/                # Output files
├── figures/                # Plots and images
└── README.md              # Project description
```

### Why This Structure?

| Folder | Purpose | Git Tracked? |
|--------|---------|--------------|
| `src/` | Reusable code, production scripts | Yes |
| `notebooks/` | Experiments, exploration, demos | Yes |
| `data/` | Raw data (often large) | No (add to .gitignore) |
| `results/` | Model outputs, logs, metrics | No (generated files) |
| `figures/` | Plots for papers/reports | Yes (if small) |

---

## 3.2 Scientific Computing Workflow Examples

All example code is included in the companion `DEMO.ipynb` notebook, organized by:

### Example 1: NumPy - Foundation of Numerical Computing

**What NumPy Does**:
- Provides n-dimensional array object (ndarray)
- Enables vectorized operations (operations on entire arrays without Python loops)
- Foundation for almost all scientific Python packages

**Key Concepts**:
- **Broadcasting**: Automatic element-wise operations on arrays of different shapes
- **Vectorization**: Replace Python loops with compiled C operations (10-100x faster)
- **Linear Algebra**: Matrix operations, eigenvalues, solving systems of equations

**When to Use NumPy**:
- Any numerical computation
- Working with arrays/matrices
- Performance-critical code (replace Python loops)
- Foundation for data passed to other libraries

**Demonstration in Notebook**:
- Performance comparison: Python loops vs NumPy vectorization
- Solving linear systems (Ax = b)
- Eigenvalue decomposition

---

### Example 2: SciPy - Advanced Scientific Algorithms

**What SciPy Does**:
- Built on NumPy, adds specialized scientific algorithms
- Optimization, integration, interpolation, signal processing
- More advanced linear algebra than NumPy

**Key Modules**:
- `scipy.optimize`: Minimize/maximize functions, curve fitting
- `scipy.integrate`: Numerical integration (definite integrals, ODEs)
- `scipy.linalg`: Advanced linear algebra
- `scipy.stats`: Statistical distributions and tests
- `scipy.signal`: Signal processing (filters, FFT)

**When to Use SciPy**:
- Optimization problems (find minimum/maximum)
- Numerical integration (when analytical solution impossible)
- Signal processing
- Advanced statistical analysis

**Demonstration in Notebook**:
- Function minimization
- Numerical integration (Gaussian integral)
- Solving differential equations

---

### Example 3: Pandas - Data Manipulation and Analysis

**What Pandas Does**:
- Provides DataFrame (2D labeled table) like SQL or Excel
- Data loading (CSV, Excel, SQL), cleaning, transformation
- Group-by operations, merging, time series

**Key Concepts**:
- **DataFrame**: 2D table with labeled rows and columns
- **Series**: 1D labeled array (single column)
- **Indexing**: Select rows/columns by label or position
- **GroupBy**: Split-apply-combine operations

**When to Use Pandas**:
- Loading data from files (CSV, Excel, JSON)
- Data cleaning (handle missing values, duplicates)
- Exploratory data analysis (statistics, correlations)
- Feature engineering for machine learning
- Any "table-like" data

**Demonstration in Notebook**:
- Generate synthetic data with noise
- DataFrame operations (statistics, correlations)
- Integration with Matplotlib for visualization

---

### Example 4: Matplotlib - Visualization

**What Matplotlib Does**:
- Create static, animated, and interactive plots
- MATLAB-like interface
- Fine-grained control over every plot element

**Key Concepts**:
- **Figure**: Top-level container (entire window)
- **Axes**: Individual plot area (can have multiple in one figure)
- **Object-oriented interface**: Explicit control vs pyplot convenience

**When to Use Matplotlib**:
- Creating publication-quality figures
- Any type of 2D plotting
- Custom visualizations requiring fine control
- Foundation for other plotting libraries (Seaborn, Pandas plotting)

**Demonstration in Notebook**:
- Scatter plots with regression line
- Histograms of residuals
- Multi-panel figures

---

### Example 5: scikit-learn - Machine Learning

**What scikit-learn Does**:
- Provides classical machine learning algorithms
- Consistent API across all algorithms
- Tools for model evaluation and selection

**Key Concepts**:
- **Estimators**: Objects that fit data (e.g., LinearRegression)
- **fit()**: Train model on data
- **predict()**: Make predictions
- **transform()**: Transform data (e.g., scaling, encoding)
- **Pipeline**: Chain multiple steps

**When to Use scikit-learn**:
- Classical ML (not deep learning)
- Supervised learning (classification, regression)
- Unsupervised learning (clustering, dimensionality reduction)
- Model evaluation (cross-validation, metrics)

**Demonstration in Notebook**:
- Complete ML workflow: train/test split
- Linear regression training
- Model evaluation (R^2, RMSE)
- Visualization of predictions

---

### Example 6: PyTorch - Deep Learning and GPU Computing

**What PyTorch Does**:
- Tensor library with GPU acceleration
- Automatic differentiation (autograd)
- Neural network building blocks
- Production deployment tools

**Key Concepts**:
- **Tensor**: Like NumPy array but can run on GPU
- **Device**: Where tensor lives (CPU or CUDA)
- **Autograd**: Automatic gradient computation for optimization
- **nn.Module**: Base class for neural networks

**When to Use PyTorch**:
- Deep learning (CNNs, RNNs, Transformers)
- Any GPU-accelerated computation
- Automatic differentiation needed
- Large-scale numerical computations

**Demonstration in Notebook**:
- Monte Carlo estimation of pi
- CPU vs GPU performance comparison
- Tensor operations
- Device management

---

## 3.3 Choosing the Right Tool

### Decision Tree

```
Do you need deep learning or GPU acceleration?
    Yes -> Use PyTorch
    No -> Continue

Is your data in table format (rows/columns)?
    Yes -> Use Pandas (+ Matplotlib for plots)
    No -> Continue

Do you need optimization or integration?
    Yes -> Use SciPy
    No -> Continue

Do you need numerical arrays/matrices?
    Yes -> Use NumPy
    No -> Use pure Python
```

### Common Combinations

| Task | Tools |
|------|-------|
| Data analysis project | Pandas + Matplotlib + NumPy |
| Classical ML project | scikit-learn + Pandas + NumPy |
| Deep learning project | PyTorch + NumPy + Matplotlib |
| Scientific simulation | NumPy + SciPy + Matplotlib |
| Optimization problem | SciPy + NumPy |

---

# Part 4. Git Version Control and GitHub

Why use GitHub?

- GitHub provides a single place to store your project history and collaborate with others. Use Git for version control and GitHub as the central remote where teammates push and pull changes.

- Issue tracking: open an Issue to record a bug, a task, or a feature request. Issues can be assigned, labeled, cross-referenced, and linked to pull requests so work stays organized.

- Collaboration and integration: GitHub connects repositories, pull requests, code review, CI/CD pipelines, and project boards. This ecosystem reduces friction when multiple people work together and helps reproducibility.

- Precise code references: GitHub lets you create permalinks to a specific file and line number. That means you can point someone directly to the exact line of code from an issue, a PR, or a slide during class.


## Learning Objectives

- Configure Git on server
- Initialize repository and commit code
- Push to GitHub



---

## 4.1 Configure Git (On Server)

```bash
# Set username and email
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"

# Verify configuration
git config --list --global
```
---

## 4.2 Cloning a repository (what to say and show)

"To work on a project, clone the remote repository to your server using git clone. If you use HTTPS, you'll be prompted for credentials or a personal access token. If you prefer SSH, clone with the SSH URL after you set up keys." 

Example (HTTPS):

```bash
git clone git clone https://github.com/YYJmay/test.git
```

Example (SSH):

```bash
git clone git@github.com:YYJmay/test.git
```

<!-- Say: "After cloning, cd into the project, create a branch for your work, and start making changes. Use pull requests to propose changes and request a review." -->

---

## 4.3 Push to GitHub

### Method 1: HTTPS (Recommended, Simple)

1. Create new repository on GitHub: https://github.com/new
2. Repository name: `sci`
4. After creation, GitHub will show commands:

```bash
git branch

git checkout -b new

git status

git add .

git commit -m "my code"

git push -u origin new
```

5. Enter GitHub username and password (or Personal Access Token)



---
## 4.4 Daily Git Workflow

```bash
# Check status
git status

# Add modifications
git add new_file.py

(git add .)

# Commit
git commit -m "feat: add new functionality"

# Push
git push
```

---

# Part 5. Running Long Tasks with tmux

## Learning Objectives

- Understand purpose of tmux
- Master basic operations
- Run long GPU tasks

---

## 5.1 Why tmux?

### Problem Without tmux

| Scenario | Consequence |
|----------|-------------|
| Wi-Fi disconnects | Program stops |
| Close laptop | SSH disconnects, task interrupted |
| VS Code closes | Terminal closes, program terminates |

### Solution: tmux

- Program continues running on server
- Can disconnect and reconnect anytime
- Suitable for long training tasks

---

## 5.2 tmux Basic Commands

| Command | Description |
|---------|-------------|
| `tmux new -s <name>` | Create session |
| `Ctrl+B` then `D` | Detach (program keeps running) |
| `tmux ls` | List all sessions |
| `tmux attach -t <name>` | Reconnect |
| `exit` | Exit and close session |

---

## 5.3 Complete Workflow for Running GPU Tasks

### Step-by-Step Demonstration

```bash
# 1. SSH login
ssh username@server.com

# 2. Activate environment
conda activate sci
cd ~/test

# 3. Create tmux session
tmux new -s train

# 4. Run program (example)
python src/long_training.py

# 5. Detach session
# Press Ctrl+B, then D

# 6. Check sessions
tmux ls

# 7. Reconnect later
tmux attach -t train
```
---

## Part 5 Completion Checklist

- [ ] Understand tmux purpose
- [ ] Can create, detach, reattach sessions
- [ ] Verified program continues running after detach

