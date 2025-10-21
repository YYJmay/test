# Short Lecture Script — Remote Scientific Computing Workshop




## Opening

Say: "Welcome."

Say: "Today we will introduce some basic programming tools"

Say: "I will introduce the use of vscode, the creation and management of conda environment, the method of remote connection to the server, and the use of github to manage code, etc."

Say: "We will run short demos in a notebook."

Learning goals (say each):
- "Set up a remote workspace."
- "Use NumPy, SciPy, Pandas, Matplotlib, scikit-learn, PyTorch."
- "Use Git and GitHub."
- "Keep jobs running with tmux."

---


## Part 1 — VS Code Remote Connection

Say: "Why use VS Code?"
Say: "It edits code and shows notebooks."
Say: "It runs a terminal on the server."

Demo steps (say and do):
1. Say: "Open Command Palette: Cmd/Ctrl+Shift+P."
2. Say: "Run: Remote-SSH: Connect to Host."
3. Say: "Type: ssh username@server.address."
4. Say: "Enter your password." Then show the connection.
5. Say: "Open the terminal and run:"

```bash
hostname
nvidia-smi  # if GPU exists
```

Say: "Look at the bottom-left. It shows the SSH host." 

Takeaways (say each):
- "Edit files on the server." 
- "Terminal runs on the remote machine." 
- "VS Code will install server tools on first connect." 

---


## Part 2 — Conda and Python Stack

Say: "Conda creates clean environments."
Say: "This avoids version conflicts." 

Run these commands and say each line while you run it:

```bash
conda create -n sci python=3.10 -y
conda activate sci
which python
python --version
```

Say: "Now install core packages and the kernel." Then run:

```bash
pip install numpy scipy pandas matplotlib seaborn scikit-learn jupyterlab ipykernel tqdm
python -m ipykernel install --user --name sci --display-name "Python (sci, remote)"
```

Demo pointer: Run notebook cell 3 (Environment Check). 
Say: "This cell prints Python and package versions and GPU info." 

If a module error appears, say: "Switch kernel or install the package in the current interpreter." 

Takeaways (say):
- "Use conda for reproducible envs." 
- "Register the env as a notebook kernel." 
- "Always run an env check." 

---


## Part 3 — Demos: NumPy, SciPy, Pandas, Matplotlib, scikit-learn

Say: "We will run short demos for each tool." 

NumPy demo (say and run): Run notebook cell 5.
Say: "Note vectorization and speed." 

SciPy demo: Run notebook cell 7.
Say: "This shows optimization and integration." 

Pandas demo: Run notebook cell 9.
Say: "This shows DataFrame and summary stats." 

Matplotlib demo: Run notebook cell 11.
Say: "This draws plots. Point out labels and legend." 

scikit-learn demo: Run notebook cell 13.
Say: "This shows simple regression, train/test split, and metrics." 

Takeaways (say):
- "Each tool has a job." 
- "Pandas for tables, Matplotlib for plots." 
- "scikit-learn for classic ML." 

---


## Part 4 — Git and GitHub

Say: "Git tracks code history. GitHub stores repos and helps teams." 

Say the benefits (short):
- "Use Issues to track tasks." 
- "Connect code, PRs, and CI." 
- "You can link to one line of code." 

Show how to clone (say and run):

```bash
git clone https://github.com/YOUR_USERNAME/sci-workshop.git
cd sci-workshop
```

Or SSH if keys exist:

```bash
git clone git@github.com:YOUR_USERNAME/sci-workshop.git
```

Quick workflow (say and run):

```bash
git status
git checkout -b demo/your-name
# edit files
git add -A
git commit -m "demo: add changes"
git push --set-upstream origin demo/your-name
```

Say: "Open a pull request on GitHub for review." 

---


## Part 5 — tmux for Long Runs

Say: "Use tmux to keep jobs running if you disconnect." 

Demo steps (say and run):

```bash
tmux new -s train
conda activate sci
python src/long_training.py
# Detach: Ctrl+B then D
tmux attach -t train
```

Say: "Use tmux ls to list sessions and tmux kill-session to stop one." 

Takeaways (say):
- "tmux keeps programs alive." 
- "You can reattach later." 
- "Log outputs to a file for records." 

---


## PyTorch Demos

Say: "Two short PyTorch demos now. First Monte Carlo. Then a small MLP."

Monte Carlo: Run notebook cell 15.
Say: "This makes random tensors and checks points inside a circle to estimate pi." 
Say: "It times CPU and GPU. Use torch.cuda.synchronize() for good timing." 

MLP demo: Run notebook cell 17.
Say: "We make moons data, build a small MLP, train with BCELoss and Adam." 
Say: "Core loop: forward, backward, update." 

If GPU is available, say and run this code to move data and model:

```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
X_train_tensor = X_train_tensor.to(device)
y_train_tensor = y_train_tensor.to(device)
```

Takeaways (say):
- "Training loop steps: forward, backward, optimizer step." 
- "Keep model and tensors on the same device." 
- "Use plots to check results." 

---


## Final Summary and Exercises

Say: "Do these three tasks." 
1. "Clone the repo, make the sci env, register the kernel." 
2. "Run the notebook and fix any kernel errors." 
3. "Change the linear model to Ridge and compare RMSE." 

Say: "Save environment with `conda env export > env.yaml`."

Say: "Push code to GitHub and open a PR for review." 

---


## Appendix — Quick references

- Environment check: Run notebook cell 3.
- NumPy demo: Run notebook cell 5.
- SciPy demo: Run notebook cell 7.
- Pandas demo: Run notebook cell 9.
- Matplotlib demo: Run notebook cell 11.
- scikit-learn demo: Run notebook cell 13.
- PyTorch Monte Carlo: Run notebook cell 15.
- PyTorch MLP: Run notebook cell 17.

If you see ModuleNotFoundError, say: "Check kernel or install the package into the running interpreter." See `TUTORIAL.md` for commands.

---

End of short script. Read lines aloud. Run demo cells when prompted.
