# Getting Started

This tutorial teaches you how to set up a complete scientific computing environment on remote servers.

---

## File Structure

| File | Purpose | When to Use |
|------|---------|-------------|
| **TUTORIAL.md** | Complete step-by-step guide | Read through during setup |
| **DEMO.ipynb** | Executable code demonstrations | Run after environment is ready |
| **README.md** | Original reference (English) | Optional reference |

---

## Learning Path

### For Classroom Teaching

**Recommended Order**:

1. **Part 1 (TUTORIAL.md)**: Demonstrate VS Code installation and SSH connection
2. **Part 2 (TUTORIAL.md)**: Guide through Conda environment setup
3. **Section 1-2 (DEMO.ipynb)**: Show NumPy and SciPy examples
4. **Part 3 (TUTORIAL.md)**: Explain project structure
5. **Section 3-6 (DEMO.ipynb)**: Demonstrate Pandas, Matplotlib, scikit-learn, PyTorch
6. **Part 4-5 (TUTORIAL.md)**: Cover Git and tmux for practical workflows

**Total Time**: 2-3 hours

---

### For Self-Study

**Step 1**: Read TUTORIAL.md Part 1-2 (install VS Code, connect to server, set up environment)

**Step 2**: Follow commands to create Conda environment and install packages

**Step 3**: Run DEMO.ipynb cells sequentially to verify installation

**Step 4**: Read TUTORIAL.md Part 3-5 (project structure, Git, tmux)

**Step 5**: Start your own project using the learned tools

---

## Quick Reference

### Connect to Server

```bash
# In VS Code
Ctrl+Shift+P -> "Remote-SSH: Connect to Host"
# Enter: ssh username@server.address
```

### Activate Environment

```bash
conda activate sci
```

### Check GPU

```bash
nvidia-smi
```

### Create tmux Session

```bash
tmux new -s training
# Detach: Ctrl+B then D
# Reattach: tmux attach -t training
```

---

## Troubleshooting

### Cannot connect to server

1. Verify server address and username
2. Check password
3. Contact administrator

### Conda command not found

```bash
source ~/.bashrc
```

### PyTorch cannot use GPU

```bash
# Check GPU availability
nvidia-smi

# Reinstall PyTorch with matching CUDA version
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### Jupyter kernel not found

```bash
# Re-register kernel
conda activate sci
python -m ipykernel install --user --name sci --display-name "Python (sci, remote)"
```

---

## Support

If you encounter issues:

1. Check TUTORIAL.md troubleshooting section
2. Verify all commands in Part 2 completed successfully
3. Check environment with DEMO.ipynb Section 1

---

**Ready to start? Open TUTORIAL.md**
