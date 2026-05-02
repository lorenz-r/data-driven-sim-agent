## 🧱 1. Create clean conda environment

```
conda create -n smart python=3.9 -y
conda activate smart
```

---

## 🔥 2. Install PyTorch (CUDA 11.3 build)

Even if your system has CUDA 12.x, this project requires CUDA 11.3 runtime:

```
conda install pytorch=1.12.1 torchvision=0.13.1 cudatoolkit=11.3 -c pytorch -c defaults --strict-channel-priority -y
```

---

## 📦 3. Install PyTorch Geometric stack

```
pip install torch-scatter==2.1.0 torch-sparse==0.6.16 torch-cluster==1.6.0 torch-spline-conv==1.2.1 -f https://data.pyg.org/whl/torch-1.12.1+cu113.html
```

```
pip install torch-geometric==2.6.1
```

---

## 🧼 4. Clean the requirements file (your key improvement)

You removed:

### ❌ These lines:

- all `+cu113` torch-geometric packages
- Waymo Open Dataset dependency

This is critical because they cause dependency resolution conflicts.

---

## 📦 5. Install remaining dependencies

Now install everything else normally:

```
pip install -r requirements.txt
```

This installs:

- numpy
- scipy
- matplotlib
- pandas
- pytorch-lightning
- etc.

without forcing Waymo or PyG conflicts.

---

## 🛰️ 6. Install Waymo dataset separately (key step)

After everything else is stable:

```
pip install waymo-open-dataset-tf-2-12-0
```