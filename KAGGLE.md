# Kaggle Integration

**Competition Link:** https://www.kaggle.com/competitions/bengali-hallucination/

Here is how you can connect your local Jupyter notebooks (such as `starter_notebook.ipynb` or `solution_pipeline.ipynb`) in VS Code to a remote Kaggle Jupyter Server instance to utilize Kaggle's GPU/TPU compute.

---

## Step-by-Step Connection Guide

### 1. Get the Connection URL from Kaggle
1. Go to the [Bengali Hallucination Competition](https://www.kaggle.com/competitions/bengali-hallucination/) on Kaggle.
2. Open or create a Notebook session.
3. Click the **Run** menu in the top toolbar of the Kaggle notebook interface.
4. Select **Kaggle Jupyter Server** from the dropdown list.
5. Copy the **VS Code Compatible URL** shown in the side panel.

### 2. Connect in VS Code
1. Open your local notebook in **VS Code**.
2. Click the **Select Kernel** button in the top-right corner of the notebook editor.
3. Select **Existing Jupyter Server...** (or *Select Another Kernel* -> *Existing Jupyter Server...*).
4. Paste the copied URL and press **Enter**.
5. Select the remote kernel from the list.

Once connected, executing cells locally in VS Code will run them on the Kaggle backend.