# 🚀 Complete Deployment Guide
## VSCode → GitHub → Streamlit Community Cloud
### AML Fraud Detection — Khalid Dharif

---

## Prerequisites Checklist

Before starting, make sure you have:
- [ ] Python 3.9–3.11 installed → https://www.python.org/downloads/
- [ ] Git installed → https://git-scm.com/downloads
- [ ] VSCode installed → https://code.visualstudio.com/
- [ ] A GitHub account → https://github.com
- [ ] A Streamlit account (sign in with GitHub) → https://streamlit.io/cloud
- [ ] The `.joblib` model files generated from running the notebook on Kaggle

---

## PART 1 — Set Up VSCode & Virtual Environment

### Step 1.1 — Install VSCode Extensions
Open VSCode and install these extensions (Ctrl+Shift+X):
```
Python          (Microsoft)
Pylance         (Microsoft)
GitLens         (GitKraken)   ← optional but very helpful
Jupyter         (Microsoft)   ← to view the notebook locally
```

### Step 1.2 — Create Your Project Folder
Open a terminal in VSCode (Ctrl+` backtick):
```bash
# Navigate to where you want your project
cd Desktop   # or wherever you prefer

# Create the project folder
mkdir aml-fraud-detection
cd aml-fraud-detection
```

### Step 1.3 — Create a Virtual Environment
A virtual environment isolates your project's packages from the rest of your system.
```bash
# Create it
python -m venv venv

# Activate it (Windows)
venv\Scripts\activate

# Activate it (macOS / Linux)
source venv/bin/activate
```

You should see **(venv)** at the start of your terminal prompt.

### Step 1.4 — Open the Folder in VSCode
```bash
code .
```
VSCode will open the folder. In the bottom-left corner, select your Python interpreter:
→ Click the Python version indicator → Select `./venv/Scripts/python` (Windows) or `./venv/bin/python` (Mac/Linux)

---

## PART 2 — Organise Project Files

### Step 2.1 — Copy Files Into the Folder
Your project folder should look like this:
```
aml-fraud-detection/
├── app.py
├── requirements.txt
├── README.md
├── GP_AML_Khalid_Dharif_v2.ipynb
└── models/
    ├── aml_best_model.joblib
    ├── aml_scaler.joblib
    ├── aml_le_from_bank.joblib
    ├── aml_le_to_bank.joblib
    ├── aml_le_bank_pair.joblib
    ├── aml_le_currency.joblib
    ├── aml_le_format.joblib
    ├── aml_feature_columns.joblib
    └── aml_model_name.joblib
```

**How to get the .joblib files:**
1. Run all cells in `GP_AML_Khalid_Dharif_v2.ipynb` on Kaggle
2. In Kaggle, go to Output → Download the `.joblib` files
3. Create a `models/` folder in your project and paste all 9 files there

### Step 2.2 — Install Dependencies
```bash
pip install -r requirements.txt
```
This will take 2–5 minutes. You'll see packages downloading.

### Step 2.3 — Test the App Locally
```bash
streamlit run app.py
```
Your browser should open automatically at **http://localhost:8501**
- Try scoring a transaction
- Make sure the gauge chart appears
- Test all 4 sidebar pages

If it works → you're ready to deploy! 🎉

---

## PART 3 — Push to GitHub

### Step 3.1 — Create a .gitignore File
Create a new file called `.gitignore` in your project root and paste this:
```
# Python
venv/
__pycache__/
*.pyc
*.pyo
.env

# VSCode
.vscode/

# Jupyter checkpoints
.ipynb_checkpoints/

# OS files
.DS_Store
Thumbs.db
```

### Step 3.2 — Initialise Git
In your VSCode terminal:
```bash
git init
git add .
git commit -m "Initial commit: AML fraud detection app"
```

You should see output listing all the files being committed.

### Step 3.3 — Create a GitHub Repository
1. Go to https://github.com/new
2. Repository name: `aml-fraud-detection`
3. Description: `AML Fraud Detection using ML - Graduation Project`
4. Set to **Public** (required for free Streamlit Cloud deployment)
5. Do NOT check "Add README" (you already have one)
6. Click **Create Repository**

### Step 3.4 — Connect Local Repo to GitHub
GitHub will show you commands. Use these (replace YOUR_USERNAME):
```bash
git remote add origin https://github.com/YOUR_USERNAME/aml-fraud-detection.git
git branch -M main
git push -u origin main
```

When prompted, enter your GitHub username and password.

> ⚠️ **GitHub no longer accepts passwords — use a Personal Access Token:**
> 1. GitHub → Settings → Developer Settings → Personal Access Tokens → Tokens (classic)
> 2. Click "Generate new token" → check `repo` scope → Generate
> 3. Copy the token and use it as your password when prompted

### Step 3.5 — Verify on GitHub
Go to `https://github.com/YOUR_USERNAME/aml-fraud-detection`
You should see all your files listed there.

---

## PART 4 — Deploy on Streamlit Community Cloud

### Step 4.1 — Log in to Streamlit Cloud
1. Go to https://streamlit.io/cloud
2. Click **Sign in** → **Continue with GitHub**
3. Authorise Streamlit to access your GitHub account

### Step 4.2 — Create a New App
1. Click **New app** (top right)
2. Fill in the form:
   - **Repository:** `YOUR_USERNAME/aml-fraud-detection`
   - **Branch:** `main`
   - **Main file path:** `app.py`
   - **App URL:** choose a custom subdomain like `aml-fraud-khalid`
3. Click **Deploy!**

### Step 4.3 — Wait for Deployment
Streamlit Cloud will:
1. Clone your repository
2. Install packages from `requirements.txt` (takes 3–8 minutes)
3. Start your app

You'll see a build log. When it says "Your app is live!" you're done.

### Step 4.4 — Access Your Live App
Your app is now live at:
```
https://aml-fraud-khalid.streamlit.app
```
(or whatever subdomain you chose)

Share this link in your graduation project! 🎓

---

## PART 5 — Update the App (Future Changes)

Whenever you make changes to `app.py` or any project file:
```bash
# In VSCode terminal (with venv activated)
git add .
git commit -m "Describe what you changed"
git push
```

Streamlit Cloud automatically detects the push and redeploys your app within ~1 minute.

---

## PART 6 — Handling Large Model Files (If Needed)

If your `.joblib` files are larger than 100MB, GitHub will reject them.
Solution: use **Git Large File Storage (LFS)**:
```bash
# Install Git LFS (once)
git lfs install

# Track large files
git lfs track "*.joblib"

# This creates a .gitattributes file — add and commit it
git add .gitattributes
git add models/
git commit -m "Add model files via Git LFS"
git push
```

---

## PART 7 — VSCode Tips for This Project

### Useful Keyboard Shortcuts
| Shortcut | Action |
|----------|--------|
| `Ctrl + ~` | Open integrated terminal |
| `Ctrl + Shift + P` | Command palette |
| `Ctrl + Shift + X` | Extensions panel |
| `Ctrl + Z` | Undo |
| `Ctrl + /` | Comment/uncomment line |
| `F5` | Run Python file in debugger |

### Source Control Panel (Git GUI in VSCode)
Instead of typing git commands, you can use VSCode's built-in Git panel:
1. Click the **Source Control** icon in the left sidebar (branching icon)
2. You'll see changed files listed
3. Click **+** next to a file to stage it (= `git add`)
4. Type a commit message and click **✓ Commit** (= `git commit`)
5. Click the **...** menu → Push (= `git push`)

This is identical to typing the commands but visual!

---

## Troubleshooting

### "ModuleNotFoundError" when running app
```bash
# Make sure venv is activated
venv\Scripts\activate   # Windows
source venv/bin/activate  # Mac/Linux

# Reinstall requirements
pip install -r requirements.txt
```

### "FileNotFoundError: models/aml_best_model.joblib"
- Make sure the `models/` folder exists inside your project
- Make sure all 9 `.joblib` files are inside `models/`
- Make sure `models/` is committed to GitHub (check on github.com)

### Streamlit app crashes on Cloud but works locally
- Check the build log on Streamlit Cloud for error messages
- Most common cause: missing package in `requirements.txt`
- Add the missing package and push again

### Port 8501 already in use (local)
```bash
# Run on a different port
streamlit run app.py --server.port 8502
```

---

## Final Checklist Before Submission

- [ ] Notebook runs end-to-end on Kaggle without errors
- [ ] All 9 `.joblib` files are in the `models/` folder
- [ ] `streamlit run app.py` works locally
- [ ] All 4 pages of the app work (Single Transaction, Batch, Model Info, About)
- [ ] Repository is on GitHub and set to Public
- [ ] Streamlit Cloud deployment is live
- [ ] Live URL is included in your GP report

---

*Guide prepared for: Khalid Dharif — AML Fraud Detection Graduation Project*
