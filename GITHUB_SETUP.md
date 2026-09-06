# GitHub Setup & Push Guide

Your local repository is ready to push to GitHub. Follow these steps:

---

## Step 1: Create GitHub Repository

1. Go to https://github.com/new
2. Create a **new repository** with these settings:

   **Repository name:** `insurance-claim-predictor`  
   **Description:** Machine learning pipeline for predicting hurricane insurance claim amounts  
   **Visibility:** Public (or Private if preferred)  
   ✓ DO NOT initialize with README (we have one)  
   ✓ DO NOT initialize with .gitignore (we have one)  

3. Click **Create repository**

---

## Step 2: Connect Local Repository to GitHub

After creating the repository, GitHub will show you commands. Run these in your terminal:

```bash
cd "C:\Users\Brian\Desktop\Claude Projects\Insurance Project"

# Add remote (replace YOUR_USERNAME with your GitHub username)
git remote add origin https://github.com/YOUR_USERNAME/insurance-claim-predictor.git

# Verify the remote was added
git remote -v
```

**Output should be:**
```
origin  https://github.com/YOUR_USERNAME/insurance-claim-predictor.git (fetch)
origin  https://github.com/YOUR_USERNAME/insurance-claim-predictor.git (push)
```

---

## Step 3: Push to GitHub

### First Time Push
```bash
cd "C:\Users\Brian\Desktop\Claude Projects\Insurance Project"

# Push to GitHub (creates 'main' branch on remote)
git push -u origin main
```

You'll be prompted to authenticate. Choose one:

**Option A: Web Browser (Recommended)**
- GitHub will open your browser
- Authenticate with your GitHub account
- Approve the connection

**Option B: Personal Access Token**
- Go to https://github.com/settings/tokens
- Create a "repo" token
- Paste token when prompted for password

**Option C: SSH Key**
- Set up SSH keys: https://docs.github.com/en/authentication/connecting-to-github-with-ssh
- Use SSH URL instead: `git@github.com:YOUR_USERNAME/insurance-claim-predictor.git`

### Verify Push Succeeded
```bash
git log --oneline
git remote -v
```

Visit your GitHub repository URL to confirm files are there.

---

## Step 4: Configure Local Settings (Optional)

Set your global Git config (one-time):
```bash
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"
```

Or configure just this project:
```bash
git config user.name "Your Name"
git config user.email "your.email@example.com"
```

---

## Future Commits & Pushes

After making changes locally:

```bash
# See what changed
git status

# Stage changes
git add .

# Create commit
git commit -m "Your commit message"

# Push to GitHub
git push origin main
```

Or combine:
```bash
git add . && git commit -m "Message" && git push origin main
```

---

## Example: Push and Update

```bash
# 1. Make your changes
# (edit files, add features, etc)

# 2. Check status
git status

# 3. Stage changes
git add .

# 4. Commit with descriptive message
git commit -m "Add SHAP analysis and prediction explanations"

# 5. Push to GitHub
git push origin main

# 6. Verify on GitHub website
# Visit https://github.com/YOUR_USERNAME/insurance-claim-predictor
```

---

## Handling Large Files

**Note:** Your CSVs are ~47 MB. GitHub's limit is 100 MB per file.

### Option 1: Keep Large Files (Current)
Files under 100 MB work fine. No changes needed.

### Option 2: Use Git LFS (Large File Storage)
If future data exceeds 100 MB:

```bash
# Install Git LFS
# Windows: https://git-lfs.github.com/

# Track large files
git lfs track "*.csv"
git add .gitattributes
git commit -m "Configure Git LFS for CSV files"
git push origin main
```

### Option 3: Exclude from Git
Add to `.gitignore` to exclude from tracking:
```
# .gitignore
insurance_day1_clean.csv
insurance_day2_predictions.csv
checkpoint_*.csv
```

Then remove from git:
```bash
git rm --cached insurance_day1_clean.csv
git commit -m "Remove large CSV from tracking"
git push origin main
```

**Recommendation:** Keep current setup (CSVs tracked) since files are under 100 MB limit.

---

## Troubleshooting

### "fatal: not a git repository"
You're not in the project directory. Run:
```bash
cd "C:\Users\Brian\Desktop\Claude Projects\Insurance Project"
```

### "remote already exists"
Remote already configured. Skip the `git remote add` step.

### Authentication failed
- Ensure GitHub account is active
- Check username is correct
- Reset personal access token at https://github.com/settings/tokens

### "Updates were rejected"
Remote has commits you don't have locally:
```bash
git pull origin main --no-ff
```

Then push:
```bash
git push origin main
```

### Large file warning
Your CSVs are 47 MB (under 100 MB limit). No action needed.

---

## Next: Sharing & Collaboration

Once pushed to GitHub:

### Share Repository URL
```
https://github.com/YOUR_USERNAME/insurance-claim-predictor
```

### Add Collaborators
1. Go to Settings → Collaborators
2. Invite GitHub usernames
3. They can clone and contribute

### Clone for Others
```bash
git clone https://github.com/YOUR_USERNAME/insurance-claim-predictor.git
cd insurance-claim-predictor
pip install -r requirements.txt
python insurance_day1_fast.py
python insurance_day2_improved.py
```

---

## Repository Structure (Remote)

After pushing, your GitHub repo will have:

```
insurance-claim-predictor/
├── README.md                       [Visible on GitHub home]
├── .gitignore
├── GITHUB_SETUP.md                 [This file]
├── insurance_day1_clean.csv        [47 MB data]
├── insurance_day2_predictions.csv  [26 MB predictions]
├── *.py files                      [Scripts]
├── *.png files                     [Visualizations]
└── *.md files                      [Documentation]
```

---

## Quick Commands Reference

```bash
# Clone repository
git clone https://github.com/YOUR_USERNAME/insurance-claim-predictor.git

# Check status
git status

# See changes
git diff

# Stage all
git add .

# Commit
git commit -m "Your message"

# Push
git push origin main

# Pull latest
git pull origin main

# View log
git log --oneline

# Create branch
git checkout -b feature/new-feature

# Switch branch
git checkout main

# Merge branch
git merge feature/new-feature

# Delete branch
git branch -d feature/new-feature
```

---

## Success Checklist

- [ ] Created GitHub repository
- [ ] Added remote: `git remote add origin ...`
- [ ] Pushed code: `git push -u origin main`
- [ ] Verified files on GitHub.com
- [ ] Updated any references (documentation, emails, etc)
- [ ] Shared URL with team/stakeholders

---

**Status:** Ready to push to GitHub  
**Local commits:** 2 (initial + README)  
**Files staged:** 37  
**Total size:** ~47 MB  

**Next:** Run `git push -u origin main` with your GitHub repository URL!
