# Render Deployment Guide

Deploy your Streamlit dashboard to Render in 5 minutes.

---

## Prerequisites

✅ GitHub account (repo already pushed)  
✅ Render account (free at render.com)  
✅ Dashboard code in repo (`dashboard.py`, `requirements.txt`, `render.yaml`)

---

## Step 1: Connect GitHub to Render

1. Go to https://render.com
2. Click **"New +"** → **"Web Service"**
3. Select **"Connect a repository"**
4. Choose **"brianravelo28/insurance-claim-predictor"**
5. Grant Render permission to access GitHub

---

## Step 2: Configure Service

Render will auto-detect your settings from `render.yaml`, but here's manual setup:

| Setting | Value |
|---------|-------|
| **Name** | `insurance-dashboard` |
| **Environment** | `Python 3` |
| **Region** | `Oregon` (US) |
| **Plan** | `Free` (or upgrade to paid) |
| **Build Command** | `pip install -r requirements.txt` |
| **Start Command** | `streamlit run dashboard.py --server.port=10000 --server.address=0.0.0.0` |

---

## Step 3: Deploy

1. Click **"Deploy"**
2. Watch the build logs (takes 2-5 minutes)
3. Once green ✅, you get a public URL:

```
https://insurance-dashboard-xxxxx.onrender.com
```

---

## Step 4: Access Your Dashboard

Visit your Render URL to see the live dashboard!

**Features available:**
- 📊 Overview with KPIs
- 🔍 Claims explorer
- 🤖 Model analysis
- 📈 Deep analytics
- 📥 Download filtered data

---

## Troubleshooting

### "Build failed: ImportError"
- Check `requirements.txt` has all dependencies
- Rebuild: Click **"Deploy"** again

### "Application error: Failed to connect to service"
- Service may still be starting (wait 2-3 min)
- Check logs: Click **"View logs"**
- Restart: Click **"Restart instance"**

### "No such file or directory: insurance_day1_clean.csv"
- Data files not in repo
- Add them: `git add *.csv && git commit -m "Add CSV data" && git push`
- Redeploy on Render

### Dashboard loads but no data appears
- Files are large (47 MB) — may take time on free tier
- Check browser console for errors
- Try refreshing page

### "Port already in use"
- Render manages ports — shouldn't happen
- Check `render.yaml` uses port `10000`

---

## Production Optimization

### 1. Use Database Instead of CSVs (Later)

When ready to scale:

```python
# Replace data loading:
# @st.cache_data
# def load_data():
#     return pd.read_csv('...')

# With:
import psycopg2
def load_data():
    conn = psycopg2.connect(os.getenv('DATABASE_URL'))
    return pd.read_sql('SELECT * FROM claims', conn)
```

### 2. Add PostgreSQL Database

1. Render → **"New +"** → **"PostgreSQL"**
2. Copy connection string
3. Add to Render environment variables:
   ```
   DATABASE_URL = postgres://user:pass@host/db
   ```
4. Update dashboard.py to use database

### 3. Upgrade from Free Tier

Free tier limitations:
- ⚠️ Spins down after 15 min inactivity
- ⚠️ Cold starts take 30+ seconds
- ⚠️ Limited CPU/RAM

To remove:
- Render dashboard → **"Settings"**
- Scroll to **"Plan"** → Upgrade to **"Paid"**
- ~$7/month for starter tier

---

## Environment Variables

### Currently Used
None — all data from CSVs

### For Future (Database)
Add these in Render dashboard:

**Settings** → **Environment** → **Add from file**

```
DATABASE_URL=postgres://user:pass@host/db
STREAMLIT_LOGGER_LEVEL=error
```

---

## Custom Domain (Optional)

1. Render dashboard → **"Settings"**
2. Scroll to **"Custom Domain"**
3. Add your domain (e.g., `dashboard.yoursite.com`)
4. Update DNS records per Render instructions

---

## Monitoring & Logs

### View Logs
1. Render dashboard → **"Logs"**
2. See real-time application output

### Monitor Resource Usage
1. Render dashboard → **"Metrics"**
2. Track CPU, memory, disk usage

### Set Up Alerts
1. Render dashboard → **"Settings"**
2. Scroll to **"Notifications"**
3. Get alerts on deploy failures

---

## Update Dashboard Code

When you make changes:

```bash
# Commit changes locally
git add dashboard.py
git commit -m "Update dashboard features"
git push origin main
```

Render **auto-deploys** on push to `main`!

Or manually redeploy:
1. Render dashboard → **"Manual Deploy"**
2. Choose commit
3. **"Deploy"**

---

## Sharing Your Dashboard

### Public URL
```
https://insurance-dashboard-xxxxx.onrender.com
```

Share with:
- Team members
- Stakeholders
- Investors
- Collaborators

### Private Access (Premium)
Add password authentication (requires code changes):

```python
import streamlit as st

def check_password():
    if "password_correct" not in st.session_state:
        st.session_state.password_correct = False
    
    if not st.session_state.password_correct:
        password = st.text_input("Enter password:", type="password")
        if password == "your_secret_password":
            st.session_state.password_correct = True
        else:
            st.stop()

check_password()
# ... rest of dashboard code
```

---

## Useful Render Links

- **Dashboard:** https://render.com/dashboard
- **Docs:** https://render.com/docs
- **Pricing:** https://render.com/pricing
- **Support:** support@render.com

---

## Quick Commands

```bash
# After code changes, just push to GitHub
git add .
git commit -m "Update dashboard"
git push origin main

# Render auto-deploys within 1-2 minutes
# No additional commands needed!
```

---

## Performance Tips

1. **First load may be slow** (cold start on free tier)
   - Subsequent loads faster (cached data)
   - Upgrade to paid tier for always-on instances

2. **Filters are fast** (client-side filtering)
   - All 100k rows loaded, filtered in browser
   - No backend queries

3. **Export data is instant**
   - CSV generated and downloaded instantly
   - Uses Streamlit's built-in download

4. **Large datasets?**
   - Consider uploading to Render storage (paid)
   - Or migrate to database (PostgreSQL)

---

## Success Checklist

- [ ] GitHub repo has `dashboard.py`
- [ ] `requirements.txt` includes streamlit, plotly, pandas
- [ ] `render.yaml` configured (or manual setup done)
- [ ] CSV data files in repo root
- [ ] Render service created and deployed
- [ ] Dashboard accessible at public URL
- [ ] All tabs load without errors
- [ ] Filters work and data updates
- [ ] Download button works
- [ ] Share URL with team

---

## Next Steps After Deployment

1. **Test the dashboard** — Try all filters and tabs
2. **Share with stakeholders** — Send them the URL
3. **Gather feedback** — What features would help?
4. **Iterate** — Update code, push to GitHub, auto-redeploy
5. **Plan database migration** — When you need real-time data or more features

---

## Support

Having issues?

1. **Check Render logs** — Render dashboard → Logs
2. **Verify requirements.txt** — All packages up to date?
3. **Test locally first** — `streamlit run dashboard.py`
4. **Reach out** — support@render.com or GitHub issues

---

**Status:** Ready to deploy  
**Estimated time:** 5 minutes  
**Cost:** Free (or $7+/month for premium)  

🚀 **Let's go live!**
