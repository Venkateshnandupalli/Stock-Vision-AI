# Setting Up Supabase for StockVision AI

Supabase is a free cloud-hosted PostgreSQL database — perfect for this project.
No installation required. Takes about 5 minutes.

---

## Step 1 — Create Free Supabase Account

1. Open your browser and go to: **https://supabase.com**
2. Click **"Start your project"**
3. Sign up with **GitHub** (recommended) or email
4. You'll land on the Supabase dashboard

---

## Step 2 — Create a New Project

1. Click **"New project"**
2. Fill in:
   - **Name:** `stockvision-ai`
   - **Database Password:** Choose a strong password (**save this — you'll need it!**)
   - **Region:** `Southeast Asia (Singapore)` — closest to India
3. Click **"Create new project"**
4. Wait ~2 minutes for the project to provision

---

## Step 3 — Get Your Connection String

1. In your project, go to: **Settings → Database**
2. Scroll down to **"Connection string"**
3. Select the **"URI"** tab
4. Copy the connection string. It looks like:
   ```
   postgresql://postgres.abcdefghijkl:YOUR_PASSWORD@aws-0-ap-south-1.pooler.supabase.com:5432/postgres
   ```

---

## Step 4 — Configure the Project

1. Copy `.env.example` to `.env`:
   ```
   copy .env.example .env
   ```

2. Open `.env` and paste your connection string:
   ```env
   DATABASE_URL=postgresql://postgres.abcdefghijkl:YOUR_PASSWORD@aws-0-ap-south-1.pooler.supabase.com:5432/postgres
   ```
   > **That's the only thing you need to change!**

---

## Step 5 — Initialize the Database

Run these commands from the project folder (activate venv first):

```bash
venv\Scripts\activate
pip install -r requirements.txt
```

Then run the database setup script:

```bash
python setup_database.py
```

This will:
- ✅ Connect to Supabase
- ✅ Create all 6 tables
- ✅ Create all 6 analytical views
- ✅ Insert seed data (all 11 stocks)
- ✅ Verify everything is working

---

## Step 6 — Run the Full Pipeline

```bash
# Download 5+ years of data and train models
python run_pipeline.py --full-reload

# Launch the dashboard
streamlit run dashboard/app.py
```

---

## Step 7 — Connect Power BI (Optional)

1. Open Power BI Desktop
2. **Get Data → PostgreSQL database**
3. Use these settings:
   - **Server:** `aws-0-ap-south-1.pooler.supabase.com`
   - **Database:** `postgres`
   - **Username:** `postgres.YOUR_PROJECT_REF`
   - **Password:** your database password
4. Connect to the analytical views:
   - `vw_daily_stock_performance`
   - `vw_stock_risk_summary`
   - `vw_sector_performance`
   - `vw_latest_predictions`
   - `vw_model_performance`

---

## Troubleshooting

| Error | Fix |
|---|---|
| `connection refused` | Check your DATABASE_URL is pasted correctly in .env |
| `password authentication failed` | Re-copy the URI from Supabase Dashboard |
| `SSL required` | Add `?sslmode=require` to end of DATABASE_URL |
| `timeout` | Try the "Session mode" connection string instead of "Transaction mode" |

---

## Supabase Free Tier Limits

| Limit | Value |
|---|---|
| Storage | 500 MB |
| Rows | Unlimited |
| Connections | 60 concurrent |
| Bandwidth | 5 GB/month |

This project uses approximately **50–100 MB** of storage — well within the free tier.
