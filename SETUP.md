# Setup — about 10 minutes, one time

Everything below is the part I can't do for me. The code is written and tested; this
is the account plumbing.

## 1. Get a CFBD API key (2 min)

Go to **https://collegefootballdata.com/key**, enter your email, and the key arrives
immediately. Free tier: 1,000 API calls per month. The weekly job uses about 40, so
free is plenty for the archive. (When we pull four seasons of play-by-play in Phase 2,
that's a one-time burst — Tier 1 at $1/month covers it.)

## 2. Create an empty GitHub repo (2 min)

- github.com → **New repository** → name it `cfb-model`
- **Make it public.** Everything in it is public sports data, and the API key lives in
  Actions secrets, never in the repo. Public also means I can read the archive from my
  side without you managing an access token for me. If you'd rather it be private,
  that's fine — you'll just need to give me a fine-grained read-only PAT later.
- Do **not** initialize with a README (the files you're about to push include one).

## 3. Push the code (3 min)

Unzip `cfb-model.zip`, then from inside that folder:

```bash
git init
git add .
git commit -m "Phase 1: point-in-time snapshot archiver"
git branch -M main
git remote add origin https://github.com/<your-username>/cfb-model.git
git push -u origin main
```

## 4. Add the API key as a secret (1 min)

Repo → **Settings** → *Secrets and variables* → **Actions** → **New repository secret**

- Name: `CFBD_API_KEY`
- Value: the key from step 1

The name must match exactly — the workflow looks for that string.

## 5. Run it once by hand (2 min)

Repo → **Actions** tab → *Weekly CFB snapshot archive* → **Run workflow**.

Watch the log. The first step is a smoke test that hits every endpoint and prints
OK/FAIL per source. **Expect one or two FAILs on this first run** — I wrote the CFBD
endpoint paths from the v2 documentation without being able to call the live API from
my sandbox, so a path or parameter name may need a correction. That's exactly what the
smoke test is for. Paste me the log and I'll fix whatever it flags.

If it succeeds you'll see a new commit adding `archive/*.csv` and `archive/raw/`.

## 6. Send me the repo URL

I'll record it and switch on the weekly verification task, which checks every Tuesday
afternoon that the capture actually landed and tells you if it didn't.

---

## After that, nothing

The job runs itself Tuesdays at 8:00 AM Eastern, August through January. The one thing
worth acting on is a red run — GitHub emails you, and a red run on a critical source
(SP+, FPI, lines, Sonny Moore, Pi-Rate) means that week can't be recovered, so it's
worth re-running by hand the same day.
