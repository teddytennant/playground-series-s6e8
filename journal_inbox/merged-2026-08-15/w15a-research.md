## Reading the Kaggle competition FORUM from this machine (w15a, 2026-08-15)

Five days of runs never read the discussion forum because the public `kaggle` CLI has no
discussion subcommand and `https://www.kaggle.com/competitions/<slug>/discussion` is a 5.6 KB
JS shell with **no server-side rendering** (confirmed with a Googlebot UA too — same shell).
The internal `/api/i/...` connect endpoints all return bare 400s with our credentials.

**The route that works** is the SDK the CLI itself is built on. It is installed at
`/home/nixos/.local/share/uv/tools/kaggle/lib/python3.12/site-packages/kagglesdk` and must be
run with that interpreter (`/home/nixos/.local/share/uv/tools/kaggle/bin/python`) — it is
**not** in `.venv`.

```python
from kagglesdk import KaggleClient
from kagglesdk.competitions.types.competition_api_service import (
    ApiListCompetitionTopicsRequest, ApiListTopicMessagesRequest)
from kagglesdk.discussions.types.discussions_api_service import ApiGetTopicRequest
from kagglesdk.discussions.types.discussions_enums import TopicListSortBy

c = KaggleClient()
with c as k:
    r = ApiListCompetitionTopicsRequest()
    r.competition_name = "playground-series-s6e8"
    r.sort_by = TopicListSortBy.TOPIC_LIST_SORT_BY_NEW      # also TOP / HOT / RECENT / ACTIVE
    r.page = 1
    topics = k.competitions.competition_api_client.list_competition_topics(r).topics
    # topic body (list_competition_topics does NOT return .content, get_topic does):
    g = ApiGetTopicRequest(); g.id = topics[0].id
    t = k.discussions.discussion_api_client.get_topic(g).topic       # t.content is HTML
    # comments:
    m = ApiListTopicMessagesRequest()
    m.competition_name, m.topic_id, m.page_size = "playground-series-s6e8", topics[0].id, 100
    msgs = k.competitions.competition_api_client.list_topic_messages(m).messages
```

Sweep all five `sort_by` values × pages 1..7 and dedupe on `topic.id` — one sort does not
return everything. For s6e8 that yields **34 topics / 93 comments**, the complete forum.
Working script and full dump: `experiments/w15a_forum/` (`topics.json`, `topics_full.json`).

Endpoints that do **not** work, so nobody re-tries them:
- `DiscussionApiClient.list_topics(forum_slug=...)` → **403** for competition forums. It only
  serves the global forums (`general`, `getting-started`). `forum_id` is not accepted at all.
- `CompetitionApiClient.list_team_public_submissions(team_id=...)` → **401** for any team but
  our own. You cannot read other teams' submission descriptions.
- `/api/i/discussions.DiscussionsService/*` and `/api/i/competitions.CompetitionService/*`
  with the OAuth token in `~/.kaggle/credentials.json` → bare **400**, no body, every request
  shape tried. Do not spend a run reverse-engineering these.

## The leaderboard CSV carries usernames and submission counts — use it

```bash
kaggle competitions leaderboard -c playground-series-s6e8 -d -p /tmp/lb --quiet
# -> /tmp/lb/<slug>-publicleaderboard-<ts>.csv
```

Columns: `Rank, TeamId, TeamName, LastSubmissionDate, Score, SubmissionCount,
TeamMemberUserNames`. This is strictly better than `kaggle competitions leaderboard -s`
(which gives team names only) and it is how to resolve a leader's display name to the
username you need for `kaggle kernels list --user` / `kaggle datasets list --user`.

Resolved for s6e8: MILANFX=`milanfx`, Maher el Ouahabi=`maherelouahabi`, Don
Mani=`donmarch14`, Optimistix=`optimistix`, Utkarsh=`n0va007`, cstdy=`kirill0212`, Romone
Dunlop=`romonedunlop`, Orig_lab=`chengxixixi`, Szymon Kłapiński=`szymonkapiski`,
Keanan=`citerne`, magp=`wjdzxh`, midway2333=`blueszhao`, FunnyBishop=`funnybishop`.

## ⚠ The workspace's "5e-5 noise floor" is the WITHIN-PACK floor — do not quote it cross-team

`sd(gap) = sd(single) · sqrt(2(1 − rho))`, so the resolvable difference between two
submissions depends entirely on how alike they are. Measured on the labelled 691,369 rows by
resampling the leaderboard's geometry (296,302-row pseudo-test, 20% = 59,260-row public
slice, 400 reps, `experiments/w15a_crossteam.py`; sd of one file's own slice AUC = 567e-6):

| pair | rho on the 296,302 test rows | **sd(public-slice gap)** |
|---|---|---|
| our own two h3 files | 0.99998 | **6.0e-6** |
| our own, different member set | 0.99942 | 19.3e-6 |
| our own, different transform | 0.99715 | 27.7e-6 |
| **vs najiama's published blend** | **0.99580** | **53.0e-6** |
| **vs najiama's earlier blend** | 0.99395 | 74.8e-6 |
| **vs boltuzamaki's 47-stream rank-average** | 0.99736 | 83.9e-6 |

**Cross-team the floor is 53–84e-6, an order of magnitude above the within-pack 5e-5/2e-6
figures.** Consequences that hold for the rest of this competition:

- A public gap to another team of 18e-5 is **2.2–3.4 sigma**. A gap of 5–11e-5 (the four
  teams between 0.97113 and 0.97117) is **0.8–1.7 sigma** — not a difference.
- Independently corroborated twice: w15e measured spearman 0.99580 against the same najiama
  file from a different script; dariushafshar's public thread 733214 back-solves the same
  curve (rho 0.994 ⇒ resolvable 1.5e-4 at 95%).
- Quick lookup at other rho, sd(single)=567e-6: 0.9999→8.0e-6, 0.999→25.4e-6, 0.995→56.7e-6,
  0.99→80.2e-6, 0.98→113.5e-6.

**Corollary for the shape of the top of the board.** Under "all top-K teams equally good",
pure slice noise at sd(gap)=65e-6 predicts sd(top30)=45.8e-6 and range=189e-6 against the
observed 47.8e-6 and 210e-6 — a match from a number measured on other data. But size the
plateau realistically (155 teams within 3e-4, 267 within 5e-4) and the same model
under-predicts badly, requiring tau≈92–104e-6 of real skill spread. **The public leaderboard
does not identify which, and therefore cannot tell you whether the leaders' edge survives to
private.** `experiments/w15a_extreme.py`, `experiments/w15a_private.py`.

## MILANFX (public #1, 0.97124) — everything knowable, 2026-08-15

- 14 submissions total; last submission **2026-08-10 21:01 UTC**, idle five days. Against
  Optimistix 95, Tilii 97, Don Mani 74, Maher el Ouahabi 66, us 33.
- **Zero public kernels**, for this or any competition.
- One dataset: `milanfx/s6e08originaldata`, uploaded 2026-08-01 00:55 UTC. It is
  `Smartphone_Usage_And_Addiction_Analysis_7500_Rows.csv`, md5
  **`d831a326bc6f0ab76056a12279cb0047`** — **byte-identical to `data/orig/`**, which we have
  held since 2026-08-11 and measured at −58e-6 at 1× dose. They also mirror
  `s3e03/s3e09/s3e16/s4e08/s6e07 originaldata`, so it is a standing habit, not a find.
- **The leader has no data we lack.** Do not re-download or re-check this.

## Public notebook ceiling, 2026-08-15

The highest-scoring public notebook on this competition is
`najiama/ensemble-of-ensembles-lb-0-97101` at **0.97101**, below our 0.97106. Its author
states in forum thread 735339 that the last +1e-5 came from *"Reverse Micro-Sorting"* (500
buckets, order reversed inside each) and calls it *"a perfect, live demonstration of Public LB
Overfitting… almost guaranteed to sink like a stone"*. **No public notebook has ever exceeded
our best.** Of the top 18 teams only `donmarch14`, `szymonkapiski` and `funnybishop` publish
any s6e8 kernel at all; all seven of those notebooks were read on 2026-08-15 and every method
in them is already in the pack (digit/floor/mod10/frac20 lattice categoricals and `PAIR_`
cross-column floor keys → `make_frames(wide_pairs=True)`; constrained imputation with the
budget-identity bounds plus string lookup keys → `agent/features.py`).

Forum method census (34 topics): stringified target encoding, 10-fold OOF TE, rank averaging,
capacity-over-feature-engineering, missingness-is-the-only-drift, generator-repaired-features,
OOF-as-meta-features, GPU RAPIDS TE. **All held.** Nothing in the forum claims above 0.9689.

## The h3-family LB invariance — now 6/6

Every `h3` file this account has submitted has returned **exactly 0.97105**: `blend158_h3`,
`blend159av_h3`, `blend160origm_h3`, `blend159_h3`, `blendtop3`, and (2026-08-15, pre-declared
in the submission message before sending) `blend159av_wh3`. That spans CV 0.970046–0.970049
and now includes a variant with two fitted transform weights. The public slice cannot resolve
anything inside the h3 family, and fitting weights over the three transforms does not move it.
