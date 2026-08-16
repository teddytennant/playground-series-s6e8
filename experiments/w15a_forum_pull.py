"""Pull the whole competition discussion forum — topics, bodies and comments.

The public `kaggle` CLI has no discussion subcommand and the web page is a JS shell with no
server-side rendering, so five days of runs in this workspace never read the forum.  The
route that works is the SDK the CLI is built on.

RUN WITH THE CLI'S INTERPRETER, NOT .venv:
    /home/nixos/.local/share/uv/tools/kaggle/bin/python experiments/w15a_forum_pull.py

Writes experiments/w15a_forum/topics.json and topics_full.json.
Does NOT work: DiscussionApiClient.list_topics(forum_slug=<competition>) -> 403 (global
forums only); list_team_public_submissions(team_id=<other team>) -> 401.
"""
from __future__ import annotations

import html
import json
import os
import re
import time

from kagglesdk import KaggleClient
from kagglesdk.competitions.types.competition_api_service import (
    ApiListCompetitionTopicsRequest, ApiListTopicMessagesRequest)
from kagglesdk.discussions.types.discussions_api_service import ApiGetTopicRequest
from kagglesdk.discussions.types.discussions_enums import TopicListSortBy

COMP = "playground-series-s6e8"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "w15a_forum")
SORTS = [TopicListSortBy.TOPIC_LIST_SORT_BY_NEW, TopicListSortBy.TOPIC_LIST_SORT_BY_TOP,
         TopicListSortBy.TOPIC_LIST_SORT_BY_HOT, TopicListSortBy.TOPIC_LIST_SORT_BY_RECENT,
         TopicListSortBy.TOPIC_LIST_SORT_BY_ACTIVE]


def strip(s):
    return html.unescape(re.sub(r"<[^>]+>", " ", str(s or "")))


def main():
    os.makedirs(OUT, exist_ok=True)
    seen = {}
    with KaggleClient() as k:
        # no single sort returns everything -- sweep all five and dedupe on topic id
        for sort in SORTS:
            for page in range(1, 8):
                r = ApiListCompetitionTopicsRequest()
                r.competition_name, r.sort_by, r.page = COMP, sort, page
                try:
                    ts = k.competitions.competition_api_client.list_competition_topics(r).topics
                except Exception as e:
                    print("ERR", sort, page, str(e)[:80])
                    break
                if not ts:
                    break
                for t in ts:
                    seen[t.id] = dict(id=t.id, title=t.title, votes=t.votes,
                                      comments=getattr(t, "comment_count", None),
                                      author=getattr(t, "author_name", None),
                                      date=str(t.post_date),
                                      url=getattr(t, "url", None))
                time.sleep(0.2)
        json.dump(list(seen.values()), open(os.path.join(OUT, "topics.json"), "w"), indent=1)
        print(f"{len(seen)} topics")

        full = []
        for t in seen.values():
            rec = dict(t)
            try:
                g = ApiGetTopicRequest()
                g.id = t["id"]
                rec["content"] = k.discussions.discussion_api_client.get_topic(g).topic.content
            except Exception as e:
                rec["content"] = "ERR " + str(e)[:80]
            msgs = []
            try:
                m = ApiListTopicMessagesRequest()
                m.competition_name, m.topic_id, m.page_size = COMP, t["id"], 100
                for x in (k.competitions.competition_api_client.list_topic_messages(m).messages or []):
                    msgs.append(dict(author=getattr(x, "author_name", None),
                                     date=str(getattr(x, "post_date", None)),
                                     votes=getattr(x, "votes", None),
                                     text=getattr(x, "content", None)))
            except Exception as e:
                msgs = [{"err": str(e)[:100]}]
            rec["messages"] = msgs
            full.append(rec)
            time.sleep(0.15)

    json.dump(full, open(os.path.join(OUT, "topics_full.json"), "w"), indent=1)
    print(f"{len(full)} bodies, {sum(len(f['messages']) for f in full)} comments -> {OUT}")

    for t in sorted(full, key=lambda x: x["date"], reverse=True):
        print(f"{t['id']} v{t['votes']:>3} {t['date'][:16]}  {t['title'][:90]}")


if __name__ == "__main__":
    main()
