"""Read a Kaggle competition's discussion forum. The `kaggle` CLI cannot do this.

Uses the internal JSON API with the OAuth access_token in ~/.kaggle/credentials.json.
No curl and no python3 on this box -- run with .venv/bin/python.

    .venv/bin/python experiments/w184b_forum.py playground-series-s6e8
    .venv/bin/python experiments/w184b_forum.py playground-series-s6e8 738592

With no topic id: prints competition facts (metric, daily cap, deadline) and the topic list.
With a topic id: prints that topic's writeup markdown.
"""
import json, ssl, sys, urllib.request

CA = '/etc/ssl/certs/ca-certificates.crt'   # venv has no certifi; default context fails
CTX = ssl.create_default_context(cafile=CA)
TOKEN = json.load(open('/home/nixos/.kaggle/credentials.json'))['access_token']


def post(service, body):
    req = urllib.request.Request(
        'https://www.kaggle.com/api/i/' + service,
        data=json.dumps(body).encode(),
        headers={'Authorization': 'Bearer ' + TOKEN,
                 'Content-Type': 'application/json',
                 'User-Agent': 'Mozilla/5.0'})
    return json.loads(urllib.request.urlopen(req, timeout=60, context=CTX).read().decode())


def competition(slug):
    return post('competitions.CompetitionService/GetCompetition', {'competitionName': slug})


def topics(forum_id):
    # Exactly this body. Adding sortBy or pageSize returns HTTP 400.
    return post('discussions.DiscussionsService/GetTopicListByForumId',
                {'forumId': forum_id}).get('topics', [])


def writeup(topic_id):
    t = post('discussions.DiscussionsService/GetForumTopicById',
             {'forumTopicId': topic_id})['forumTopic']
    w = t.get('writeUp')
    return t, (w['message']['rawMarkdown'] if w else None)


if __name__ == '__main__':
    slug = sys.argv[1] if len(sys.argv) > 1 else 'playground-series-s6e8'
    if len(sys.argv) > 2:
        t, md = writeup(int(sys.argv[2]))
        print(f"# {t['name']}\nby {t['authorUserDisplayName']}  votes {t['totalVotes']}")
        print(f"https://www.kaggle.com{t['url']}\n")
        print(md or '(no writeup body on this topic)')
        sys.exit(0)
    c = competition(slug)
    print(f"{c['title']}  ({slug})")
    print(f"  metric   {c['evaluationAlgorithm']['name']} (isMax={c['evaluationAlgorithm']['isMax']})")
    print(f"  daily    {c['maxDailySubmissions']}   scored subs {c['numScoredSubmissions']}")
    print(f"  deadline {c['deadline']}   teams {c['totalTeams']}")
    print(f"  public   {c['leaderboardPercentage']}% of {c['totalSolutionRows']} test rows")
    print(f"  forumId  {c['forumId']}\n")
    for t in topics(c['forumId']):
        print(f"  v{t.get('votes') or 0:>4} c{t.get('commentCount') or 0:>3} "
              f"id={t['id']:<8} {t['authorUser']['displayName'][:20]:<20} {t.get('title','')[:66]}")
