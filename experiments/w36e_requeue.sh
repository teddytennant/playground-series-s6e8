#!/usr/bin/env bash
# w36e -- validate the w36 build and rebuild the send queue, so the 08-21 UTC day opens on a
# queue that can SEE the files this slot created.
#
# ⚠ w34d's warning applies verbatim and is the reason this exists: w26g_send.py sends the top
# of w26d_queueprice.csv, and a queue built before a file existed cannot send it. This slot
# created `w34_ad195std_wh3.csv` and `w34_ad195std_w.csv` (w36d) plus the whole w36b build.
#
# ⚠ The head of the queue must remain `w34_ad195stdcorr` — it is the CV leader, it is still
# UNSENT, and it is NOT SELECTABLE until it lands. Verify that with check_selection.py after.
cd "$(dirname "$0")/.."

echo "waiting for the w36b build to exit..."
while ! grep -q 'w36b done' experiments/w36b_build.log 2>/dev/null; do sleep 60; done
echo "w36b clear at $(date -u)"

echo "########## w28b_verify  $(date -u) ##########"
nice -n 15 .venv/bin/python experiments/w28b_verify.py
echo "########## requeue  $(date -u) ##########"
nice -n 15 .venv/bin/python experiments/w23b_sendqueue.py
nice -n 15 .venv/bin/python experiments/w26d_queueprice.py
echo "########## selection  $(date -u) ##########"
nice -n 15 .venv/bin/python experiments/check_selection.py | head -20
echo "########## w36e done  $(date -u) ##########"
