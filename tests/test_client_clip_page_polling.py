"""The clip page lets go of an analysis job it adopted, checked in the source.

The behaviour was checked in a browser against scripts/dev_web.py (a clip
marked as analysing, the page's own fetches intercepted to report the job as
ended): polling stopped and Reanalyze came back. There is no JavaScript test
runner, so this pins the two conditions that matter.

Background (bug hunt, 2026-09-19): a clip page that arrived while an analysis
was running adopts it. Nothing ever cleared an adopted job: the request that
started it is not the page's, so no response arrives, and a clip that keeps
its name sends no rename to follow. The page polled every two seconds for as
long as it stayed open, counting "Elapsed" upwards with Reanalyze disabled.
And the watch on a clip that is only queued for the recovery sweep ended for
good the first time the tab was hidden.
"""
from __future__ import annotations

import re
from pathlib import Path

DETAIL = (Path(__file__).resolve().parents[1] / "src" / "animaltracker" / "static" / "views" / "detail.js").read_text()


def test_an_adopted_job_ends_when_the_server_says_the_clip_is_free():
    poll = re.search(r"function pollJob\(\) \{.*?\n\}\n", DETAIL, re.S).group(0)
    ended = poll.index("S.job.adopted && payload && !payload.reprocessing")
    stop = poll.index("stopJobPolling();")
    assert "dropJob();" in poll[ended:stop]             # cleared first, so the stop below applies to it


def test_the_queued_watch_comes_back_with_the_tab():
    handler = re.search(r"function installVisibility\(\) \{.*?\n  \}\)\);", DETAIL, re.S).group(0)
    assert "S.job || (S.clip && S.clip.analysis === 'queued')" in handler


def test_every_way_a_job_ends_closes_its_progress_toast():
    # Bug hunt, 2026-09-22: progress toasts have no timeout, and the adopted
    # job's end, the 404 and the follow-the-rename path set S.job to null
    # without closing it, so "Reanalysis already running" spun for good.
    drop = re.search(r"function dropJob\(\) \{.*?\n\}\n", DETAIL, re.S).group(0)
    assert "S.job.toast.close()" in drop
    for name in ("pollJob", "followRenamedClip"):
        body = re.search(r"function %s\(.*?\n\}\n" % name, DETAIL, re.S).group(0)
        assert "S.job = null" not in body, name
