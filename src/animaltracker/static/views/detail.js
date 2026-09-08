/* ============================================================================
   views/detail.js — CLIP DETAIL, route /app/clip/:path*

   One recording, everything the pipeline knows about it, and the four things
   an operator does to it: watch it, re-analyse it, download it, delete it.

   WHAT IS LOAD-BEARING HERE

   · The scrubber is not a progress bar. Above the bed sit one presence lane
     per track, coloured by species, so you can see WHEN each animal was in
     frame before you press play. Detection key frames are ticks on the bed.
   · Reanalyze is fire-and-poll, never a blocking request that ends in an
     alert(). POST /recordings/reprocess is a long request that resolves when
     the post-processor is finished; we do not await it for the UI. We show an
     indeterminate meter, an elapsed clock and a live inline log, and we poll
     GET /api/clip (its `reprocessing` flag) and GET /recordings/log for the
     sidecar as it lands. A 409 means somebody else already started the job —
     the server hands back that job's start time, so we adopt it and keep
     polling instead of showing an error.
   · Delete is deferred, not soft. The server has no undo window, so the UNDO
     TOAST holds the request: onExpire fires the DELETE, onUndo cancels it.
     The toast outlives this view on purpose — we navigate back to the grid
     immediately and the deadline keeps running in the shell.
   · Nothing here calls location.reload(); every refresh is a fetch diffed in.

   Exports exactly { mount(root, ctx), unmount() }.
   ========================================================================= */

import { h, clear, on, delegate, keyedList } from '../core/dom.js';
import { icon } from '../core/icons.js';
import { api } from '../core/api.js';
import { toast } from '../core/toast.js';
import { dialog, sheet, isOverlayOpen } from '../core/overlay.js';
import { router } from '../core/router.js';
import { store } from '../core/store.js';
import {
  clockTime, longDate, dayKey, fileSize, durationClock, speciesClass,
  isUnclassified, pct, confidenceSegments, joinMeta, filmClass, parseServerTime,
  plural, timeAgo
} from '../core/format.js';

/* Poll cadence while a reanalyze job is in flight. Slow enough that a 75-day
   archive server is not bothered, fast enough that "did it finish?" is never
   a question the user has to ask twice. */
var JOB_POLL_MS = 2000;
var TICK_MS = 1000;
var LANES = 3;                    /* .scrub__lanes is 12px tall @ 5px pitch */
var SEEK_STEP = 5;                /* arrow keys */
var PAGE_STEP = 30;               /* PageUp / PageDown */

/* Query keys that describe the filtered result set this clip was opened from.
   They ride along on every navigation so prev/next and Back stay inside the
   set the user was actually looking at. */
var FILTER_KEYS = ['camera', 'species', 'from', 'to', 'q', 'sort', 'view', 'year', 'month'];

var S = null;

/* ------------------------------------------------------------------ utils */

function freshState() {
  return {
    root: null,
    path: '',
    query: {},
    clip: null,
    log: null,              /* parsed sidecar data, or null */
    logMissing: false,
    logError: null,
    tracks: [],             /* normalised per-track model */
    duration: 0,
    fps: 15,
    activeTrack: -1,
    dragging: false,
    logOpen: false,
    neighbors: null,        /* { prev, next, index, total } */
    job: null,              /* { started, startedAt, lines[], done, adopted } */
    disposers: [],
    timers: [],
    intervals: [],
    overlays: [],
    jobLines: [],
    abort: (typeof AbortController === 'function') ? new AbortController() : null,
    mq: null,
    mqHandler: null,
    els: {},
    dead: false
  };
}

function keep(off) { if (S && typeof off === 'function') S.disposers.push(off); return off; }

function later(fn, ms) {
  var id = window.setTimeout(function () { fn(); }, ms);
  if (S) S.timers.push(id);
  return id;
}

function every(fn, ms) {
  var id = window.setInterval(fn, ms);
  if (S) S.intervals.push(id);
  return id;
}

function stopAllTimers() {
  if (!S) return;
  for (var i = 0; i < S.timers.length; i++) window.clearTimeout(S.timers[i]);
  for (var j = 0; j < S.intervals.length; j++) window.clearInterval(S.intervals[j]);
  S.timers = [];
  S.intervals = [];
}

function signal() { return S && S.abort ? S.abort.signal : undefined; }

function filterQuery(query) {
  var out = {};
  if (!query) return out;
  for (var i = 0; i < FILTER_KEYS.length; i++) {
    var k = FILTER_KEYS[i];
    if (query[k] !== undefined && query[k] !== '') out[k] = query[k];
  }
  return out;
}

function cameraName(id) {
  var list = store.get('cameras') || [];
  for (var i = 0; i < list.length; i++) {
    if (list[i].id === id) return list[i].name || list[i].id;
  }
  return id || 'unknown camera';
}

function clipHref(path, query) {
  return router.href('/clips/' + api.encodePath(path), query || {});
}

function num(v, fallback) {
  var n = Number(v);
  return isFinite(n) ? n : fallback;
}

/** A button with the app's icon+label anatomy. */
function btn(label, opts) {
  var o = opts || {};
  var cls = 'button.btn';
  if (o.variant) cls += '.btn--' + o.variant;
  if (o.size) cls += '.btn--' + o.size;
  var el = h(cls, {
    type: 'button',
    'aria-label': o.ariaLabel || null,
    disabled: !!o.disabled
  });
  if (o.icon) el.appendChild(icon(o.icon, { size: 'sm', 'class': 'btn__icon' }));
  el.appendChild(h('span.btn__spinner', h('span.spinner')));
  el.appendChild(h('span.btn__label', { text: label }));
  if (o.onClick) keep(on(el, 'click', o.onClick));
  return el;
}

function iconBtn(name, label, onClick, extraClass) {
  var el = h('button.icon-btn' + (extraClass ? '.' + extraClass : ''), {
    type: 'button',
    'aria-label': label,
    title: label
  }, icon(name));
  if (onClick) keep(on(el, 'click', onClick));
  return el;
}

function busy(el, isBusy) {
  if (!el) return;
  if (isBusy) el.setAttribute('aria-busy', 'true');
  else el.removeAttribute('aria-busy');
  el.disabled = !!isBusy;
}

/** A metadata row: label on the left, machine value on the right. */
function metaRow(label, value, valueClass) {
  return h('div.row.row--between',
    h('dt.t-xs', { text: label, style: { color: 'var(--c-text-3)' } }),
    h('dd.mono.t-xs', { text: value === null || value === undefined ? '—' : String(value),
      'class': valueClass || null,
      style: { margin: '0', textAlign: 'right', minWidth: '0', overflowWrap: 'anywhere' } }));
}

/* --------------------------------------------------------------- the model */

/**
 * Normalise the clip payload's thumbnails into the track model this view
 * draws everywhere: the scrubber lanes, the key-frame strip, the track chips
 * and the lightbox all read from this one array.
 */
function buildTracks(clip, logData) {
  var thumbs = (clip && clip.thumbnails) || [];
  var fps = num(clip && clip.fps, 15) || 15;
  var logTracks = {};
  if (logData && logData.tracking_summary && Array.isArray(logData.tracking_summary.tracks)) {
    var lt = logData.tracking_summary.tracks;
    for (var t = 0; t < lt.length; t++) {
      if (lt[t] && lt[t].track_id !== undefined) logTracks[String(lt[t].track_id)] = lt[t];
    }
  }

  var out = [];
  for (var i = 0; i < thumbs.length; i++) {
    var th = thumbs[i] || {};
    var start = num(th.start_time, null);
    var end = num(th.end_time, null);
    var dur = num(th.duration, null);
    if (start === null && dur !== null && end !== null) start = end - dur;
    if (end === null && start !== null && dur !== null) end = start + dur;
    var lg = th.track_id !== undefined ? logTracks[String(th.track_id)] : null;

    /* The sidecar knows the frame span; it is the only place a merged track
       announces itself, because the merged span is wider than the fragment
       the thumbnail was cut from. */
    var frames = null;
    var spanSeconds = null;
    if (lg && lg.first_frame !== undefined && lg.last_frame !== undefined) {
      frames = (num(lg.last_frame, 0) - num(lg.first_frame, 0)) + 1;
      spanSeconds = frames / fps;
      if (start === null) start = num(lg.first_frame, 0) / fps;
      if (end === null) end = num(lg.last_frame, 0) / fps;
    }

    out.push({
      key: th.track_id !== undefined ? 'track-' + th.track_id : 'thumb-' + i,
      index: num(th.track_index, i),
      trackId: th.track_id === undefined ? null : th.track_id,
      species: th.species || (clip && clip.species) || 'Unknown',
      url: th.url || (th.path ? api.thumbUrl(th.path) : ''),
      path: th.path || '',
      confidence: num(th.confidence, null),
      start: start === null ? 0 : Math.max(0, start),
      end: end === null ? null : end,
      duration: dur === null ? (end !== null && start !== null ? end - start : null) : dur,
      frames: frames,
      firstFrame: lg ? num(lg.first_frame, null) : null,
      lastFrame: lg ? num(lg.last_frame, null) : null,
      spanSeconds: spanSeconds,
      row: 0
    });
  }

  out.sort(function (a, b) { return a.start - b.start; });

  /* Greedy lane packing: a track goes on the lowest lane whose last track has
     already ended. Above LANES lanes we wrap, because the CSS bed is 12px. */
  var lastEnd = [];
  for (var k = 0; k < out.length; k++) {
    var tr = out[k];
    var placed = false;
    for (var lane = 0; lane < LANES; lane++) {
      if (lastEnd[lane] === undefined || tr.start >= lastEnd[lane] - 0.001) {
        tr.row = lane;
        lastEnd[lane] = tr.end === null ? tr.start : tr.end;
        placed = true;
        break;
      }
    }
    if (!placed) {
      tr.row = k % LANES;
      lastEnd[tr.row] = tr.end === null ? tr.start : tr.end;
    }
  }
  return out;
}

function fallbackDuration(tracks, logData, fps) {
  if (logData && logData.video && logData.video.frames && fps) {
    return num(logData.video.frames, 0) / fps;
  }
  var max = 0;
  for (var i = 0; i < tracks.length; i++) {
    var e = tracks[i].end === null ? tracks[i].start : tracks[i].end;
    if (e > max) max = e;
  }
  return max;
}

function bestConfidence(tracks) {
  var best = null;
  for (var i = 0; i < tracks.length; i++) {
    var c = tracks[i].confidence;
    if (c === null) continue;
    if (best === null || c > best) best = c;
  }
  return best;
}

/* ------------------------------------------------------------------ chrome */

function syncChrome() {
  var clip = S.clip;
  var title = clip ? (clip.species || 'Unclassified') : 'Clip';
  var subtitle = '';
  if (clip) {
    var day = dayKey(clip.time);
    subtitle = joinMeta(
      cameraName(clip.camera),
      day ? longDate(day) : '',
      clockTime(clip.time, { seconds: true }),
      fileSize(clip.size)
    );
  }
  store.setChrome({
    title: title,
    subtitle: subtitle,
    actions: S.els.chromeActions || [],
    toolbar: null,
    rail: null,
    norail: true,
    selbar: null,
    mods: []
  });
}

function buildChromeActions() {
  var back = iconBtn('arrow-left', 'Back to recordings', function () {
    router.go('/recordings', S.query);
  });
  var prev = iconBtn('chevron-left', 'Previous clip in this result set', function () { step(-1); });
  var next = iconBtn('chevron-right', 'Next clip in this result set', function () { step(1); });
  prev.disabled = true;
  next.disabled = true;
  S.els.prevBtn = prev;
  S.els.nextBtn = next;

  var more = iconBtn('more', 'More actions for this clip', function () {
    openOverflow(more);
  });
  S.els.moreBtn = more;
  S.els.chromeActions = [back, prev, next, more];
}

function openOverflow(anchor) {
  var clip = S.clip;
  var handle = sheet({
    title: 'Clip actions',
    snap: 'peek',
    items: [
      { label: 'Copy clip path', icon: 'layers', onSelect: function () { copyText(S.path, 'Clip path copied.'); } },
      { label: 'Copy direct link', icon: 'external', onSelect: function () {
        copyText(window.location.origin + clipHref(S.path, S.query), 'Link copied.');
      } },
      { label: 'Open raw file in a new tab', icon: 'film', onSelect: function () {
        window.open(api.clipUrl(S.path), '_blank', 'noopener');
      } },
      { label: 'Reanalyze with these settings', icon: 'refresh', disabled: !clip, onSelect: function () {
        startReanalyze(null);
      } },
      { label: 'Delete recording', icon: 'trash', danger: true, disabled: !clip, onSelect: function () {
        confirmDelete();
      } }
    ],
    initialFocus: null,
    onClose: function () { if (anchor && anchor.isConnected) anchor.focus(); }
  });

  /* Tracked so unmount() can tear it down: a route change with the sheet open
     must not leave a modal surface (and its scroll lock) behind. */
  S.overlays.push(handle);
  handle.result.then(function () {
    if (!S) return;
    var oi = S.overlays.indexOf(handle);
    if (oi >= 0) S.overlays.splice(oi, 1);
  });
}

function copyText(text, okMessage) {
  function fail(err) {
    toast.error('Could not copy to the clipboard.', {
      detail: (err && err.message) || 'The browser refused clipboard access; select the text and copy by hand.'
    });
  }
  try {
    if (navigator.clipboard && navigator.clipboard.writeText) {
      navigator.clipboard.writeText(text).then(function () {
        toast.success(okMessage);
      }, fail);
      return;
    }
  } catch (e) { /* falls through to the manual path below */ }
  fail(new Error('This browser exposes no clipboard API over plain HTTP.'));
}

/* ------------------------------------------------------------ prev / next */

function loadNeighbors() {
  var q = Object.assign({}, S.query, { limit: 500, offset: 0 });
  delete q.view;
  delete q.year;
  delete q.month;
  api.recordings(q, { signal: signal() }).then(function (payload) {
    if (!S || S.dead) return;
    var clips = (payload && payload.clips) || [];
    var idx = -1;
    for (var i = 0; i < clips.length; i++) {
      if (clips[i].path === S.path) { idx = i; break; }
    }
    if (idx < 0 && S.els.live) {
      S.els.live.textContent =
        'This clip is outside the first 500 results, so previous and next are unavailable.';
    }
    S.neighbors = {
      list: clips,
      index: idx,
      total: num(payload && payload.total, clips.length),
      prev: idx > 0 ? clips[idx - 1] : null,
      next: idx >= 0 && idx < clips.length - 1 ? clips[idx + 1] : null
    };
    paintNeighbors();
  }, function (err) {
    if (!S || S.dead || api.isAbort(err)) return;
    S.neighbors = null;
    paintNeighbors();
    /* Rule 8: a failed fetch is never silent. This one is non-fatal, so it
       gets one dismissible line rather than a persistent error. */
    toast.danger('Could not load the surrounding result set.', {
      detail: api.describe(err) + ' Previous and next are unavailable.'
    });
  });
}

function paintNeighbors() {
  var n = S.neighbors;
  var prev = S.els.prevBtn;
  var next = S.els.nextBtn;
  if (prev) {
    prev.disabled = !(n && n.prev);
    prev.setAttribute('aria-label', n && n.prev
      ? 'Previous clip: ' + (n.prev.species || 'Unclassified') + ', ' + clockTime(n.prev.time)
      : 'Previous clip in this result set');
  }
  if (next) {
    next.disabled = !(n && n.next);
    next.setAttribute('aria-label', n && n.next
      ? 'Next clip: ' + (n.next.species || 'Unclassified') + ', ' + clockTime(n.next.time)
      : 'Next clip in this result set');
  }
  if (S.els.position) {
    S.els.position.textContent = n && n.index >= 0
      ? (n.index + 1) + ' of ' + n.list.length
      : '';
  }
}

function step(dir) {
  var n = S.neighbors;
  if (!n) return;
  var target = dir < 0 ? n.prev : n.next;
  if (!target) {
    toast.info(dir < 0 ? 'This is the first clip in the set.' : 'This is the last clip in the set.');
    return;
  }
  router.navigate(clipHref(target.path, S.query));
}

/* ------------------------------------------------------------- the player */

function buildPlayer() {
  var clip = S.clip;
  var poster = clip && clip.thumbnail ? api.thumbUrl(clip.thumbnail) : '';

  var video = h('video.player__video', {
    preload: 'metadata',
    controls: false,
    tabIndex: -1,
    'aria-label': 'Recording of ' + (clip ? clip.species : 'a clip') + ' on ' + cameraName(clip && clip.camera)
  });
  video.playsInline = true;
  video.setAttribute('playsinline', '');
  video.setAttribute('webkit-playsinline', '');
  if (poster) video.setAttribute('poster', poster);
  /* setAttribute, not .src — it is the attribute that unmount() removes. */
  video.setAttribute('src', api.clipUrl(S.path));
  S.els.video = video;

  var playGlyph = h('span.player__play-glyph', icon('play', { size: 'lg' }));
  var playBtn = h('button.player__play', { type: 'button', 'aria-label': 'Play' }, playGlyph);
  S.els.playBtn = playBtn;
  S.els.playGlyph = playGlyph;

  var lanes = h('div.scrub__lanes');
  var marks = h('div.scrub__marks');
  var buffered = h('div.scrub__buffered');
  var played = h('div.scrub__played');
  var knob = h('div.scrub__knob');
  var scrub = h('div.scrub', {
    role: 'slider',
    tabIndex: 0,
    'aria-label': 'Seek within the recording',
    'aria-valuemin': '0',
    'aria-valuemax': '0',
    'aria-valuenow': '0',
    'aria-valuetext': '0:00 of 0:00'
  }, lanes, buffered, played, marks, knob);

  S.els.scrub = scrub;
  S.els.lanes = lanes;
  S.els.marks = marks;
  S.els.played = played;
  S.els.buffered = buffered;

  var now = h('b', { text: '0:00' });
  var total = h('span', { text: '0:00' });
  S.els.timeNow = now;
  S.els.timeTotal = total;

  /* The sprite has no speaker glyph, so the STATE is carried by the label and
     aria-pressed, never by the icon alone. */
  var muteBtn = iconBtn('live', 'Mute audio', null, 'icon-btn--on-media');
  muteBtn.setAttribute('aria-pressed', 'false');
  keep(on(muteBtn, 'click', function () {
    var v = S.els.video;
    if (!v) return;
    v.muted = !v.muted;
    muteBtn.setAttribute('aria-pressed', v.muted ? 'true' : 'false');
    muteBtn.setAttribute('aria-label', v.muted ? 'Unmute audio' : 'Mute audio');
    muteBtn.title = muteBtn.getAttribute('aria-label');
  }));

  var bar = h('div.player__bar',
    scrub,
    h('div.player__row',
      now,
      h('span', { text: '/' }),
      total,
      h('span.player__spacer'),
      h('span', { text: 'Space plays · ← → 5 s · , . frame' }),
      muteBtn));

  var notice = h('div.player__notice', { hidden: true, role: 'status' });
  S.els.notice = notice;

  var frame = h('div.frame.frame--flush' + (clip ? '.' + filmClass(clip.time) : ''),
    h('div.frame__film'));
  if (poster) {
    frame.appendChild(h('img.frame__img', {
      src: poster, alt: '', width: 640, height: 360,
      loading: 'eager', decoding: 'async'
    }));
  }

  var player = h('div.player.player--capped', frame, video, playBtn, bar, notice);
  S.els.player = player;
  wirePlayer();
  return player;
}

function wirePlayer() {
  var v = S.els.video;
  var player = S.els.player;

  keep(on(S.els.playBtn, 'click', function () { togglePlay(); }));

  keep(on(v, 'loadedmetadata', function () {
    if (isFinite(v.duration) && v.duration > 0) S.duration = v.duration;
    paintScrub();
    paintLanes();
  }));
  keep(on(v, 'timeupdate', function () {
    if (!S.dragging) paintScrub();
    syncActiveFromTime();
  }));
  keep(on(v, 'progress', paintBuffered));
  keep(on(v, 'play', function () {
    player.classList.add('player--playing');
    S.els.playBtn.setAttribute('aria-label', 'Pause');
    hideNotice();
  }));
  keep(on(v, 'pause', function () {
    player.classList.remove('player--playing');
    S.els.playBtn.setAttribute('aria-label', 'Play');
  }));
  keep(on(v, 'waiting', function () { showNotice('Buffering the clip from disk…'); }));
  keep(on(v, 'playing', hideNotice));
  keep(on(v, 'ended', function () {
    player.classList.remove('player--playing');
    S.els.playBtn.setAttribute('aria-label', 'Replay');
  }));
  keep(on(v, 'error', function () {
    var code = v.error ? v.error.code : 0;
    var why = code === 4
      ? 'The browser cannot decode this file — it may still be being written, or the codec is unsupported here.'
      : 'The server stopped sending the clip.';
    showNotice('Playback failed. ' + why);
    toast.error('Playback failed for this recording.', {
      detail: why + ' The file is still downloadable.',
      retry: function () {
        hideNotice();
        v.load();
      }
    });
  }));

  /* Pointer scrubbing. touch-action:none is already on .scrub in CSS. */
  var scrub = S.els.scrub;
  function fromEvent(ev) {
    var rect = scrub.getBoundingClientRect();
    if (!rect.width) return 0;
    var f = (ev.clientX - rect.left) / rect.width;
    return Math.max(0, Math.min(1, f));
  }
  keep(on(scrub, 'pointerdown', function (ev) {
    if (!duration()) return;
    S.dragging = true;
    try { scrub.setPointerCapture(ev.pointerId); } catch (e) {}
    seekTo(fromEvent(ev) * duration(), true);
    ev.preventDefault();
  }));
  keep(on(scrub, 'pointermove', function (ev) {
    if (!S.dragging) return;
    seekTo(fromEvent(ev) * duration(), true);
  }));
  function endDrag() {
    if (!S.dragging) return;
    S.dragging = false;
    seekTo(currentTime(), false);
  }
  keep(on(scrub, 'pointerup', endDrag));
  keep(on(scrub, 'pointercancel', endDrag));

  keep(on(scrub, 'keydown', function (ev) {
    var d = duration();
    if (!d) return;
    var handled = true;
    if (ev.key === 'ArrowRight' || ev.key === 'ArrowUp') {
      seekTo(currentTime() + (ev.shiftKey ? frameStep() : SEEK_STEP));
    } else if (ev.key === 'ArrowLeft' || ev.key === 'ArrowDown') {
      seekTo(currentTime() - (ev.shiftKey ? frameStep() : SEEK_STEP));
    } else if (ev.key === 'PageUp') {
      seekTo(currentTime() + PAGE_STEP);
    } else if (ev.key === 'PageDown') {
      seekTo(currentTime() - PAGE_STEP);
    } else if (ev.key === 'Home') {
      seekTo(0);
    } else if (ev.key === 'End') {
      seekTo(d);
    } else if (ev.key === ' ' || ev.key === 'Enter') {
      togglePlay();
    } else {
      handled = false;
    }
    if (handled) { ev.preventDefault(); ev.stopPropagation(); }
  }));
}

function duration() {
  var v = S.els.video;
  if (v && isFinite(v.duration) && v.duration > 0) return v.duration;
  return S.duration || 0;
}

function currentTime() {
  var v = S.els.video;
  return v && isFinite(v.currentTime) ? v.currentTime : 0;
}

function frameStep() {
  return 1 / (S.fps || 15);
}

function togglePlay() {
  var v = S.els.video;
  if (!v) return;
  if (v.paused) {
    var p = v.play();
    if (p && p.catch) {
      p.catch(function (err) {
        showNotice('The browser blocked playback. Press play again.');
        toast.danger('Playback did not start.', { detail: (err && err.message) || 'The browser blocked autoplay.' });
      });
    }
  } else {
    v.pause();
  }
}

function showNotice(text) {
  if (!S.els.notice) return;
  S.els.notice.textContent = text;
  S.els.notice.hidden = false;
}

function hideNotice() {
  if (!S.els.notice) return;
  S.els.notice.hidden = true;
  S.els.notice.textContent = '';
}

function seekTo(t, quiet) {
  var v = S.els.video;
  var d = duration();
  var clamped = Math.max(0, Math.min(d || 0, t));
  if (v) {
    try { v.currentTime = clamped; } catch (e) { /* seeking before metadata */ }
  }
  if (!quiet) hideNotice();
  paintScrub(clamped);
  syncActiveFromTime(clamped);
}

function paintScrub(forced) {
  var d = duration();
  var t = forced === undefined ? currentTime() : forced;
  var f = d ? Math.max(0, Math.min(1, t / d)) : 0;
  /* --played drives BOTH .scrub__played's width and .scrub__knob's left, so
     it lives on .scrub itself, not on either child. */
  if (S.els.scrub) S.els.scrub.style.setProperty('--played', (f * 100).toFixed(3) + '%');
  if (S.els.timeNow) S.els.timeNow.textContent = durationClock(t);
  if (S.els.timeTotal) S.els.timeTotal.textContent = durationClock(d);
  var scrub = S.els.scrub;
  if (scrub) {
    scrub.setAttribute('aria-valuemax', String(Math.round(d)));
    scrub.setAttribute('aria-valuenow', String(Math.round(t)));
    scrub.setAttribute('aria-valuetext', durationClock(t) + ' of ' + durationClock(d));
  }
  paintBuffered();
}

function paintBuffered() {
  var v = S.els.video;
  var scrub = S.els.scrub;
  if (!v || !scrub) return;
  var d = duration();
  if (!d || !v.buffered || !v.buffered.length) return;
  var end = 0;
  for (var i = 0; i < v.buffered.length; i++) {
    if (v.buffered.start(i) <= currentTime() + 0.01) end = Math.max(end, v.buffered.end(i));
  }
  scrub.style.setProperty('--buffered', ((end / d) * 100).toFixed(2) + '%');
}

/** Presence lanes + detection ticks. Reconciled, never wholesale replaced. */
function paintLanes() {
  var d = duration() || fallbackDuration(S.tracks, S.log, S.fps) || 1;
  var lanes = S.els.lanes;
  var marks = S.els.marks;
  if (!lanes || !marks) return;

  keyedList(lanes, S.tracks, {
    key: function (tr) { return tr.key; },
    create: function (tr) { return h('span.scrub__lane.' + speciesClass(tr.species)); },
    update: function (el, tr) {
      var start = Math.max(0, Math.min(1, tr.start / d));
      var end = tr.end === null ? start : Math.max(0, Math.min(1, tr.end / d));
      var len = Math.max(0.008, end - start);
      el.style.setProperty('--at', (start * 100).toFixed(3) + '%');
      el.style.setProperty('--len', (len * 100).toFixed(3) + '%');
      el.style.setProperty('--row', String(tr.row));
      el.classList.toggle('scrub__lane--muted',
        S.activeTrack >= 0 && S.tracks[S.activeTrack] !== tr);
    }
  });

  keyedList(marks, S.tracks, {
    key: function (tr) { return 'mark-' + tr.key; },
    create: function (tr) { return h('span.scrub__mark.' + speciesClass(tr.species)); },
    update: function (el, tr) {
      el.style.setProperty('--at', (Math.max(0, Math.min(1, tr.start / d)) * 100).toFixed(3) + '%');
    }
  });
}

/* -------------------------------------------------------- key-frame strip */

function buildKeystrip() {
  var strip = h('div.keystrip', {
    role: 'listbox',
    'aria-label': 'Key frames, one per detected track'
  });
  S.els.keystrip = strip;

  keep(delegate(strip, 'click', '.keystrip__frame', function (ev, el) {
    var i = Number(el.dataset.index);
    if (isFinite(i)) activateTrack(i, true);
  }));
  keep(delegate(strip, 'dblclick', '.keystrip__frame', function (ev, el) {
    var i = Number(el.dataset.index);
    if (isFinite(i)) openLightbox(i);
  }));
  keep(delegate(strip, 'keydown', '.keystrip__frame', function (ev, el) {
    var i = Number(el.dataset.index);
    if (!isFinite(i)) return;
    if (ev.key === 'ArrowRight') { focusFrame(i + 1); ev.preventDefault(); }
    else if (ev.key === 'ArrowLeft') { focusFrame(i - 1); ev.preventDefault(); }
    else if (ev.key === 'Home') { focusFrame(0); ev.preventDefault(); }
    else if (ev.key === 'End') { focusFrame(S.tracks.length - 1); ev.preventDefault(); }
    else if (ev.shiftKey && ev.key === 'Enter') { openLightbox(i); ev.preventDefault(); }
  }));

  return strip;
}

function focusFrame(i) {
  if (i < 0 || i >= S.tracks.length) return;
  var el = S.els.keystrip.querySelector('[data-index="' + i + '"]');
  if (el) el.focus();
}

function paintKeystrip() {
  var strip = S.els.keystrip;
  if (!strip) return;
  keyedList(strip, S.tracks, {
    key: function (tr) { return tr.key; },
    create: function (tr) {
      return h('button.keystrip__frame', { type: 'button', role: 'option' },
        h('span.frame.frame--sm', h('span.frame__film')),
        h('span.keystrip__t'));
    },
    update: function (el, tr, key, i) {
      el.dataset.index = String(i);
      el.setAttribute('aria-current', S.activeTrack === i ? 'true' : 'false');
      el.setAttribute('aria-selected', S.activeTrack === i ? 'true' : 'false');
      el.tabIndex = (S.activeTrack === i || (S.activeTrack < 0 && i === 0)) ? 0 : -1;

      var frame = el.firstElementChild;
      frame.className = 'frame frame--sm ' + (S.clip ? filmClass(S.clip.time) : 'film--night');
      var img = frame.querySelector('img');
      if (tr.url) {
        if (!img) {
          img = h('img.frame__img', { alt: '', width: 132, height: 74, loading: 'lazy', decoding: 'async' });
          frame.appendChild(img);
        }
        if (img.getAttribute('src') !== tr.url) img.setAttribute('src', tr.url);
        img.alt = 'Key frame for track ' + (tr.trackId === null ? (i + 1) : tr.trackId) +
          ', ' + tr.species;
      } else if (img) {
        frame.removeChild(img);
      }

      var caption = el.lastElementChild;
      caption.textContent = durationClock(tr.start) +
        (tr.end !== null ? '–' + durationClock(tr.end) : '');
      el.setAttribute('aria-label', tr.species + ', track ' +
        (tr.trackId === null ? (i + 1) : tr.trackId) + ', starts at ' + durationClock(tr.start) +
        '. Activates to seek the video here; shift+Enter enlarges the frame.');
    }
  });
}

/* -------------------------------------------------------------- track chips */

function buildTrackChips() {
  var wrap = h('div.trackchips', { role: 'group', 'aria-label': 'Detected tracks' });
  S.els.trackchips = wrap;
  keep(delegate(wrap, 'click', '.trackchip', function (ev, el) {
    var i = Number(el.dataset.index);
    if (isFinite(i)) activateTrack(i, true);
  }));
  return wrap;
}

function paintTrackChips() {
  var wrap = S.els.trackchips;
  if (!wrap) return;
  keyedList(wrap, S.tracks, {
    key: function (tr) { return tr.key; },
    create: function () {
      return h('button.trackchip', { type: 'button' },
        h('span.trackchip__id'),
        h('span.trackchip__sp'),
        h('span.trackchip__t'),
        h('span.trackchip__note'));
    },
    update: function (el, tr, key, i) {
      el.dataset.index = String(i);
      el.className = 'trackchip ' + speciesClass(tr.species);
      el.setAttribute('aria-pressed', S.activeTrack === i ? 'true' : 'false');

      var kids = el.children;
      kids[0].textContent = '#' + (tr.trackId === null ? (i + 1) : tr.trackId);
      kids[1].textContent = tr.species;
      kids[2].textContent = joinMeta(
        durationClock(tr.start) + (tr.end !== null ? '–' + durationClock(tr.end) : ''),
        tr.duration !== null ? tr.duration.toFixed(1) + ' s' : null,
        tr.confidence !== null ? pct(tr.confidence) : null
      );

      var note = '';
      if (tr.frames !== null) {
        note = plural(tr.frames, 'frame') + ' ' + tr.firstFrame + '–' + tr.lastFrame;
        /* A merged track's frame span is wider than the fragment the
           thumbnail was cut from — say so rather than leaving the gap
           looking like a bug. */
        if (tr.duration !== null && tr.spanSeconds !== null &&
            tr.spanSeconds > tr.duration + 0.5) {
          note += ' · merged from fragments by the post-processor, spanning ' +
            tr.spanSeconds.toFixed(1) + ' s';
        }
      }
      kids[3].textContent = note;
      kids[3].hidden = !note;

      el.setAttribute('aria-label', 'Seek to track ' +
        (tr.trackId === null ? (i + 1) : tr.trackId) + ', ' + tr.species +
        ', at ' + durationClock(tr.start));
    }
  });
}

function activateTrack(i, seek) {
  if (i < 0 || i >= S.tracks.length) return;
  S.activeTrack = i;
  if (seek) seekTo(S.tracks[i].start);
  paintKeystrip();
  paintTrackChips();
  paintLanes();
  var el = S.els.keystrip && S.els.keystrip.querySelector('[data-index="' + i + '"]');
  if (el && el.scrollIntoView) {
    try { el.scrollIntoView({ block: 'nearest', inline: 'nearest' }); } catch (e) {}
  }
}

function syncActiveFromTime(forced) {
  var t = forced === undefined ? currentTime() : forced;
  var found = -1;
  for (var i = 0; i < S.tracks.length; i++) {
    var tr = S.tracks[i];
    var end = tr.end === null ? tr.start + 0.5 : tr.end;
    if (t >= tr.start - 0.05 && t <= end + 0.05) { found = i; break; }
  }
  if (found === S.activeTrack) return;
  S.activeTrack = found;
  paintKeystrip();
  paintTrackChips();
  paintLanes();
}

/* ----------------------------------------------------------------- lightbox */

function openLightbox(startIndex) {
  if (!S.tracks.length) return;
  var index = Math.max(0, Math.min(S.tracks.length - 1, startIndex));
  var imgEl = null;
  var capEl = null;
  var counterEl = null;
  var handle = null;

  function paint() {
    if (!S || S.dead || !S.tracks[index]) return;
    var tr = S.tracks[index];
    if (imgEl) {
      imgEl.setAttribute('src', tr.url || '');
      imgEl.alt = 'Key frame for track ' + (tr.trackId === null ? (index + 1) : tr.trackId) +
        ', ' + tr.species + ', at ' + durationClock(tr.start);
    }
    if (capEl) {
      capEl.textContent = joinMeta(
        tr.species,
        'track ' + (tr.trackId === null ? (index + 1) : tr.trackId),
        durationClock(tr.start) + (tr.end !== null ? '–' + durationClock(tr.end) : ''),
        tr.confidence !== null ? pct(tr.confidence) + ' confidence' : null
      );
    }
    if (counterEl) counterEl.textContent = (index + 1) + ' of ' + S.tracks.length;
  }

  function move(delta) {
    if (!S || S.dead || !S.tracks.length) return;
    index = (index + delta + S.tracks.length) % S.tracks.length;
    paint();
  }

  handle = dialog({
    role: 'dialog',
    width: 1040,
    title: 'Key frame',
    dismissible: true,
    content: function (box) {
      var frame = h('div.frame', { style: { width: '100%' } }, h('div.frame__film'));
      imgEl = h('img.frame__img', {
        alt: '', decoding: 'async',
        style: { objectFit: 'contain', background: 'var(--c-surface-sunken)' }
      });
      frame.appendChild(imgEl);
      capEl = h('p.t-sm', { style: { margin: '0', color: 'var(--c-text-2)', overflowWrap: 'anywhere' } });
      counterEl = h('span.mono.t-xs', { style: { color: 'var(--c-text-3)' } });

      var prev = btn('Previous', { variant: 'secondary', icon: 'chevron-left',
        onClick: function () { move(-1); } });
      var next = btn('Next', { variant: 'secondary', icon: 'chevron-right',
        onClick: function () { move(1); } });
      var seekBtn = btn('Seek video here', { variant: 'primary', icon: 'play', onClick: function () {
        activateTrack(index, true);
        handle.close('seek');
      } });

      box.appendChild(frame);
      box.appendChild(capEl);
      box.appendChild(h('div.row.row--between',
        counterEl,
        h('div.btn-row', prev, next, seekBtn)));
      paint();
    },
    actions: [{ label: 'Close', variant: 'secondary', value: 'close' }]
  });

  var offKeys = on(handle.el, 'keydown', function (ev) {
    if (ev.key === 'ArrowRight') { move(1); ev.preventDefault(); }
    else if (ev.key === 'ArrowLeft') { move(-1); ev.preventDefault(); }
  });
  S.overlays.push(handle);
  handle.result.then(function () {
    offKeys();
    if (!S) return;
    var i = S.overlays.indexOf(handle);
    if (i >= 0) S.overlays.splice(i, 1);
  });
  keep(offKeys);
}

/* ------------------------------------------------------------ species panel */

function buildSpeciesPanel() {
  var name = h('h2.t-h2', { style: { margin: '0', overflowWrap: 'anywhere' } });
  var badge = h('span.badge.badge--lg',
    h('span.badge__dot'),
    h('span'),
    h('span.badge__conf'));
  var raw = h('p.mono.t-xs', { style: { margin: '0', color: 'var(--c-text-3)', overflowWrap: 'anywhere' } });
  var meterWrap = h('div.stack.stack--tight');
  var status = h('p.t-sm', {
    role: 'status', 'aria-live': 'polite',
    style: { margin: '0', color: 'var(--c-text-2)' }
  });
  var panel = h('section.stack.stack--tight', {
    'aria-label': 'Species classification'
  },
    h('span.overline.overline--strong', { text: 'Species' }),
    name, badge, raw, meterWrap, status);

  S.els.spBadge = badge;
  S.els.spName = name;
  S.els.spRaw = raw;
  S.els.spMeters = meterWrap;
  S.els.spStatus = status;
  S.els.spPanel = panel;
  return panel;
}

function paintSpeciesPanel() {
  var clip = S.clip;
  if (!clip || !S.els.spPanel) return;
  var species = clip.species || 'Unclassified';
  S.els.spPanel.className = 'stack stack--tight ' + speciesClass(species);
  S.els.spName.textContent = species;
  S.els.spName.style.fontStyle = isUnclassified(species) ? 'italic' : 'normal';
  S.els.spRaw.textContent = clip.raw_species || '';
  S.els.spRaw.hidden = !clip.raw_species;

  /* Colour is never the only channel: the badge carries the species NAME and,
     below 0.5, the dashed --low treatment plus the number in full. */
  var best = bestConfidence(S.tracks);
  var badge = S.els.spBadge;
  badge.className = 'badge badge--lg ' + speciesClass(species) +
    (isUnclassified(species) ? ' badge--unclassified' : '') +
    (best !== null && best < 0.5 ? ' badge--low' : '');
  badge.children[1].textContent = species;
  badge.children[2].textContent = best === null ? 'no score' : pct(best);
  badge.setAttribute('aria-label', species +
    (best === null ? ', confidence unavailable' : ', ' + pct(best) + ' confidence') +
    (best !== null && best < 0.5 ? ', below the review threshold' : ''));

  /* One labelled meter per track — this is the distribution the pipeline
     actually recorded for this clip. */
  var rows = S.tracks.slice(0).sort(function (a, b) {
    return (b.confidence === null ? -1 : b.confidence) - (a.confidence === null ? -1 : a.confidence);
  }).slice(0, 3);

  keyedList(S.els.spMeters, rows, {
    key: function (tr) { return 'conf-' + tr.key; },
    create: function () {
      return h('div.stack.stack--tight',
        h('div.row.row--between',
          h('span.t-xs', { style: { color: 'var(--c-text-2)', overflowWrap: 'anywhere' } }),
          h('span.mono.t-xs', { style: { color: 'var(--c-text-3)' } })),
        h('div.meter.meter--accent', { role: 'img' }));
    },
    update: function (el, tr, key, i) {
      var head = el.firstElementChild;
      head.children[0].textContent = tr.species +
        ' · track ' + (tr.trackId === null ? (i + 1) : tr.trackId);
      head.children[1].textContent = tr.confidence === null ? 'n/a' : tr.confidence.toFixed(2);
      var meter = el.lastElementChild;
      var n = tr.confidence === null ? 0 : confidenceSegments(tr.confidence, 8);
      meter.style.setProperty('--n', String(n));
      meter.style.setProperty('--of', '8');
      meter.classList.remove('meter--indeterminate');
      meter.setAttribute('aria-label', tr.species + ' confidence ' +
        (tr.confidence === null ? 'unavailable' : pct(tr.confidence)));
    }
  });

  paintJobState();
}

/** The processing state: an indeterminate meter, an elapsed clock, live text. */
function paintJobState() {
  var st = S.els.spStatus;
  if (!st) return;
  var job = S.job;
  var meters = S.els.spMeters;

  if (!job) {
    st.textContent = S.clip && S.clip.reprocessing
      ? 'The post-processor is working on this clip right now.'
      : '';
    st.hidden = !st.textContent;
    if (S.els.jobMeter && S.els.jobMeter.parentNode) {
      S.els.jobMeter.parentNode.removeChild(S.els.jobMeter);
      S.els.jobMeter = null;
    }
    busy(S.els.reanalyzeBtn, false);
    return;
  }

  if (!S.els.jobMeter) {
    S.els.jobMeter = h('div.meter.meter--tall.meter--indeterminate.meter--accent', {
      role: 'progressbar',
      'aria-label': 'Reanalysis in progress'
    });
    meters.parentNode.insertBefore(S.els.jobMeter, st);
  }

  var elapsed = Math.max(0, Math.round((Date.now() - job.startedAt) / 1000));
  st.hidden = false;
  st.textContent = (job.adopted
    ? 'A reanalysis was already running when this page asked for one (started ' +
      job.startedLabel + ').'
    : 'Reanalyzing with SpeciesNet.') +
    ' Elapsed ' + durationClock(elapsed) + '. ' +
    (job.note || 'Waiting for the post-processor to write its log.');
  busy(S.els.reanalyzeBtn, true);
}

/* ---------------------------------------------------------------- metadata */

function buildMetadata() {
  var dl = h('dl.stack.stack--tight', { style: { margin: '0' } });
  S.els.meta = dl;
  return h('section.stack.stack--tight', { 'aria-label': 'Capture metadata' },
    h('span.overline.overline--strong', { text: 'Capture' }), dl);
}

function paintMetadata() {
  var clip = S.clip;
  var dl = S.els.meta;
  if (!clip || !dl) return;
  var t = parseServerTime(clip.time);
  var video = S.log && S.log.video ? S.log.video : null;
  var day = dayKey(clip.time);

  var rows = [
    ['Camera', cameraName(clip.camera) + ' (' + clip.camera + ')'],
    ['Started', (day ? longDate(day) + ' · ' : '') + clockTime(clip.time, { seconds: true }) +
      (t && t.offset ? ' ' + t.offset : '')],
    ['Age', timeAgo(clip.time)],
    ['Duration', duration() ? durationClock(duration()) : (video && video.frames
      ? durationClock(num(video.frames, 0) / (S.fps || 15)) : 'unknown')],
    ['Frame rate', (S.fps || 15) + ' fps'],
    ['Resolution', video && video.width ? video.width + ' × ' + video.height : 'unknown'],
    ['Frames', video && video.frames !== undefined ? String(video.frames) : 'unknown'],
    ['File size', fileSize(clip.size)],
    ['Tracks', String(S.tracks.length)],
    ['Path', clip.path]
  ];

  keyedList(dl, rows, {
    key: function (r) { return r[0]; },
    create: function (r) { return metaRow(r[0], r[1]); },
    update: function (el, r) {
      el.lastElementChild.textContent = r[1] === null || r[1] === undefined ? '—' : String(r[1]);
    }
  });
}

/* ----------------------------------------------------------- settings sheet */

var SETTING_FIELDS = [
  { key: 'sample_rate', label: 'Sample every N frames', hint: 'Lower is slower and more thorough.',
    type: 'number', min: 1, max: 30, step: 1 },
  { key: 'confidence_threshold', label: 'Detection confidence', hint: 'Detections below this are discarded.',
    type: 'number', min: 0, max: 1, step: 0.05 },
  { key: 'generic_confidence', label: 'Generic-label confidence', hint: 'Threshold for falling back to a coarse label.',
    type: 'number', min: 0, max: 1, step: 0.05 },
  { key: 'same_species_merge_gap', label: 'Same-species merge gap (frames)', hint: 'Fragments this far apart still merge.',
    type: 'number', min: 0, max: 900, step: 10 },
  { key: 'spatial_merge_iou', label: 'Spatial merge IoU', hint: 'Box overlap required to treat two tracks as one.',
    type: 'number', min: 0, max: 1, step: 0.05 },
  { key: 'tracking_enabled', label: 'Tracking', hint: 'Group detections into tracks with persistent ids.', type: 'switch' },
  { key: 'spatial_merge_enabled', label: 'Spatial merging', hint: 'Merge tracks that overlap in space.', type: 'switch' },
  { key: 'hierarchical_merge_enabled', label: 'Hierarchical merging', hint: 'Fold a generic label into a specific one.', type: 'switch' },
  { key: 'single_animal_mode', label: 'Single-animal mode', hint: 'Assume at most one animal per clip.', type: 'switch' },
  { key: 'thumbnail_cropped', label: 'Crop thumbnails', hint: 'Cut key frames to the detection box.', type: 'switch' }
];

function openSettingsSheet(anchor) {
  var base = (S.clip && S.clip.global_settings) || {};
  var draft = {};
  for (var i = 0; i < SETTING_FIELDS.length; i++) {
    var f = SETTING_FIELDS[i];
    if (base[f.key] !== undefined) draft[f.key] = base[f.key];
  }

  var handle = sheet({
    title: 'Analysis settings',
    snap: 'full',
    content: function (body) {
      body.appendChild(h('p.t-sm', {
        style: { margin: '0 0 var(--s-4)', color: 'var(--c-text-2)' },
        text: 'These are the pipeline defaults this clip was processed with. ' +
          'Changing them here does not touch cameras.yml — they are sent as ' +
          'overrides for one reanalysis of this clip.'
      }));
      var form = h('div.stack');
      for (var k = 0; k < SETTING_FIELDS.length; k++) {
        (function (field) {
          var value = draft[field.key];
          if (field.type === 'switch') {
            var state = h('span.switch-row__state', { text: value ? 'ON' : 'OFF' });
            var row = h('button.switch-row', {
              type: 'button', role: 'switch',
              'aria-checked': value ? 'true' : 'false'
            },
              h('span.switch-row__text',
                h('span.switch-row__title', { text: field.label }),
                h('span.switch-row__hint', { text: field.hint })),
              state,
              h('span.switch', h('span.switch__knob')));
            row.addEventListener('click', function () {
              var next = row.getAttribute('aria-checked') !== 'true';
              row.setAttribute('aria-checked', next ? 'true' : 'false');
              state.textContent = next ? 'ON' : 'OFF';
              draft[field.key] = next;
            });
            form.appendChild(row);
            return;
          }
          var id = 'at-set-' + field.key;
          var input = h('input.input.input--mono#' + id, {
            type: 'number',
            min: String(field.min), max: String(field.max), step: String(field.step),
            value: value === undefined ? '' : String(value),
            'aria-describedby': id + '-hint'
          });
          input.addEventListener('input', function () {
            var n = Number(input.value);
            if (input.value === '' || !isFinite(n)) { delete draft[field.key]; return; }
            draft[field.key] = n;
          });
          form.appendChild(h('div.field',
            h('label.field__label', { 'for': id, text: field.label }),
            input,
            h('span.field__hint#' + id + '-hint', { text: field.hint })));
        }(SETTING_FIELDS[k]));
      }
      body.appendChild(form);
    },
    onClose: function () { if (anchor && anchor.isConnected) anchor.focus(); }
  });

  S.overlays.push(handle);
  handle.result.then(function () {
    if (!S) return;
    var oi = S.overlays.indexOf(handle);
    if (oi >= 0) S.overlays.splice(oi, 1);
  });

  var foot = h('div.sheet__foot.sheet__foot--filter',
    btn('Cancel', { variant: 'secondary', onClick: function () { handle.close(null); } }),
    btn('Reanalyze with these', { variant: 'primary', icon: 'refresh', onClick: function () {
      handle.close('run');
      startReanalyze(draft);
    } }));
  foot.firstChild.classList.add('sheet__reset');
  foot.lastChild.classList.add('sheet__apply');
  handle.el.appendChild(foot);
}

/* ------------------------------------------------------------- reanalyze */

function parseServerError(err) {
  if (!err || !err.detail) return null;
  try { return JSON.parse(err.detail); } catch (e) { return null; }
}

function startReanalyze(overrides) {
  if (S.job) {
    toast.info('A reanalysis of this clip is already running.', {
      detail: 'Progress is shown under the species name.'
    });
    return;
  }
  var species = (S.clip && S.clip.species) || 'this clip';
  S.job = {
    startedAt: Date.now(),
    startedLabel: clockTime(new Date().toISOString()),
    adopted: false,
    note: 'Sent to the post-processor.',
    lines: []
  };
  jobLog('Reanalyze requested from this browser.');
  paintJobState();

  var progress = toast.progress('Reanalyzing ' + species, {
    detail: 'SpeciesNet runs on the server; you can leave this page.'
  });
  S.job.toast = progress;

  /* Fire and forget: the POST resolves only when the post-processor is done,
     and blocking the UI on it is the bug this rewrite exists to fix. */
  api.reprocess(S.path, overrides && Object.keys(overrides).length ? { settings: overrides } : null,
    { signal: signal() }).then(onReanalyzeDone, onReanalyzeFail);

  startJobPolling();
}

function onReanalyzeDone(payload) {
  if (!S || S.dead) return;
  var job = S.job;
  if (job && job.toast) job.toast.close();
  S.job = null;
  stopJobPolling();

  if (payload && payload.success === false) {
    toast.error('The reanalysis failed on the server.', {
      detail: payload.error || 'The post-processor reported no reason.'
    });
    jobLog('Failed: ' + (payload.error || 'no reason given'), 'error');
    refresh();
    return;
  }

  var newSpecies = payload && payload.new_species;
  var detail = joinMeta(
    payload && payload.frames_analyzed !== undefined
      ? payload.frames_analyzed + ' of ' + payload.total_frames + ' frames analysed' : null,
    payload && payload.tracks_detected !== undefined
      ? plural(payload.tracks_detected, 'track') : null,
    payload && payload.thumbnails_saved !== undefined
      ? plural(payload.thumbnails_saved, 'key frame') : null
  );
  jobLog('Complete. ' + detail);

  if (payload && payload.renamed && payload.new_path && payload.new_path !== S.path) {
    toast.success('Reanalyzed · now ' + (newSpecies || 'reclassified'), { detail: detail });
    /* The file moved on disk, so the route moved with it. */
    router.navigate(clipHref(payload.new_path, S.query), { replace: true });
    return;
  }

  toast.success('Reanalyzed' + (newSpecies ? ' · ' + newSpecies : ''), { detail: detail });
  refresh();
}

function onReanalyzeFail(err) {
  if (!S || S.dead) return;
  if (api.isAbort(err)) return;

  if (err && err.status === 409) {
    /* Somebody — another tab, another operator — already started this job.
       The server hands back its start time; adopt it and keep polling. */
    var body = parseServerError(err);
    var msg = (body && body.error) || err.detail || 'Processing already in progress.';
    if (S.job) {
      S.job.adopted = true;
      S.job.note = msg;
      if (S.job.toast) S.job.toast.update({ title: 'Reanalysis already running', detail: msg });
    }
    jobLog(msg, 'warn');
    paintJobState();
    startJobPolling();
    return;
  }

  if (S.job && S.job.toast) S.job.toast.close();
  S.job = null;
  stopJobPolling();
  jobLog('Request failed: ' + api.describe(err), 'error');
  paintJobState();
  toast.error('Could not reanalyze this recording.', {
    detail: api.describe(err),
    retry: function () { startReanalyze(null); }
  });
  paintProcessingLog();
}

function jobLog(message, level) {
  var entry = { at: new Date(), level: level || 'info', message: message };
  if (!S.jobLines) S.jobLines = [];
  S.jobLines.push(entry);
  paintProcessingLog();
}

function startJobPolling() {
  stopJobPolling();
  /* every() already records the id for teardown; do not double-register. */
  S.jobTicker = every(function () { paintJobState(); }, TICK_MS);
  S.jobPoller = every(function () {
    if (!store.get('visible')) return;      /* rule 7: hidden tabs poll nothing */
    pollJob();
  }, JOB_POLL_MS);
}

function stopJobPolling() {
  if (S.jobTicker) { window.clearInterval(S.jobTicker); S.jobTicker = null; }
  if (S.jobPoller) { window.clearInterval(S.jobPoller); S.jobPoller = null; }
}

/**
 * While a job runs we watch two things: the clip payload's `reprocessing`
 * flag (the server's own truth about whether a job is in flight) and the
 * sidecar log, which appears when the post-processor writes it.
 */
function pollJob() {
  api.clip(S.path, { signal: signal(), timeout: 10000 }).then(function (payload) {
    if (!S || S.dead) return;
    var wasReprocessing = S.clip && S.clip.reprocessing;
    applyClip(payload);
    if (S.job && payload && payload.reprocessing && S.job.note !== 'Running on the server.') {
      S.job.note = 'Running on the server.';
    }
    if (S.job && wasReprocessing && !payload.reprocessing) {
      S.job.note = 'Finishing up — writing thumbnails and the log.';
    }
    paintJobState();
  }, function (err) {
    if (!S || S.dead || api.isAbort(err)) return;
    if (S.job) S.job.note = 'Progress check failed: ' + api.describe(err);
    paintJobState();
  });
  loadProcessingLog(true);
}

/* --------------------------------------------------------- processing log */

function buildLogSection() {
  var toggle = h('button.btn.btn--secondary', {
    type: 'button',
    'aria-expanded': 'false',
    'aria-controls': 'at-clip-log'
  },
    icon('chevron-down', { size: 'sm', 'class': 'btn__icon' }),
    h('span.btn__spinner', h('span.spinner')),
    h('span.btn__label', { text: 'Processing log' }));
  S.els.logToggle = toggle;
  keep(on(toggle, 'click', function () {
    S.logOpen = !S.logOpen;
    toggle.setAttribute('aria-expanded', S.logOpen ? 'true' : 'false');
    S.els.logRegion.hidden = !S.logOpen;
    if (S.logOpen && S.log === null && !S.logMissing) loadProcessingLog(false);
  }));

  var copy = btn('Copy JSON', { variant: 'ghost', size: 'sm', icon: 'layers', onClick: function () {
    if (!S.log) { toast.info('There is no sidecar log to copy yet.'); return; }
    copyText(JSON.stringify(S.log, null, 2), 'Processing log copied as JSON.');
  } });
  S.els.logCopy = copy;

  var body = h('div.logview__body', { role: 'log', 'aria-label': 'Processing log entries' });
  S.els.logBody = body;

  var viewer = h('div.logview',
    h('div.logview__head',
      h('span.overline', { text: 'Post-processor' }),
      h('span.spacer'),
      copy),
    body);

  var region = h('div#at-clip-log', { hidden: true }, viewer);
  S.els.logRegion = region;

  return h('section.stack.stack--tight', { 'aria-label': 'Processing log' }, toggle, region);
}

function loadProcessingLog(quiet) {
  if (!quiet) busy(S.els.logToggle, true);
  api.processingLog(S.path, { signal: signal(), timeout: 10000 }).then(function (payload) {
    if (!S || S.dead) return;
    busy(S.els.logToggle, false);
    S.logError = null;
    if (payload && payload.exists) {
      S.logMissing = false;
      S.log = payload.data || null;
      S.tracks = buildTracks(S.clip, S.log);
      if (!duration()) S.duration = fallbackDuration(S.tracks, S.log, S.fps);
      paintLanes();
      paintKeystrip();
      paintTrackChips();
      paintMetadata();
    } else {
      S.logMissing = true;
      S.log = null;
      S.logNote = (payload && payload.message) ||
        'No processing log on disk. Reanalyze this clip to generate one.';
    }
    paintProcessingLog();
  }, function (err) {
    if (!S || S.dead || api.isAbort(err)) return;
    busy(S.els.logToggle, false);
    S.logError = err;
    paintProcessingLog();
    if (!quiet) {
      toast.error('Could not read the processing log.', {
        detail: api.describe(err),
        retry: function () { loadProcessingLog(false); }
      });
    }
  });
}

/** Turn the sidecar JSON plus this session's own events into log rows. */
function logRows() {
  var rows = [];
  var i;

  if (S.logError) {
    rows.push({ key: 'err', level: 'error', ts: '',
      msg: 'Could not read /recordings/log/' + S.path + ' — ' + api.describe(S.logError) });
  }

  var data = S.log;
  if (data) {
    if (data.video) {
      rows.push({ key: 'video', level: 'info', ts: 'video',
        msg: joinMeta(
          data.video.width && data.video.height ? data.video.width + '×' + data.video.height : null,
          data.video.fps !== undefined ? data.video.fps + ' fps' : null,
          data.video.frames !== undefined ? plural(data.video.frames, 'frame') : null) });
    }
    if (data.settings) {
      var parts = [];
      for (var k in data.settings) {
        if (!Object.prototype.hasOwnProperty.call(data.settings, k)) continue;
        parts.push(k + '=' + String(data.settings[k]));
      }
      rows.push({ key: 'settings', level: 'debug', ts: 'settings', msg: parts.join('  ') });
    }
    var tracks = data.tracking_summary && data.tracking_summary.tracks;
    if (Array.isArray(tracks)) {
      for (i = 0; i < tracks.length; i++) {
        var t = tracks[i] || {};
        rows.push({
          key: 'track-' + (t.track_id === undefined ? i : t.track_id),
          level: 'info',
          ts: 'track ' + (t.track_id === undefined ? i : t.track_id),
          msg: joinMeta(
            'frames ' + t.first_frame + '–' + t.last_frame,
            t.best_species || null,
            t.best_confidence !== undefined ? pct(t.best_confidence) : null)
        });
      }
    }
    /* Some deployments write a flat event array; render it if present. */
    var events = data.log || data.events;
    if (Array.isArray(events)) {
      for (i = 0; i < events.length; i++) {
        var e = events[i];
        if (e === null || e === undefined) continue;
        if (typeof e === 'string') {
          rows.push({ key: 'ev-' + i, level: 'info', ts: '', msg: e });
        } else {
          rows.push({
            key: 'ev-' + i,
            level: String(e.level || 'info').toLowerCase(),
            ts: e.time || e.timestamp || '',
            msg: String(e.message || e.msg || JSON.stringify(e))
          });
        }
      }
    }
  } else if (S.logMissing && !S.logError) {
    rows.push({ key: 'none', level: 'warn', ts: '', msg: S.logNote || 'No processing log on disk.' });
  }

  var lines = S.jobLines || [];
  for (i = 0; i < lines.length; i++) {
    rows.push({
      key: 'job-' + i,
      level: lines[i].level,
      ts: clockTime(lines[i].at.toISOString(), { seconds: true }),
      msg: lines[i].message
    });
  }
  return rows;
}

function paintProcessingLog() {
  var body = S.els.logBody;
  if (!body) return;
  var rows = logRows();

  if (!rows.length) {
    keyedList(body, [{ key: 'empty' }], {
      key: function (r) { return r.key; },
      create: function () {
        return h('div.logrow.logrow--debug',
          h('span.logrow__ts', { text: '' }),
          h('span.logrow__lvl', { text: 'none' }),
          h('span.logrow__msg', { text: 'Nothing logged for this clip yet.' }));
      }
    });
    return;
  }

  keyedList(body, rows, {
    key: function (r) { return r.key; },
    create: function () {
      var copy = h('button.logrow__copy', { type: 'button', 'aria-label': 'Copy this line' },
        icon('layers', { size: 'sm' }));
      return h('div.logrow',
        h('span.logrow__ts'),
        h('span.logrow__lvl'),
        h('span.logrow__msg'),
        copy);
    },
    update: function (el, r) {
      el.className = 'logrow logrow--' + (['debug', 'info', 'warn', 'error'].indexOf(r.level) >= 0
        ? r.level : 'info');
      el.children[0].textContent = r.ts || '';
      el.children[1].textContent = r.level;
      el.children[2].textContent = r.msg;
      el.children[3].dataset.line = (r.ts ? r.ts + ' ' : '') + r.msg;
    }
  });

  if (!S.els.logDelegated) {
    S.els.logDelegated = true;
    keep(delegate(body, 'click', '.logrow__copy', function (ev, el) {
      copyText(el.dataset.line || '', 'Log line copied.');
    }));
  }
}

/* ------------------------------------------------------------------ delete */

function confirmDelete() {
  var clip = S.clip;
  if (!clip) return;
  var species = clip.species || 'Unclassified';
  var label = species + ' · ' + clockTime(clip.time) + ' · ' + fileSize(clip.size);
  var path = S.path;
  var query = S.query;
  var cancelled = false;

  /* Optimistic + undo, per the reversibility rule. We leave for the grid at
     once; the toast (which lives in the shell, not in this view) holds the
     DELETE until its deadline. */
  toast('Recording deleted', {
    kind: 'danger',
    detail: label,
    undo: {
      label: 'Undo',
      onUndo: function () {
        cancelled = true;
        toast.info('Delete cancelled. The recording is still on disk.');
      },
      onExpire: function () {
        if (cancelled) return;
        api.deleteClip(path).then(function () {
          toast.success('Deleted ' + species + ' · ' + fileSize(clip.size) + ' freed.');
        }, function (err) {
          toast.error('The recording could not be deleted.', {
            detail: api.describe(err) + ' It is still on disk.',
            action: {
              label: 'Open the clip',
              onClick: function () { router.navigate(clipHref(path, query)); }
            }
          });
        });
      }
    }
  });

  router.go('/recordings', S.query);
}

/* ------------------------------------------------------------ global keys */

function installKeys() {
  keep(on(document, 'keydown', function (ev) {
    if (!S || S.dead) return;
    if (isOverlayOpen()) return;
    if (ev.metaKey || ev.ctrlKey || ev.altKey) return;
    var t = ev.target;
    var tag = t && t.tagName ? t.tagName.toLowerCase() : '';
    if (tag === 'input' || tag === 'textarea' || tag === 'select' ||
        (t && t.isContentEditable)) return;
    /* The scrubber owns its own arrow keys. */
    if (t === S.els.scrub) return;

    var key = ev.key;
    if (key === ' ' || key === 'Spacebar') {
      if (tag === 'button' || tag === 'a') return;   /* let the control fire */
      togglePlay();
      ev.preventDefault();
    } else if (key === 'ArrowRight') {
      seekTo(currentTime() + (ev.shiftKey ? frameStep() : SEEK_STEP));
      ev.preventDefault();
    } else if (key === 'ArrowLeft') {
      seekTo(currentTime() - (ev.shiftKey ? frameStep() : SEEK_STEP));
      ev.preventDefault();
    } else if (key === '.' || key === '>') {
      if (S.els.video && !S.els.video.paused) S.els.video.pause();
      seekTo(currentTime() + frameStep());
      ev.preventDefault();
    } else if (key === ',' || key === '<') {
      if (S.els.video && !S.els.video.paused) S.els.video.pause();
      seekTo(currentTime() - frameStep());
      ev.preventDefault();
    } else if (key === 'j' || key === 'J') {
      step(-1);
      ev.preventDefault();
    } else if (key === 'k' || key === 'K') {
      step(1);
      ev.preventDefault();
    }
  }));
}

/* ------------------------------------------------------ visibility / poll */

function installVisibility() {
  keep(on(document, 'visibilitychange', function () {
    if (!S || S.dead) return;
    var hidden = document.hidden;
    if (hidden) {
      /* Pause the media and stop every poll; nothing decodes in a hidden tab. */
      if (S.els.video && !S.els.video.paused) S.els.video.pause();
      stopJobPolling();
    } else if (S.job) {
      startJobPolling();
      pollJob();
    }
  }));

  keep(on(window, 'pagehide', function () {
    if (S && S.els.video) {
      S.els.video.pause();
    }
  }));
}

/* --------------------------------------------------------------- rendering */

function applyClip(payload) {
  S.clip = payload;
  S.fps = num(payload && payload.fps, S.fps) || 15;
  S.tracks = buildTracks(payload, S.log);
  if (!duration()) S.duration = fallbackDuration(S.tracks, S.log, S.fps);
  syncChrome();
  paintSpeciesPanel();
  paintKeystrip();
  paintTrackChips();
  paintLanes();
  paintMetadata();
  paintScrub();
  var heading = S.els.heading;
  if (heading) {
    heading.textContent = (payload.species || 'Clip') + ' — clip detail';
  }
  if (S.els.countLive) {
    S.els.countLive.textContent = plural(S.tracks.length, 'track') + ' in this recording.';
  }
  if (S.els.downloadLink) {
    S.els.downloadLink.setAttribute('href', api.clipUrl(S.path));
    S.els.downloadLink.setAttribute('download', payload.filename || 'clip.mp4');
  }
}

function buildLayout() {
  var left = h('div.stack', { style: { minWidth: '0' } });
  var right = h('div.stack', { style: { minWidth: '0' } });
  var grid = h('div', {
    style: { display: 'grid', gap: 'var(--s-6)', alignItems: 'start', minWidth: '0' }
  }, left, right);

  S.els.left = left;
  S.els.right = right;
  S.els.grid = grid;

  /* The two-column evidence layout is a media query in the mockups; here it is
     a matchMedia listener, because this view owns no stylesheet. */
  function applyMq(matches) {
    grid.style.gridTemplateColumns = matches
      ? 'minmax(0, 1fr) var(--aside-w)'
      : 'minmax(0, 1fr)';
  }
  if (window.matchMedia) {
    S.mq = window.matchMedia('(min-width: 1024px)');
    S.mqHandler = function (ev) { applyMq(ev.matches); };
    if (S.mq.addEventListener) {
      S.mq.addEventListener('change', S.mqHandler);
      keep(function () { S.mq.removeEventListener('change', S.mqHandler); });
    } else if (S.mq.addListener) {
      S.mq.addListener(S.mqHandler);
      keep(function () { S.mq.removeListener(S.mqHandler); });
    }
    applyMq(S.mq.matches);
  } else {
    applyMq(false);
  }
  return grid;
}

function buildActions() {
  var download = h('a.btn.btn--secondary', {
    href: api.clipUrl(S.path),
    download: (S.clip && S.clip.filename) || 'clip.mp4',
    rel: 'noopener'
  },
    icon('download', { size: 'sm', 'class': 'btn__icon' }),
    h('span.btn__label', { text: 'Download' }));
  S.els.downloadLink = download;

  var reanalyze = btn('Reanalyze', { variant: 'primary', icon: 'refresh', onClick: function () {
    startReanalyze(null);
  } });
  S.els.reanalyzeBtn = reanalyze;

  var settingsBtn = btn('Settings', { variant: 'secondary', icon: 'settings', onClick: function () {
    openSettingsSheet(settingsBtn);
  } });

  var del = btn('Delete', { variant: 'danger', icon: 'trash', onClick: function () {
    confirmDelete();
  } });

  return h('div.btn-row', reanalyze, download, settingsBtn, del);
}

function buildBody() {
  var grid = buildLayout();

  /* --- left: player, scrubber, key frames -------------------------------- */
  S.els.left.appendChild(buildPlayer());

  var stripHead = h('div.row.row--between',
    h('span.overline.overline--strong', { text: 'Key frames' }),
    h('div.row',
      h('span.mono.t-xs', { style: { color: 'var(--c-text-3)' } }),
      btn('Enlarge', { variant: 'ghost', size: 'sm', icon: 'layers', onClick: function () {
        openLightbox(S.activeTrack >= 0 ? S.activeTrack : 0);
      } })));
  S.els.position = stripHead.lastElementChild.firstElementChild;

  S.els.left.appendChild(h('section.stack.stack--tight', { 'aria-label': 'Key frames' },
    stripHead,
    buildKeystrip(),
    h('p.t-xs', {
      style: { margin: '0', color: 'var(--c-text-3)' },
      text: 'Click a frame to seek the video to that track. Shift+Enter enlarges it.'
    })));

  /* --- right: the evidence rail ------------------------------------------ */
  S.els.right.appendChild(buildSpeciesPanel());
  S.els.right.appendChild(h('section.stack.stack--tight', { 'aria-label': 'Tracks' },
    h('span.overline.overline--strong', { text: 'Tracks' }),
    buildTrackChips()));
  S.els.right.appendChild(buildMetadata());
  S.els.right.appendChild(buildActions());
  S.els.right.appendChild(buildLogSection());

  return grid;
}

function renderError(err) {
  var root = S.root;
  clear(root);
  root.appendChild(S.els.heading);
  root.appendChild(S.els.live);
  S.els.live.textContent = 'This recording could not be opened.';
  root.appendChild(h('div.empty.empty--error',
    h('div.empty__art', icon('image-off', { size: 'lg' })),
    h('h2.empty__title', { text: 'This recording could not be opened.' }),
    h('p.empty__body', { text: api.describe(err) }),
    h('p.empty__cause', { text: err && err.status === 404
      ? 'The file is not on disk any more — it may have been deleted or renamed by a reanalysis.'
      : 'The archive server answered, but not with this clip.' }),
    h('code.empty__endpoint', { text: 'GET /api/clip/' + S.path }),
    h('div.empty__actions',
      btn('Retry', { variant: 'primary', icon: 'refresh', onClick: function () { load(); } }),
      btn('Back to recordings', { variant: 'secondary', icon: 'arrow-left', onClick: function () {
        router.go('/recordings', S.query);
      } }))));
}

function renderSkeleton() {
  var root = S.root;
  var box = h('div.stack.stack--loose',
    h('div.frame.frame--skel', { style: { width: '100%' } }),
    h('div.row', h('span.skel.skel--pill'), h('span.skel.skel--pill')),
    h('div.stack.stack--tight',
      h('span.skel.skel--title'),
      h('span.skel.skel--text'),
      h('span.skel.skel--text')));
  S.els.skeleton = box;
  root.appendChild(box);
}

function load() {
  var root = S.root;
  clear(root);
  root.appendChild(S.els.heading);
  root.appendChild(S.els.live);
  renderSkeleton();

  api.clip(S.path, { signal: signal() }).then(function (payload) {
    if (!S || S.dead) return;
    clear(root);
    root.appendChild(S.els.heading);
    root.appendChild(S.els.live);
    /* buildBody() reads the poster, the film stock and the file name off the
       payload, so it has to be on S before the structure is built. */
    S.clip = payload;
    S.fps = num(payload && payload.fps, 15) || 15;
    S.tracks = buildTracks(payload, null);
    root.appendChild(buildBody());
    applyClip(payload);
    paintProcessingLog();
    loadProcessingLog(true);
    loadNeighbors();
    if (payload && payload.reprocessing && !S.job) {
      /* The server was already busy with this clip when we arrived. */
      S.job = {
        startedAt: Date.now(),
        startedLabel: clockTime(new Date().toISOString()),
        adopted: true,
        note: 'Started before this page was opened.'
      };
      paintJobState();
      startJobPolling();
    }
  }, function (err) {
    if (!S || S.dead || api.isAbort(err)) return;
    renderError(err);
    toast.error('Could not load this recording.', {
      detail: api.describe(err),
      retry: function () { load(); }
    });
  });
}

function refresh() {
  api.clip(S.path, { signal: signal() }).then(function (payload) {
    if (!S || S.dead) return;
    applyClip(payload);
  }, function (err) {
    if (!S || S.dead || api.isAbort(err)) return;
    toast.danger('The clip could not be refreshed.', { detail: api.describe(err) });
  });
  loadProcessingLog(true);
}

/* ------------------------------------------------------------------ view */

export const view = {
  mount: function (root, ctx) {
    S = freshState();
    S.root = root;
    S.path = (ctx && ctx.params && ctx.params.path) || '';
    S.query = filterQuery(ctx && ctx.query);

    S.els.heading = h('h1.visually-hidden', { tabIndex: -1, text: 'Clip detail' });
    S.els.live = h('p.visually-hidden', { role: 'status', 'aria-live': 'polite' });
    S.els.countLive = S.els.live;

    buildChromeActions();
    syncChrome();

    if (!S.path) {
      root.appendChild(S.els.heading);
      root.appendChild(h('div.empty.empty--error',
        h('h2.empty__title', { text: 'No clip was named in the URL.' }),
        h('p.empty__body', { text: 'A clip detail URL looks like /app/clip/cam1/2026/09/05/....mp4' }),
        h('div.empty__actions', btn('Back to recordings', {
          variant: 'primary', icon: 'arrow-left',
          onClick: function () { router.go('/recordings', S.query); }
        }))));
      return;
    }

    installKeys();
    installVisibility();
    load();
  },

  unmount: function () {
    if (!S) return;
    S.dead = true;

    stopJobPolling();
    stopAllTimers();

    for (var i = 0; i < S.disposers.length; i++) {
      try { S.disposers[i](); } catch (e) { /* a disposer must never block teardown */ }
    }
    S.disposers = [];

    /* Detach the media so nothing keeps decoding or holding the connection. */
    var v = S.els.video;
    if (v) {
      try {
        v.pause();
        v.removeAttribute('src');
        v.removeAttribute('poster');
        v.load();
      } catch (e2) {}
    }

    /* An overlay this view opened must not outlive it — its handlers read
       state that is about to be gone. */
    for (var o = S.overlays.length - 1; o >= 0; o--) {
      try { S.overlays[o].close(null); } catch (e4) {}
    }
    S.overlays = [];

    /* Cancel every in-flight fetch this view started. */
    if (S.abort) {
      try { S.abort.abort(); } catch (e3) {}
    }

    /* A running reanalysis toast belongs to the shell, not to this view: the
       job is on the server and the user asked for it. It is left alone on
       purpose, and so is a pending delete's undo deadline. */
    if (S.job && S.job.toast) S.job.toast.update({ detail: 'Still running on the server.' });

    S.els = {};
    S = null;
  }
};

export default view;
