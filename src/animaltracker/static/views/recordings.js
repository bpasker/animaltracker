/* ============================================================================
   views/recordings.js — the archive. Grid and Month are two modes of one view.

   WHAT LIVES IN THE URL (and nowhere else)
     view=grid|month   cameras=cam1,cam2   species=Deer,Raccoon
     from=YYYY-MM-DD   to=YYYY-MM-DD       date=YYYY-MM-DD
     year=2026         month=9             q=text   sort=newest
     density=compact
   The old UI's params are the only bookmarkable surface this rewrite inherits,
   so they round-trip exactly and every navigation goes through the router —
   Back and Forward walk months and filter sets, and nothing here reloads.

   WHAT SURVIVES A REFRESH
     scroll, sort, the selection, the open day panel and the collapsed days.
     Every list is reconciled by key (core/dom.js keyedList), never replaced.

   DELETE IS DEFERRED, NOT SOFT
     The server deletes for real. So bulk delete collapses the cards out of the
     model, shows a danger toast with a visible deadline, and only fires the
     request when that deadline expires. Undo puts every card back at its exact
     index because the model was never reordered — only masked.
   ========================================================================= */

import { h, clear, on, delegate, keyedList } from '../core/dom.js';
import { icon } from '../core/icons.js';
import { api } from '../core/api.js';
import { router } from '../core/router.js';
import { store } from '../core/store.js';
import { toast } from '../core/toast.js';
import { sheet, dialog, menu } from '../core/overlay.js';
import {
  clockTime, dayLabel, longDate, monthLabel, timeAgo, mb, plural,
  speciesClass, filmClass, joinMeta, dateFromKey, keyFromDate,
  confidenceSegments, isUnclassified, DAY_NAMES_SHORT
} from '../core/format.js';

/* --- constants ----------------------------------------------------------- */

var PAGE = 60;                 /* clips per append */
var MAX_PAGE = 500;            /* the server's hard cap */
var SKEL_FIRST = 12;           /* skeleton cards on first paint */
var SKEL_TAIL = 4;             /* skeleton cards while appending */
var UNDO_MS = 6000;
var LONG_PRESS_MS = 500;
var REFRESH_MS = 60000;
var DESKTOP = '(min-width: 1024px)';

var SORTS = [
  { value: 'newest', label: 'Newest first' },
  { value: 'oldest', label: 'Oldest first' },
  { value: 'species', label: 'Species A–Z' },
  { value: 'camera', label: 'Camera' },
  { value: 'largest', label: 'Largest first' }
];

var RANGES = [
  { value: '', label: 'All time' },
  { value: '1', label: 'Today' },
  { value: '7', label: '7 days' },
  { value: '30', label: '30 days' }
];

/* The single live instance. mount() builds it, unmount() tears every part of
   it down; nothing at module scope survives between mounts. */
var S = null;

/* --- small helpers ------------------------------------------------------- */

function isDesktop() {
  return window.matchMedia && window.matchMedia(DESKTOP).matches;
}

function splitList(value) {
  if (!value) return [];
  return String(value).split(',').map(function (s) { return s.trim(); })
    .filter(function (s) { return !!s; });
}

function sameList(a, b) {
  if (a.length !== b.length) return false;
  for (var i = 0; i < a.length; i++) if (a[i] !== b[i]) return false;
  return true;
}

function todayKey() {
  return keyFromDate(new Date());
}

function shiftDays(days) {
  var d = new Date();
  d.setHours(0, 0, 0, 0);
  d.setDate(d.getDate() - (days - 1));
  return keyFromDate(d);
}

function monthBounds(year, month) {
  var first = new Date(year, month - 1, 1);
  var last = new Date(year, month, 0);
  return { from: keyFromDate(first), to: keyFromDate(last) };
}

function clipId(path) {
  return 'clip-' + String(path).replace(/[^A-Za-z0-9]+/g, '-');
}

function clipHref(path) {
  return router.href('/clips/' + api.encodePath(path));
}

/** "White-tailed Deer, 6:42 PM · cam1" — the accessible name of a card. */
function clipLabel(clip) {
  return joinMeta(clip.species || 'Unclassified', clockTime(clip.time), clip.camera);
}

function thumbUrl(clip) {
  if (clip.thumbnail) return clip.thumbnail;
  var t = clip.thumbnails && clip.thumbnails[0];
  if (t && t.url) return t.url;
  if (t && t.path) return api.thumbUrl(t.path);
  return null;
}

/** The day-panel payload has a different shape; normalise it into a clip. */
function normaliseDayClip(raw, date) {
  return {
    path: raw.path,
    filename: raw.filename,
    camera: raw.camera,
    species: raw.species,
    raw_species: raw.raw_species,
    date: date,
    time: date + 'T' + (raw.time || '00:00:00'),
    size_mb: raw.size_mb,
    confidence: raw.confidence,
    thumbnails: raw.thumbnails || [],
    thumbnail: (raw.thumbnails && raw.thumbnails[0] && raw.thumbnails[0].url) || null
  };
}

function filtersActive(f) {
  return !!(f.cameras.length || f.species.length || f.from || f.to || f.q);
}

function describeFilters(f) {
  var parts = [];
  if (f.q) parts.push('matching “' + f.q + '”');
  if (f.species.length) parts.push(f.species.join(', '));
  if (f.cameras.length) parts.push('on ' + f.cameras.join(' and '));
  if (f.from && f.to && f.from === f.to) parts.push('on ' + longDate(f.from));
  else if (f.from && f.to) parts.push('between ' + longDate(f.from) + ' and ' + longDate(f.to));
  else if (f.from) parts.push('since ' + longDate(f.from));
  else if (f.to) parts.push('up to ' + longDate(f.to));
  return parts.join(' ');
}

/* --- URL <-> state ------------------------------------------------------- */

function readState(ctx) {
  var q = (ctx && ctx.query) || {};
  var view = q.view === 'month' || q.view === 'calendar' ? 'month' : 'grid';
  var density = (q.density === 'compact' || q.view === 'list') ? 'compact' : 'comfortable';

  var from = q.from || '';
  var to = q.to || '';
  /* The old UI's ?date= means "this one day" in the grid and "the open day"
     in the month view. Both are honoured. */
  if (view === 'grid' && q.date) { from = q.date; to = q.date; }

  var sort = 'newest';
  for (var i = 0; i < SORTS.length; i++) if (SORTS[i].value === q.sort) sort = q.sort;

  var now = new Date();
  var year = parseInt(q.year, 10);
  var month = parseInt(q.month, 10);
  if (!(year >= 1970 && year <= 3000)) year = now.getFullYear();
  if (!(month >= 1 && month <= 12)) month = now.getMonth() + 1;

  return {
    view: view,
    density: density,
    year: year,
    month: month,
    date: q.date || '',
    filters: {
      cameras: splitList(q.cameras || q.camera),
      species: splitList(q.species),
      from: from,
      to: to,
      q: q.q || '',
      sort: sort
    }
  };
}

function sameFilters(a, b) {
  return sameList(a.cameras, b.cameras) && sameList(a.species, b.species) &&
    a.from === b.from && a.to === b.to && a.q === b.q && a.sort === b.sort;
}

function apiQuery(f, extra) {
  var query = { sort: f.sort };
  if (f.cameras.length) query.camera = f.cameras.join(',');
  if (f.species.length) query.species = f.species.join(',');
  if (f.from) query.from = f.from;
  if (f.to) query.to = f.to;
  if (f.q) query.q = f.q;
  if (extra) for (var k in extra) if (Object.prototype.hasOwnProperty.call(extra, k)) query[k] = extra[k];
  return query;
}

/** Push a filter change into the URL — the router brings it back via update(). */
function applyFilters(next, opts) {
  var patch = {
    cameras: next.cameras.length ? next.cameras.join(',') : null,
    species: next.species.length ? next.species.join(',') : null,
    from: next.from || null,
    to: next.to || null,
    q: next.q || null,
    sort: next.sort === 'newest' ? null : next.sort
  };
  /* In the grid the day lives in from/to, so ?date= would fight it. In the
     month view ?date= is the open day panel and survives a filter change. */
  if (!S || S.view !== 'month') patch.date = null;
  router.setQuery(patch, opts || {});
}

/* --- timers, listeners, requests: one place to release them -------------- */

function track(off) { if (off) S.cleanups.push(off); return off; }

function later(fn, ms) {
  var id = window.setTimeout(function () {
    var i = S.timers.indexOf(id);
    if (i >= 0) S.timers.splice(i, 1);
    fn();
  }, ms);
  S.timers.push(id);
  return id;
}

function abortRequest(name) {
  var c = S.aborts[name];
  if (c) { try { c.abort(); } catch (e) {} }
  var next = typeof AbortController === 'function' ? new AbortController() : null;
  S.aborts[name] = next;
  return next ? next.signal : undefined;
}

function reportError(where, err, retry) {
  if (api.isAbort(err)) return;
  toast.error(where, { detail: api.describe(err), retry: retry });
}

/* ============================================================================
   CARD
   ========================================================================= */

function buildSkeletonCard() {
  var li = h('li.cliptile', { 'aria-hidden': 'true' },
    h('article.clip',
      h('div.clip__media.frame.frame--skel'),
      h('div.clip__meta',
        h('span.clip__skelbar'),
        h('span.clip__skelbar.clip__skelbar--short'))));
  return li;
}

function buildCard(clip) {
  var li = h('li.cliptile', { 'class': speciesClass(clip.species) });
  var art = h('article.clip');
  var id = clipId(clip.path);

  var link = h('a.clip__open', {
    href: clipHref(clip.path),
    'data-path': clip.path,
    tabIndex: -1,
    text: clipLabel(clip)
  });

  var check = h('input.clip__check', {
    type: 'checkbox',
    id: 'sel-' + id,
    'data-path': clip.path,
    tabIndex: -1
  });
  var checkLabel = h('label.visually-hidden', { 'for': 'sel-' + id,
    text: 'Select ' + clipLabel(clip) });

  var media = h('div.clip__media.frame');
  media.appendChild(h('div.frame__film', { 'class': filmClass(clip.time) }));

  var src = thumbUrl(clip);
  var img = null;
  if (src) {
    img = h('img.frame__img', {
      src: src, alt: '', loading: 'lazy', decoding: 'async'
    });
    on(img, 'error', function () {
      media.classList.add('frame--error');
      if (!media.querySelector('.frame__errpill')) {
        media.appendChild(h('span.frame__errpill', icon('image-off', { size: 'sm' }),
          h('span', 'No frame')));
      }
    });
    media.appendChild(img);
  } else {
    media.classList.add('frame--error');
    media.appendChild(h('span.frame__errpill', icon('image-off', { size: 'sm' }),
      h('span', 'No frame')));
  }
  media.appendChild(h('div.frame__scrim'));

  var play = h('button.icon-btn.icon-btn--on-media', {
    type: 'button',
    'data-play': clip.path,
    tabIndex: -1,
    'aria-label': 'Play ' + clipLabel(clip) + ' without leaving the archive'
  }, icon('play', { size: 'sm' }));
  media.appendChild(h('span.frame__tr', { style: { zIndex: '6' } }, play));

  media.appendChild(h('span.frame__bl', h('span.mpill', { text: clockTime(clip.time) })));
  if (clip.size_mb !== undefined && clip.size_mb !== null) {
    media.appendChild(h('span.frame__br', h('span.mpill', { text: mb(clip.size_mb) })));
  }

  var title = h('h3.clip__title', h('span.sp-dot'),
    h('span', { text: clip.species || 'Unclassified' }));
  if (isUnclassified(clip.species)) art.classList.add('clip--unclassified');

  var sub = h('div.clip__sub',
    h('span.clip__when', { text: timeAgo(clip.time) }),
    h('span.clip__sep', '·'),
    h('span.clip__cam', { text: clip.camera || 'unknown camera' }));
  if (clip.confidence !== undefined && clip.confidence !== null) {
    sub.appendChild(h('span.meter.clip__conf', {
      'role': 'img',
      'aria-label': 'Confidence ' + Math.round(Number(clip.confidence) * 100) + '%',
      style: { '--n': String(confidenceSegments(clip.confidence)), '--of': '8' }
    }));
  }

  art.appendChild(link);
  art.appendChild(check);
  art.appendChild(checkLabel);
  art.appendChild(media);
  art.appendChild(h('div.clip__meta', title, sub));
  li.appendChild(art);

  li._parts = { link: link, check: check, art: art, play: play,
    media: media, title: title.lastChild, dot: title.firstChild,
    when: sub.querySelector('.clip__when'), cam: sub.querySelector('.clip__cam') };
  return li;
}

function updateCard(li, clip) {
  var p = li._parts;
  if (!p) return;

  /* Post-processing renames a clip in place; the card follows without being
     rebuilt, so its decoded image and its focus survive. */
  var label = clipLabel(clip);
  if (p.title.textContent !== (clip.species || 'Unclassified')) {
    p.title.textContent = clip.species || 'Unclassified';
    li.className = 'cliptile ' + speciesClass(clip.species);
    p.art.classList.toggle('clip--unclassified', isUnclassified(clip.species));
  }
  if (p.link.textContent !== label) p.link.textContent = label;
  if (p.when) p.when.textContent = timeAgo(clip.time);
  if (p.cam) p.cam.textContent = clip.camera || 'unknown camera';
  var selected = S.selected.has(clip.path);
  p.art.classList.toggle('clip--selected', selected);
  p.check.checked = selected;
  p.check.tabIndex = S.selectMode ? 0 : -1;
  p.link.tabIndex = (S.focusKey === clip.path) ? 0 : -1;
  p.play.tabIndex = (!S.selectMode && S.focusKey === clip.path) ? 0 : -1;
  p.art.classList.toggle('clip--deleting', S.pendingDelete.has(clip.path));
}

/* ============================================================================
   EMPTY STATES — diagnose, do not shrug
   ========================================================================= */

function emptyState(spec) {
  var box = h('div.empty', { 'class': spec.error ? 'empty--error' : null });
  box.appendChild(h('div.empty__art', icon(spec.icon || 'film', { size: 'lg' })));
  box.appendChild(h('h2.empty__title', { text: spec.title }));
  if (spec.body) box.appendChild(h('p.empty__body', { text: spec.body }));
  if (spec.cause) box.appendChild(h('p.empty__cause', { text: spec.cause }));
  if (spec.endpoint) box.appendChild(h('p.empty__endpoint', { text: spec.endpoint }));
  if (spec.actions && spec.actions.length) {
    var row = h('div.empty__actions');
    spec.actions.forEach(function (a) {
      var btn = h('button.btn', {
        type: 'button',
        'class': 'btn--' + (a.variant || 'secondary')
      }, h('span.btn__label', { text: a.label }));
      on(btn, 'click', a.onClick);
      row.appendChild(btn);
    });
    box.appendChild(row);
  }
  return box;
}

/** Name a camera the user filtered to that is not currently producing. */
function staleCause(f) {
  var cams = store.get('cameras') || [];
  for (var i = 0; i < cams.length; i++) {
    var c = cams[i];
    if (f.cameras.length && f.cameras.indexOf(c.id) < 0) continue;
    if (c.state === 'offline') {
      return (c.name || c.id) + ' is offline, which may be why nothing new has landed.';
    }
    if (c.state === 'stale') {
      return (c.name || c.id) + ' has been stale, which may be why nothing new has landed.';
    }
  }
  return null;
}

function gridEmpty() {
  var f = S.filters;

  if (S.error) {
    return emptyState({
      error: true, icon: 'alert',
      title: 'The archive did not answer',
      body: api.describe(S.error),
      endpoint: '/api/recordings',
      actions: [{ label: 'Retry', variant: 'primary', onClick: function () { loadGrid(true); } }]
    });
  }

  if (!S.archiveTotal) {
    return emptyState({
      icon: 'film',
      title: 'No clips on disk yet',
      body: 'The detector writes a clip only when it sees something. Nothing has been ' +
            'recorded since this archive was created.',
      cause: staleCause({ cameras: [] }) || undefined,
      actions: [
        { label: 'Open Live', variant: 'primary', onClick: function () { router.go('/live', {}); } },
        { label: 'Open Monitor', onClick: function () { router.go('/monitor', {}); } }
      ]
    });
  }

  var actions = [{
    label: 'Clear filters', variant: 'primary',
    onClick: function () { applyFilters({ cameras: [], species: [], from: '', to: '', q: '', sort: f.sort }); }
  }];
  if (f.from || f.to) {
    actions.push({
      label: 'Widen to 30 days',
      onClick: function () {
        applyFilters({ cameras: f.cameras, species: f.species, from: shiftDays(30), to: '', q: f.q, sort: f.sort });
      }
    });
  }
  if (f.species.length) {
    actions.push({
      label: 'Any species',
      onClick: function () {
        applyFilters({ cameras: f.cameras, species: [], from: f.from, to: f.to, q: f.q, sort: f.sort });
      }
    });
  }

  return emptyState({
    icon: 'filter',
    title: 'No clip matches these filters',
    body: 'You are filtered to ' + (describeFilters(f) || 'nothing in particular') +
          '. The archive holds ' + plural(S.archiveTotal, 'clip') + ' in total.',
    cause: staleCause(f) || undefined,
    actions: actions
  });
}

/* ============================================================================
   GRID
   ========================================================================= */

function visibleClips() {
  var out = [];
  for (var i = 0; i < S.clips.length; i++) {
    var path = S.clips[i].path;
    /* A clip being deleted stays in the list for one animation frame set, so
       the collapse is seen; then the mask takes it out. */
    if (S.pendingDelete.has(path) && !S.collapsing.has(path)) continue;
    out.push(S.clips[i]);
  }
  return out;
}

function groupByDay(clips) {
  var groups = [];
  var index = {};
  for (var i = 0; i < clips.length; i++) {
    var c = clips[i];
    var key = c.date || 'undated';
    if (index[key] === undefined) {
      index[key] = groups.length;
      groups.push({ key: key, date: c.date, clips: [] });
    }
    groups[index[key]].clips.push(c);
  }
  return groups;
}

function renderGrid() {
  var host = S.els.groups;
  var clips = visibleClips();
  var groups = groupByDay(clips);

  if (S.loading && !clips.length) {
    groups = [{ key: '__skel__', skeleton: SKEL_FIRST, clips: [] }];
  } else if (S.loadingMore) {
    groups = groups.concat([{ key: '__skeltail__', skeleton: SKEL_TAIL, clips: [] }]);
  }

  S.els.emptyHost.hidden = !!clips.length || S.loading;
  if (!clips.length && !S.loading) {
    clear(S.els.emptyHost).appendChild(gridEmpty());
  } else if (!S.els.emptyHost.hidden) {
    clear(S.els.emptyHost);
  }

  keyedList(host, groups, {
    key: function (g) { return g.key; },
    create: function (g) {
      var section = h('section.daygroup');
      var head = h('div.dayhead');
      var toggle = h('button.dayhead__toggle', {
        type: 'button', 'data-day': g.key, 'aria-expanded': 'true'
      }, icon('chevron-down', { size: 'sm' }));
      var title = h('h2.dayhead__title');
      var when = h('span.dayhead__when');
      var count = h('span.dayhead__count');
      head.appendChild(toggle);
      head.appendChild(title);
      head.appendChild(when);
      head.appendChild(count);
      var list = h('ul.clipgrid', { role: 'list' });
      section.appendChild(head);
      section.appendChild(list);
      section._parts = { toggle: toggle, title: title, when: when, count: count, list: list, head: head };
      return section;
    },
    update: function (section, g) {
      var p = section._parts;
      if (g.skeleton) {
        p.head.hidden = true;
        var stubs = [];
        for (var i = 0; i < g.skeleton; i++) stubs.push({ key: g.key + '-' + i });
        keyedList(p.list, stubs, {
          key: function (s) { return s.key; },
          create: buildSkeletonCard
        });
        p.list.className = 'clipgrid' + (S.density === 'compact' ? ' clipgrid--compact' : '');
        p.list.setAttribute('aria-hidden', 'true');
        return;
      }
      p.head.hidden = false;
      p.list.removeAttribute('aria-hidden');
      var label = dayLabel(g.date);
      p.title.textContent = label.relative || label.full;
      p.when.textContent = label.relative ? label.full : '';
      p.count.textContent = plural(g.clips.length, 'clip');
      var collapsed = S.collapsed.has(g.key);
      section.classList.toggle('daygroup--collapsed', collapsed);
      p.toggle.setAttribute('aria-expanded', collapsed ? 'false' : 'true');
      p.toggle.setAttribute('aria-label', (collapsed ? 'Expand ' : 'Collapse ') + label.full);
      p.list.className = 'clipgrid' +
        (S.density === 'compact' ? ' clipgrid--compact' : '') +
        (S.selectMode ? ' clipgrid--select' : '');
      keyedList(p.list, g.clips, {
        key: function (c) { return c.path; },
        create: buildCard,
        update: updateCard
      });
    }
  });

  /* Roving tabindex: exactly one card is in the tab order. */
  if (clips.length) {
    var still = false;
    for (var j = 0; j < clips.length; j++) if (clips[j].path === S.focusKey) still = true;
    if (!still) {
      S.focusKey = clips[0].path;
      var first = host.querySelector('.clip__open[data-path]');
      if (first) first.tabIndex = 0;
    }
  }

  S.els.more.hidden = !S.hasMore || !!S.loading;
  S.els.more.disabled = !!S.loadingMore;
  clear(S.els.more).appendChild(h('span.btn__label', {
    text: S.loadingMore ? 'Loading…'
      : 'Load ' + Math.min(PAGE, Math.max(0, S.total - clips.length)) + ' more'
  }));
  S.els.sentinel.hidden = !S.hasMore;

  renderCount();
  renderSelbar();
}

function renderCount() {
  var shown = visibleClips().length;
  var text = S.loading && !shown ? 'Loading the archive…'
    : plural(S.total, 'clip') + (S.total > shown ? ' · ' + shown + ' loaded' : '');
  S.els.count.textContent = text;

  var speciesCount = (S.universe.species || []).length;
  store.setChrome({
    title: 'Recordings',
    subtitle: S.loading && !shown ? 'Loading…' : joinMeta(
      plural(S.total, 'clip'),
      speciesCount ? plural(speciesCount, 'species', 'species') : '',
      S.total < S.archiveTotal ? 'of ' + S.archiveTotal : ''),
    actions: S.chromeActions,
    toolbar: null,
    rail: S.els.rail,
    norail: false,
    selbar: S.selectMode ? S.els.selbar : null,
    mods: []
  });
}

/* --- fetching ------------------------------------------------------------ */

function loadGrid(force) {
  var signal = abortRequest('grid');
  S.loading = true;
  S.loadingMore = false;
  S.error = null;
  if (force) { S.clips = []; S.offset = 0; }
  renderGrid();

  var q = apiQuery(S.filters, { limit: PAGE, offset: 0 });
  api.recordings(q, { signal: signal }).then(function (data) {
    S.loading = false;
    S.clips = data.clips || [];
    S.total = data.total || 0;
    S.archiveTotal = data.archive_total || 0;
    S.hasMore = !!data.has_more;
    S.offset = S.clips.length;
    S.facets = data.facets || { cameras: [], species: [] };
    renderGrid();
    renderFilterUI();
    loadUniverse();
  }).catch(function (err) {
    if (api.isAbort(err)) return;
    S.loading = false;
    S.error = err;
    renderGrid();
    reportError('Could not load the archive', err, function () { loadGrid(true); });
  });
}

function loadMore() {
  if (S.loadingMore || S.loading || !S.hasMore) return;
  S.loadingMore = true;
  renderGrid();
  var signal = abortRequest('more');
  var q = apiQuery(S.filters, { limit: PAGE, offset: S.offset });
  api.recordings(q, { signal: signal }).then(function (data) {
    S.loadingMore = false;
    var seen = {};
    for (var i = 0; i < S.clips.length; i++) seen[S.clips[i].path] = true;
    (data.clips || []).forEach(function (c) {
      if (!seen[c.path]) S.clips.push(c);
    });
    S.offset = S.clips.length;
    S.hasMore = !!data.has_more;
    S.total = data.total || S.total;
    renderGrid();
  }).catch(function (err) {
    if (api.isAbort(err)) return;
    S.loadingMore = false;
    renderGrid();
    reportError('Could not load more clips', err, loadMore);
  });
}

/**
 * The facet universe: counts for every camera and species AVAILABLE under the
 * non-category filters. The main response's facets narrow to what is selected,
 * which would make a chip disappear the moment you picked it.
 */
function loadUniverse() {
  var signal = abortRequest('universe');
  var base = { from: S.filters.from, to: S.filters.to, q: S.filters.q,
    cameras: [], species: [], sort: S.filters.sort };
  api.recordings(apiQuery(base, { limit: 1, offset: 0 }), { signal: signal })
    .then(function (data) {
      S.universe = data.facets || { cameras: [], species: [] };
      renderFilterUI();
      renderCount();
    }).catch(function (err) {
      if (api.isAbort(err)) return;
      /* Not fatal: the chips fall back to the narrowed facets. */
      S.universe = S.facets;
      renderFilterUI();
    });
}

/** A silent refresh: new clips are merged in; scroll, selection and focus stay. */
function refreshGrid() {
  if (S.loading || S.loadingMore || !store.get('visible')) return;
  var signal = abortRequest('refresh');
  api.recordings(apiQuery(S.filters, { limit: PAGE, offset: 0 }), { signal: signal })
    .then(function (data) {
      var incoming = data.clips || [];
      var known = {};
      for (var i = 0; i < S.clips.length; i++) known[S.clips[i].path] = S.clips[i];
      var fresh = [];
      for (var j = 0; j < incoming.length; j++) {
        if (!known[incoming[j].path]) fresh.push(incoming[j]);
        else Object.assign(known[incoming[j].path], incoming[j]);
      }
      if (fresh.length) {
        if (S.filters.sort === 'newest') S.clips = fresh.concat(S.clips);
        else S.clips = S.clips.concat(fresh);
        S.offset = S.clips.length;
      }
      S.total = data.total || S.total;
      S.archiveTotal = data.archive_total || S.archiveTotal;
      S.facets = data.facets || S.facets;
      renderGrid();
      if (fresh.length) {
        S.els.live.textContent = plural(fresh.length, 'new clip') + ' added';
      }
    }).catch(function (err) {
      if (api.isAbort(err)) return;
      /* A background refresh still has to be visible when it fails. */
      toast('Refresh failed', { kind: 'error', detail: api.describe(err), timeout: 6000 });
    });
}

/* ============================================================================
   SELECTION
   ========================================================================= */

function setSelectMode(on_) {
  if (S.selectMode === on_) return;
  S.selectMode = on_;
  if (!on_) { S.selected.clear(); S.anchor = null; }
  S.els.live.textContent = on_ ? 'Selection mode on' : 'Selection mode off';
  renderGrid();
  renderChromeActions();
}

function toggleSelection(path, extend) {
  if (!S.selectMode) setSelectMode(true);
  if (extend && S.anchor) {
    var clips = visibleClips();
    var a = -1, b = -1;
    for (var i = 0; i < clips.length; i++) {
      if (clips[i].path === S.anchor) a = i;
      if (clips[i].path === path) b = i;
    }
    if (a >= 0 && b >= 0) {
      var lo = Math.min(a, b), hi = Math.max(a, b);
      for (var j = lo; j <= hi; j++) S.selected.add(clips[j].path);
      renderGrid();
      return;
    }
  }
  if (S.selected.has(path)) S.selected.delete(path);
  else { S.selected.add(path); S.anchor = path; }
  renderGrid();
}

function selectAllMatching() {
  var signal = abortRequest('selectall');
  var btn = S.els.selAll;
  btn.disabled = true;
  api.recordings(apiQuery(S.filters, { limit: MAX_PAGE, offset: 0 }), { signal: signal })
    .then(function (data) {
      btn.disabled = false;
      (data.clips || []).forEach(function (c) { S.selected.add(c.path); });
      renderGrid();
      if (data.total > MAX_PAGE) {
        toast('Selected the first ' + MAX_PAGE + ' of ' + data.total, {
          kind: 'info',
          detail: 'The server returns at most ' + MAX_PAGE + ' clips per request. Narrow the filters to reach the rest.'
        });
      }
    }).catch(function (err) {
      btn.disabled = false;
      if (api.isAbort(err)) return;
      reportError('Could not select every matching clip', err, selectAllMatching);
    });
}

function renderSelbar() {
  if (!S.els.selbar) return;
  var n = S.selected.size;
  S.els.selCount.textContent = plural(n, 'clip') + ' selected';
  S.els.selCountSub.textContent = 'of ' + plural(S.total, 'clip') + ' matching';
  S.els.selAll.hidden = n >= S.total;
  clear(S.els.selAll).appendChild(h('span.btn__label', { text: 'Select all ' + S.total }));
  var disabled = n === 0;
  S.els.selDelete.disabled = disabled;
  S.els.selReanalyze.disabled = disabled;
  S.els.selClear.disabled = disabled;
}

function buildSelbar() {
  var count = h('strong', { text: '0 clips selected' });
  var sub = h('span');
  var box = h('div.selbar__count', { 'aria-live': 'polite' }, count, sub);

  var selAll = h('button.btn.btn--secondary.btn--sm', { type: 'button' },
    h('span.btn__label', 'Select all'));
  var clearBtn = h('button.btn.btn--ghost', { type: 'button' },
    h('span.btn__label', 'Deselect'));
  var reanalyze = h('button.btn.btn--secondary', { type: 'button' },
    icon('sparkle', { size: 'sm', 'class': 'btn__icon' }),
    h('span.btn__label', 'Reanalyze'));
  var del = h('button.btn.btn--danger', { type: 'button' },
    icon('trash', { size: 'sm', 'class': 'btn__icon' }),
    h('span.btn__label', 'Delete'));
  var done = h('button.btn.btn--ghost', { type: 'button' },
    h('span.btn__label', 'Done'));

  track(on(selAll, 'click', selectAllMatching));
  track(on(clearBtn, 'click', function () { S.selected.clear(); S.anchor = null; renderGrid(); }));
  track(on(reanalyze, 'click', reanalyzeSelection));
  track(on(del, 'click', deleteSelection));
  track(on(done, 'click', function () { setSelectMode(false); }));

  var bar = h('div.selbar', { role: 'region', 'aria-label': 'Selection actions' },
    box,
    h('span.selbar__sep'),
    h('div.selbar__actions', selAll, clearBtn, reanalyze, del, done));

  S.els.selCount = count;
  S.els.selCountSub = sub;
  S.els.selAll = selAll;
  S.els.selClear = clearBtn;
  S.els.selDelete = del;
  S.els.selReanalyze = reanalyze;
  return bar;
}

/* --- optimistic bulk delete --------------------------------------------- */

function deleteSelection() {
  var paths = [];
  S.selected.forEach(function (p) { paths.push(p); });
  if (!paths.length) return;

  /* Collapse first (the class animates), then mask from the model. */
  paths.forEach(function (p) {
    S.pendingDelete.add(p);
    S.collapsing.add(p);
  });
  S.selected.clear();
  S.anchor = null;
  var restoreTotal = S.total;
  S.total = Math.max(0, S.total - paths.length);
  setSelectMode(false);
  renderGrid();
  later(function () {
    paths.forEach(function (p) { S.collapsing.delete(p); });
    renderGrid();
  }, 200);

  var deadline = Date.now() + UNDO_MS;
  var undone = false;

  function restore() {
    undone = true;
    paths.forEach(function (p) { S.pendingDelete.delete(p); S.collapsing.delete(p); });
    S.total = restoreTotal;
    renderGrid();
    S.els.live.textContent = plural(paths.length, 'clip') + ' restored';
  }

  var t = toast(plural(paths.length, 'clip') + ' deleted', {
    kind: 'danger',
    detail: 'Undo within ' + Math.round(UNDO_MS / 1000) + ' seconds — the files are removed when this closes.',
    timeout: UNDO_MS,
    undo: {
      label: 'Undo',
      onUndo: restore,
      onExpire: function () {
        if (undone) return;
        commitDelete(paths, restoreTotal);
      }
    }
  });

  /* A visible deadline, counted down in the toast's own detail line. */
  var tick = window.setInterval(function () {
    var left = Math.max(0, Math.ceil((deadline - Date.now()) / 1000));
    if (left <= 0 || undone) { window.clearInterval(tick); return; }
    t.update({ detail: 'Undo within ' + left + 's — the files are removed when this closes.' });
  }, 500);
  S.intervals.push(tick);
}

function commitDelete(paths, restoreTotal) {
  api.bulkDelete(paths, {}).then(function (res) {
    var n = (res && res.deleted_count) || paths.length;
    toast.success(plural(n, 'clip') + ' removed from disk');
    if (!S) return;
    /* Success: drop them from the model for good. */
    S.clips = S.clips.filter(function (c) { return paths.indexOf(c.path) < 0; });
    paths.forEach(function (p) { S.pendingDelete.delete(p); S.collapsing.delete(p); });
    S.offset = S.clips.length;
    renderGrid();
    if (S.view === 'month') loadMonth(true);
  }).catch(function (err) {
    /* Failure returns every card to its exact position. */
    if (S) {
      paths.forEach(function (p) { S.pendingDelete.delete(p); S.collapsing.delete(p); });
      S.total = restoreTotal;
      renderGrid();
    }
    reportError('The server refused to delete those clips', err, function () {
      commitDelete(paths, restoreTotal);
    });
  });
}

function reanalyzeSelection() {
  var paths = [];
  S.selected.forEach(function (p) { paths.push(p); });
  if (!paths.length) return;
  var done = 0, failed = 0;
  var t = toast.progress('Reanalyzing ' + plural(paths.length, 'clip'), {
    timeout: 0, detail: '0 of ' + paths.length + ' complete'
  });
  setSelectMode(false);

  function step(i) {
    if (i >= paths.length) {
      t.close();
      if (failed) {
        toast.error(plural(failed, 'clip') + ' could not be reanalyzed', {
          detail: 'The rest were queued. Monitor shows the post-processor queue.'
        });
      } else {
        toast.success(plural(done, 'clip') + ' queued for SpeciesNet');
      }
      later(function () { refreshGrid(); }, 1500);
      return;
    }
    api.reprocess(paths[i], null, {}).then(function () { done++; }, function () { failed++; })
      .then(function () {
        t.update({ detail: (done + failed) + ' of ' + paths.length + ' complete' });
        step(i + 1);
      });
  }
  step(0);
}

/* CSS.escape is not on the iOS 15 floor for every engine we care about. */
function cssEscape(value) {
  return String(value).replace(/["\\]/g, '\\$&');
}

/* ============================================================================
   QUICK PLAY
   ========================================================================= */

function quickPlay(path) {
  var clip = null;
  for (var i = 0; i < S.clips.length; i++) if (S.clips[i].path === path) clip = S.clips[i];
  if (!clip && S.day.clips) {
    for (var j = 0; j < S.day.clips.length; j++) if (S.day.clips[j].path === path) clip = S.day.clips[j];
  }
  if (!clip) return;

  var video = h('video.player__video', {
    src: api.clipUrl(clip.path),
    controls: true,
    autoplay: true,
    playsInline: true,
    preload: 'metadata',
    poster: thumbUrl(clip) || null,
    style: { width: '100%', borderRadius: '8px', background: '#000' }
  });
  var notice = h('p.player__notice', { hidden: true });
  on(video, 'error', function () {
    notice.hidden = false;
    notice.textContent = 'This clip would not play. The file may still be being written.';
  });

  var title = joinMeta(clip.species || 'Unclassified', clockTime(clip.time),
    dayLabel(clip.date).full);

  var handle = dialog({
    role: 'dialog',
    title: title,
    width: 860,
    content: h('div.stack', video, notice),
    actions: [
      { label: 'Open clip detail', variant: 'primary', value: 'open', focus: true },
      { label: 'Close', variant: 'secondary', value: null }
    ]
  });
  S.dialogs.push(handle);
  handle.result.then(function (value) {
    try { video.pause(); video.removeAttribute('src'); video.load(); } catch (e) {}
    if (S) {
      var k = S.dialogs.indexOf(handle);
      if (k >= 0) S.dialogs.splice(k, 1);
    }
    if (value === 'open') router.go('/clips/' + api.encodePath(clip.path), {});
  });
}

/* ============================================================================
   FILTER UI — the rail on desktop, a sheet on mobile. One builder.
   ========================================================================= */

function chipButton(spec) {
  var btn = h('button.chip', {
    type: 'button',
    'aria-pressed': spec.active ? 'true' : 'false',
    'data-value': spec.value
  });
  if (spec.dotClass) btn.appendChild(h('span.chip__dot', { 'class': spec.dotClass }));
  btn.appendChild(h('span', { text: spec.label }));
  if (spec.count !== undefined && spec.count !== null) {
    btn.appendChild(h('span.chip__count', { text: String(spec.count) }));
  }
  on(btn, 'click', spec.onClick);
  return btn;
}

function facetRows(list, selected) {
  var counts = {};
  (list || []).forEach(function (f) { counts[f.value] = f.count; });
  var rows = (list || []).map(function (f) {
    return { value: f.value, count: f.count };
  });
  /* A selected value the current universe no longer contains still needs its
     chip, or the user cannot switch it off. */
  (selected || []).forEach(function (v) {
    if (counts[v] === undefined) rows.push({ value: v, count: null });
  });
  return rows;
}

/**
 * mode.live === true  -> every change goes straight to the URL (desktop rail)
 * mode.live === false -> changes mutate a draft; the caller applies on Show
 */
function buildFilterFields(mode) {
  var draft = mode.live ? null : {
    cameras: S.filters.cameras.slice(),
    species: S.filters.species.slice(),
    from: S.filters.from, to: S.filters.to, q: S.filters.q, sort: S.filters.sort
  };
  function current() { return mode.live ? S.filters : draft; }
  function commit(next, opts) {
    if (mode.live) applyFilters(next, opts || { replace: false });
    else {
      draft.cameras = next.cameras; draft.species = next.species;
      draft.from = next.from; draft.to = next.to; draft.q = next.q; draft.sort = next.sort;
      if (mode.onDraft) mode.onDraft(draft);
      paint();
    }
  }
  function patch(part, opts) {
    var c = current();
    var next = {
      cameras: part.cameras || c.cameras.slice(),
      species: part.species || c.species.slice(),
      from: part.from !== undefined ? part.from : c.from,
      to: part.to !== undefined ? part.to : c.to,
      q: part.q !== undefined ? part.q : c.q,
      sort: part.sort !== undefined ? part.sort : c.sort
    };
    commit(next, opts);
  }

  var wrap = h('div.rail__fields');

  /* --- search --- */
  var searchInput = h('input.search__input', {
    type: 'search', autocomplete: 'off', spellcheck: 'false',
    placeholder: 'Species, camera or filename',
    value: current().q
  });
  searchInput.setAttribute('aria-label', 'Search this archive');
  var searchClear = h('button.search__clear', { type: 'button', 'aria-label': 'Clear the search' },
    icon('x', { size: 'sm' }));
  var searchField = h('div.search.search--rail',
    h('span.search__icon', icon('search', { size: 'sm' })),
    searchInput, searchClear);
  var searchTimer = null;
  on(searchInput, 'input', function () {
    searchField.classList.toggle('search--filled', !!searchInput.value);
    if (searchTimer) window.clearTimeout(searchTimer);
    searchTimer = window.setTimeout(function () {
      patch({ q: searchInput.value.trim() }, { replace: true });
    }, 280);
  });
  track(function () { if (searchTimer) window.clearTimeout(searchTimer); });
  on(searchClear, 'click', function () {
    searchInput.value = '';
    searchField.classList.remove('search--filled');
    patch({ q: '' });
    searchInput.focus();
  });
  wrap.appendChild(h('div.rail__section', h('div.rail__label', 'Search'), searchField));

  /* --- when --- */
  var rangeRow = h('div.chip-row.chip-row--wrap', { role: 'group', 'aria-label': 'Quick date ranges' });
  var fromInput = h('input.input', { type: 'date', value: current().from, 'aria-label': 'From date' });
  var toInput = h('input.input', { type: 'date', value: current().to, 'aria-label': 'To date' });
  on(fromInput, 'change', function () { patch({ from: fromInput.value }); });
  on(toInput, 'change', function () { patch({ to: toInput.value }); });
  var whenSection = h('div.rail__section',
    h('div.rail__label', 'When'),
    rangeRow,
    h('div.row',
      h('label.field', h('span.field__label', 'From'), fromInput),
      h('label.field', h('span.field__label', 'To'), toInput)));
  wrap.appendChild(whenSection);

  /* --- cameras --- */
  var camRow = h('div.chip-row.chip-row--wrap', { role: 'group', 'aria-label': 'Cameras' });
  wrap.appendChild(h('div.rail__section', h('div.rail__label', 'Cameras'), camRow));

  /* --- species --- */
  var spRow = h('div.chip-row.chip-row--wrap', { role: 'group', 'aria-label': 'Species' });
  wrap.appendChild(h('div.rail__section', h('div.rail__label', 'Species'), spRow));

  /* --- sort --- */
  var sortSel = h('select.select__el', { 'aria-label': 'Sort clips' });
  SORTS.forEach(function (s) {
    sortSel.appendChild(h('option', { value: s.value, text: s.label }));
  });
  on(sortSel, 'change', function () { patch({ sort: sortSel.value }); });
  wrap.appendChild(h('div.rail__section',
    h('div.rail__label', 'Sort'),
    h('div.select.select--rail', sortSel, h('span.select__chevron', icon('chevron-down', { size: 'sm' })))));

  /* --- reset --- */
  var reset = h('button.btn.btn--ghost.btn--block', { type: 'button' },
    h('span.btn__label', 'Reset every filter'));
  on(reset, 'click', function () {
    commit({ cameras: [], species: [], from: '', to: '', q: '', sort: 'newest' });
    if (mode.live) return;
    searchInput.value = ''; fromInput.value = ''; toInput.value = '';
  });
  var footCount = h('p.rail__foot', { 'aria-live': 'polite' });
  wrap.appendChild(h('div.rail__section', reset, footCount));

  function paint() {
    var c = current();

    if (document.activeElement !== searchInput) searchInput.value = c.q;
    searchField.classList.toggle('search--filled', !!c.q);
    if (document.activeElement !== fromInput) fromInput.value = c.from;
    if (document.activeElement !== toInput) toInput.value = c.to;
    sortSel.value = c.sort;

    keyedList(rangeRow, RANGES, {
      key: function (r) { return 'r' + r.value; },
      create: function (r) {
        return chipButton({
          value: r.value, label: r.label, active: false,
          onClick: function () {
            if (!r.value) patch({ from: '', to: '' });
            else patch({ from: shiftDays(parseInt(r.value, 10)), to: todayKey() });
          }
        });
      },
      update: function (el, r) {
        var active = r.value
          ? (c.from === shiftDays(parseInt(r.value, 10)) && (c.to === todayKey() || !c.to))
          : (!c.from && !c.to);
        el.setAttribute('aria-pressed', active ? 'true' : 'false');
      }
    });

    keyedList(camRow, facetRows(S.universe.cameras, c.cameras), {
      key: function (f) { return 'c' + f.value; },
      create: function (f) {
        return chipButton({
          value: f.value, label: f.value, count: f.count,
          active: c.cameras.indexOf(f.value) >= 0,
          onClick: function () {
            var list = current().cameras.slice();
            var i = list.indexOf(f.value);
            if (i >= 0) list.splice(i, 1); else list.push(f.value);
            patch({ cameras: list });
          }
        });
      },
      update: function (el, f) {
        el.setAttribute('aria-pressed', current().cameras.indexOf(f.value) >= 0 ? 'true' : 'false');
        var cnt = el.querySelector('.chip__count');
        if (cnt) cnt.textContent = f.count === null ? '0' : String(f.count);
      }
    });

    keyedList(spRow, facetRows(S.universe.species, c.species), {
      key: function (f) { return 's' + f.value; },
      create: function (f) {
        return chipButton({
          value: f.value, label: f.value, count: f.count,
          dotClass: speciesClass(f.value),
          active: current().species.indexOf(f.value) >= 0,
          onClick: function () {
            var list = current().species.slice();
            var i = list.indexOf(f.value);
            if (i >= 0) list.splice(i, 1); else list.push(f.value);
            patch({ species: list });
          }
        });
      },
      update: function (el, f) {
        el.setAttribute('aria-pressed', current().species.indexOf(f.value) >= 0 ? 'true' : 'false');
        var cnt = el.querySelector('.chip__count');
        if (cnt) cnt.textContent = f.count === null ? '0' : String(f.count);
        var dot = el.querySelector('.chip__dot');
        if (dot) dot.className = 'chip__dot ' + speciesClass(f.value);
      }
    });

    footCount.textContent = S.loading ? 'Counting…' : plural(S.total, 'clip') + ' match';
  }

  paint();
  return { el: wrap, paint: paint, draft: draft, focus: function () { searchInput.focus(); } };
}

function renderFilterUI() {
  if (S.railFields) S.railFields.paint();
  renderActiveChips();
}

function renderActiveChips() {
  var f = S.filters;
  var chips = [];
  f.cameras.forEach(function (v) { chips.push({ key: 'c' + v, label: v, kind: 'camera', value: v }); });
  f.species.forEach(function (v) { chips.push({ key: 's' + v, label: v, kind: 'species', value: v, dot: speciesClass(v) }); });
  if (f.from || f.to) {
    chips.push({ key: 'date', kind: 'date',
      label: f.from && f.to && f.from === f.to ? longDate(f.from)
        : (f.from ? 'From ' + f.from : '') + (f.to ? (f.from ? ' to ' : 'Up to ') + f.to : '') });
  }
  if (f.q) chips.push({ key: 'q', kind: 'q', label: '“' + f.q + '”' });

  S.els.chipRow.hidden = !chips.length;
  var items = chips.slice();
  if (chips.length) items.push({ key: '__clear__', kind: 'clear', label: 'Clear all' });

  keyedList(S.els.chipRow, items, {
    key: function (c) { return c.key; },
    create: function (c) {
      if (c.kind === 'clear') {
        var clearChip = h('button.chip.chip--clear', { type: 'button' },
          h('span', { text: 'Clear all' }));
        on(clearChip, 'click', function () {
          applyFilters({ cameras: [], species: [], from: '', to: '', q: '', sort: S.filters.sort });
        });
        return clearChip;
      }
      var chip = h('button.chip.chip--tonal', { type: 'button' });
      if (c.dot) chip.appendChild(h('span.chip__dot', { 'class': c.dot }));
      chip.appendChild(h('span', { text: c.label }));
      chip.appendChild(h('span.chip__x', icon('x', { size: 'sm' })));
      chip.setAttribute('aria-label', 'Remove filter ' + c.label);
      on(chip, 'click', function () {
        var f2 = S.filters;
        if (c.kind === 'camera') {
          applyFilters({ cameras: f2.cameras.filter(function (v) { return v !== c.value; }),
            species: f2.species, from: f2.from, to: f2.to, q: f2.q, sort: f2.sort });
        } else if (c.kind === 'species') {
          applyFilters({ cameras: f2.cameras,
            species: f2.species.filter(function (v) { return v !== c.value; }),
            from: f2.from, to: f2.to, q: f2.q, sort: f2.sort });
        } else if (c.kind === 'date') {
          applyFilters({ cameras: f2.cameras, species: f2.species, from: '', to: '', q: f2.q, sort: f2.sort });
        } else if (c.kind === 'q') {
          applyFilters({ cameras: f2.cameras, species: f2.species, from: f2.from, to: f2.to, q: '', sort: f2.sort });
        }
      });
      return chip;
    },
    update: function (el, c) {
      var label = el.querySelector('span:not(.chip__dot):not(.chip__x)');
      if (label) label.textContent = c.label;
    }
  });
}

function openFilterSheet() {
  var built = null;
  var handle = sheet({
    title: 'Filters',
    snap: 'full',
    content: function (body) {
      built = buildFilterFields({ live: false, onDraft: function () { updateFoot(); } });
      body.appendChild(built.el);
    },
    onClose: function () {
      var i = S.sheets.indexOf(handle);
      if (i >= 0) S.sheets.splice(i, 1);
    }
  });
  S.sheets.push(handle);

  var showBtn = h('button.btn.btn--primary.btn--block.sheet__apply', { type: 'button' },
    h('span.btn__label', 'Show clips'));
  var resetBtn = h('button.btn.btn--ghost.sheet__reset', { type: 'button' },
    h('span.btn__label', 'Reset'));
  var foot = h('div.sheet__foot.sheet__foot--filter', resetBtn, showBtn);
  handle.el.appendChild(foot);

  function updateFoot() { /* the count comes from the live archive, not the draft */ }

  on(showBtn, 'click', function () {
    if (built) applyFilters(built.draft);
    handle.close(null);
  });
  on(resetBtn, 'click', function () {
    applyFilters({ cameras: [], species: [], from: '', to: '', q: '', sort: 'newest' });
    handle.close(null);
  });
}

/* ============================================================================
   MONTH
   ========================================================================= */

function monthYearsFromCalendar() {
  /* KEYS ARE STRINGS IN DESCENDING INSERTION ORDER — never sorted numerically
     and never assumed ascending. Order is preserved exactly as delivered. */
  var out = [];
  var years = (S.calendar && S.calendar.years) || {};
  for (var y in years) {
    if (!Object.prototype.hasOwnProperty.call(years, y)) continue;
    var months = [];
    var mObj = years[y].months || {};
    for (var m in mObj) {
      if (!Object.prototype.hasOwnProperty.call(mObj, m)) continue;
      months.push({ key: m, month: parseInt(m, 10), total: mObj[m].total || 0, days: mObj[m].days || {} });
    }
    out.push({ key: y, year: parseInt(y, 10), total: years[y].total || 0, months: months });
  }
  return out;
}

function calendarMonthData(year, month) {
  var years = (S.calendar && S.calendar.years) || {};
  var y = years[String(year)];
  if (!y || !y.months) return null;
  var m = y.months[String(month)];
  return m || null;
}

function loadCalendar() {
  if (S.calendar || S.calendarLoading) return;
  S.calendarLoading = true;
  var signal = abortRequest('calendar');
  api.calendar({ signal: signal }).then(function (data) {
    S.calendarLoading = false;
    S.calendar = data;
    if (S.view === 'month') renderMonth();
  }).catch(function (err) {
    S.calendarLoading = false;
    if (api.isAbort(err)) return;
    S.calendarError = err;
    if (S.view === 'month') renderMonth();
    reportError('Could not load the calendar', err, function () { S.calendarError = null; loadCalendar(); });
  });
}

/**
 * The calendar endpoint takes no filters, so filtered cell counts are derived
 * from the month's own clips. "18 clips" therefore always means 18 MATCHING
 * clips, which is what the month view promises.
 */
function loadMonth(force) {
  var b = monthBounds(S.year, S.month);
  var token = S.year + '-' + S.month + '|' + JSON.stringify(S.filters);
  if (!force && S.monthToken === token && S.monthClips) return;
  S.monthToken = token;
  S.monthLoading = true;
  S.monthError = null;
  renderMonth();

  var signal = abortRequest('month');
  var f = S.filters;
  var q = apiQuery({
    cameras: f.cameras, species: f.species, q: f.q, sort: 'oldest',
    from: f.from && f.from > b.from ? f.from : b.from,
    to: f.to && f.to < b.to ? f.to : b.to
  }, { limit: MAX_PAGE, offset: 0 });

  api.recordings(q, { signal: signal }).then(function (data) {
    S.monthLoading = false;
    S.monthClips = data.clips || [];
    S.monthTruncated = !!data.has_more;
    renderMonth();
  }).catch(function (err) {
    if (api.isAbort(err)) return;
    S.monthLoading = false;
    S.monthError = err;
    renderMonth();
    reportError('Could not load this month', err, function () { loadMonth(true); });
  });
}

function monthDayIndex() {
  var index = {};
  (S.monthClips || []).forEach(function (c) {
    var key = c.date;
    if (!index[key]) index[key] = { count: 0, species: [], thumbs: [] };
    var slot = index[key];
    slot.count++;
    if (slot.species.indexOf(c.species) < 0) slot.species.push(c.species);
    var t = thumbUrl(c);
    if (t && slot.thumbs.length < 4) slot.thumbs.push(t);
  });
  return index;
}

function renderMonth() {
  var host = S.els.monthHost;
  if (!S.els.calendar) return;

  var cal = S.els.calendar;
  var index = monthDayIndex();
  var unfiltered = calendarMonthData(S.year, S.month);
  var filtered = filtersActive(S.filters);

  /* --- head --- */
  S.els.monthBtn.textContent = monthLabel(S.year, S.month);

  /* --- summary --- */
  var total = 0, speciesSet = {}, busiest = null, busiestCount = 0;
  for (var k in index) {
    if (!Object.prototype.hasOwnProperty.call(index, k)) continue;
    total += index[k].count;
    index[k].species.forEach(function (s) { speciesSet[s] = 1; });
    if (index[k].count > busiestCount) { busiestCount = index[k].count; busiest = k; }
  }
  var speciesCount = Object.keys(speciesSet).length;
  S.els.monthStats.textContent = S.monthLoading ? 'Counting this month…'
    : joinMeta(
      plural(total, 'clip'),
      speciesCount ? plural(speciesCount, 'species', 'species') : '',
      busiest ? 'busiest ' + longDate(busiest).replace(/,.*/, '') + ' ' +
        dateFromKey(busiest).getDate() + ' (' + busiestCount + ')' : '',
      filtered ? 'filtered' : '',
      S.monthTruncated ? 'first ' + MAX_PAGE + ' only' : '');

  /* --- grid --- */
  var first = new Date(S.year, S.month - 1, 1);
  var lead = first.getDay();
  var daysInMonth = new Date(S.year, S.month, 0).getDate();
  var today = todayKey();
  var cells = [];
  for (var b = 0; b < lead; b++) cells.push({ key: 'blank-' + b, blank: true });
  for (var d = 1; d <= daysInMonth; d++) {
    var key = keyFromDate(new Date(S.year, S.month - 1, d));
    var slot = index[key];
    var base = unfiltered && unfiltered.days ? unfiltered.days[String(d)] : null;
    cells.push({
      key: key, day: d, date: key,
      count: slot ? slot.count : (filtered ? 0 : (base ? base.count : 0)),
      species: slot ? slot.species : (base ? (base.species || []) : []),
      thumbs: slot ? slot.thumbs : [],
      future: key > today,
      loading: S.monthLoading
    });
  }

  keyedList(cal.grid, cells, {
    key: function (c) { return c.key; },
    create: function (c) {
      if (c.blank) return h('div.calendar__cell.calendar__cell--empty', { 'aria-hidden': 'true' });
      var cell = h('button.calendar__cell', {
        type: 'button', 'data-date': c.date, tabIndex: -1
      },
        h('span.calendar__num', { text: String(c.day) }),
        h('span.calendar__count'),
        h('span.calendar__dots'),
        h('span.calendar__strip'));
      return cell;
    },
    update: function (cell, c) {
      if (c.blank) return;
      var num = cell.querySelector('.calendar__num');
      var count = cell.querySelector('.calendar__count');
      var dots = cell.querySelector('.calendar__dots');
      var strip = cell.querySelector('.calendar__strip');
      num.textContent = String(c.day);
      cell.classList.toggle('calendar__cell--skel', !!c.loading);
      cell.classList.toggle('calendar__cell--empty', !c.count && !c.loading);
      count.textContent = c.loading ? '' : (c.count ? String(c.count) : '');
      cell.setAttribute('aria-disabled', c.future ? 'true' : 'false');
      if (c.date === today) cell.setAttribute('aria-current', 'date');
      else cell.removeAttribute('aria-current');
      cell.setAttribute('aria-pressed', S.day.date === c.date ? 'true' : 'false');
      cell.setAttribute('aria-label', longDate(c.date) + ', ' +
        (c.count ? plural(c.count, 'clip') : 'no clips'));
      cell.tabIndex = (S.monthFocus === c.date) ? 0 : -1;

      keyedList(dots, c.species.slice(0, 3).map(function (sp) { return { sp: sp }; }), {
        key: function (x) { return x.sp; },
        create: function (x) { return h('span.sp-dot', { 'class': speciesClass(x.sp) }); }
      });
      keyedList(strip, c.thumbs.map(function (u, i) { return { u: u, i: i }; }), {
        key: function (x) { return 't' + x.i; },
        create: function (x) {
          return h('span.frame.frame--sm',
            h('img.frame__img', { src: x.u, alt: '', loading: 'lazy', decoding: 'async' }));
        },
        update: function (el, x) {
          var img = el.querySelector('img');
          if (img && img.getAttribute('src') !== x.u) img.setAttribute('src', x.u);
        }
      });
    }
  });

  /* Keep a focusable cell even after a month change, and carry the keyboard
     caret across the month boundary it just crossed. */
  var focusable = cal.grid.querySelector('.calendar__cell[tabindex="0"]');
  if (!focusable) {
    var candidate = cal.grid.querySelector('.calendar__cell');
    if (candidate) { candidate.tabIndex = 0; S.monthFocus = candidate.getAttribute('data-date'); }
  }
  if (S.pendingMonthFocus) {
    var want = cal.grid.querySelector('.calendar__cell[data-date="' + cssEscape(S.pendingMonthFocus) + '"]');
    S.pendingMonthFocus = null;
    if (want) { want.tabIndex = 0; want.focus(); }
  }

  /* --- empty month --- */
  var noCells = !total && !S.monthLoading;
  S.els.monthEmpty.hidden = !noCells;
  if (noCells) {
    clear(S.els.monthEmpty).appendChild(S.monthError
      ? emptyState({
        error: true, icon: 'alert',
        title: 'This month did not load',
        body: api.describe(S.monthError),
        endpoint: '/api/recordings',
        actions: [{ label: 'Retry', variant: 'primary', onClick: function () { loadMonth(true); } }]
      })
      : emptyState({
        icon: 'calendar',
        title: 'No clips in ' + monthLabel(S.year, S.month),
        body: filtered
          ? 'You are filtered to ' + describeFilters(S.filters) + '. Another month may still match.'
          : 'Nothing was recorded in this month.',
        cause: filtered ? staleCause(S.filters) || undefined : undefined,
        actions: filtered ? [
          { label: 'Clear filters', variant: 'primary', onClick: function () {
            applyFilters({ cameras: [], species: [], from: '', to: '', q: '', sort: S.filters.sort });
          } },
          { label: 'Previous month', onClick: function () { stepMonth(-1); } }
        ] : [
          { label: 'Previous month', variant: 'primary', onClick: function () { stepMonth(-1); } }
        ]
      }));
  } else {
    clear(S.els.monthEmpty);
  }

  renderDayPanel();
  renderCount();
  if (host) host.hidden = S.view !== 'month';
}

function stepMonth(delta) {
  var m = S.month + delta;
  var y = S.year;
  while (m < 1) { m += 12; y--; }
  while (m > 12) { m -= 12; y++; }
  router.setQuery({ year: String(y), month: String(m) });
}

function openMonthPicker(anchor) {
  var years = monthYearsFromCalendar();
  var items = [];
  if (!years.length) {
    items.push({ label: 'The calendar has not loaded yet', disabled: true });
  }
  years.forEach(function (y) {
    items.push({ group: y.key + ' · ' + plural(y.total, 'clip') });
    y.months.forEach(function (m) {
      items.push({
        label: monthLabel(y.year, m.month) + ' · ' + plural(m.total, 'clip'),
        checked: y.year === S.year && m.month === S.month,
        onSelect: function () {
          router.setQuery({ year: String(y.year), month: String(m.month) });
        }
      });
    });
  });
  menu(anchor, items, { label: 'Jump to a month' });
}

/* --- the day panel ------------------------------------------------------- */

function openDay(date) {
  router.setQuery({ date: date });
}

function closeDay() {
  if (!S || S.unmounting) return;
  router.setQuery({ date: null });
}

function loadDay(date) {
  if (!date) { S.day = { date: '', clips: null, loading: false, error: null }; return; }
  if (S.day.date === date && (S.day.clips || S.day.loading)) return;
  S.day = { date: date, clips: null, loading: true, error: null };
  renderDayPanel();

  var signal = abortRequest('day');
  var f = S.filters;
  var q = {};
  /* The server matches one value per key and species as a substring, so a
     multi-select is narrowed here rather than half-applied there. */
  if (f.cameras.length === 1) q.camera = f.cameras[0];
  if (f.species.length === 1) q.species = f.species[0];

  api.day(date, q, { signal: signal }).then(function (data) {
    var clips = (data.clips || []).map(function (c) { return normaliseDayClip(c, date); });
    /* THE FIX: the old UI ignored the active filters here, so filtering to
       Deer and opening a day showed everything. */
    clips = clips.filter(function (c) {
      if (f.cameras.length && f.cameras.indexOf(c.camera) < 0) return false;
      if (f.species.length && f.species.indexOf(c.species) < 0) return false;
      if (f.q) {
        var hay = ((c.species || '') + ' ' + (c.camera || '') + ' ' + (c.filename || '')).toLowerCase();
        if (hay.indexOf(f.q.toLowerCase()) < 0) return false;
      }
      return true;
    });
    S.day = { date: date, clips: clips, loading: false, error: null, summary: data.summary || null };
    renderDayPanel();
  }).catch(function (err) {
    if (api.isAbort(err)) return;
    S.day = { date: date, clips: null, loading: false, error: err };
    renderDayPanel();
    reportError('Could not load ' + longDate(date), err, function () {
      S.day = { date: '', clips: null, loading: false, error: null };
      loadDay(date);
    });
  });
}

function buildDayPanel() {
  var title = h('h2.daypanel__title');
  var count = h('span.daypanel__count', { 'aria-live': 'polite' });
  var closeBtn = h('button.icon-btn', { type: 'button', 'aria-label': 'Close the day panel' },
    icon('x', { size: 'sm' }));
  var body = h('div.daypanel__body');
  var openGrid = h('button.btn.btn--primary.btn--block', { type: 'button' },
    h('span.btn__label', 'Open this day in Grid'));
  var foot = h('div.daypanel__foot', openGrid);
  var panel = h('aside.daypanel', { role: 'region', 'aria-label': 'Clips for the selected day' },
    h('div.daypanel__head', title, count, closeBtn), body, foot);

  track(on(closeBtn, 'click', closeDay));
  track(on(openGrid, 'click', function () {
    var date = S.day.date;
    if (!date) return;
    router.setQuery({ view: null, date: null, from: date, to: date });
  }));

  panel._parts = { title: title, count: count, body: body, foot: foot };
  return panel;
}

function renderDayPanelInto(panel) {
  var p = panel._parts;
  var d = S.day;
  p.title.textContent = d.date ? longDate(d.date) : 'No day selected';
  p.count.textContent = d.loading ? '…' : (d.clips ? plural(d.clips.length, 'clip') : '');
  p.foot.hidden = !d.date;

  if (d.loading) {
    var stubs = [];
    for (var i = 0; i < 4; i++) stubs.push({ key: 'ds' + i });
    var skelList = p.body.querySelector('.clipgrid');
    if (!skelList) { clear(p.body); skelList = h('ul.clipgrid.clipgrid--compact', { role: 'list' }); p.body.appendChild(skelList); }
    keyedList(skelList, stubs, { key: function (s) { return s.key; }, create: buildSkeletonCard });
    return;
  }

  if (d.error) {
    clear(p.body).appendChild(emptyState({
      error: true, icon: 'alert',
      title: 'That day did not load',
      body: api.describe(d.error),
      endpoint: '/api/recordings/day/' + d.date,
      actions: [{ label: 'Retry', variant: 'primary', onClick: function () {
        var date = d.date; S.day = { date: '', clips: null, loading: false, error: null }; loadDay(date);
      } }]
    }));
    return;
  }

  if (!d.clips || !d.clips.length) {
    clear(p.body).appendChild(emptyState({
      icon: 'filter',
      title: 'Nothing on this day matches',
      body: filtersActive(S.filters)
        ? 'You are filtered to ' + describeFilters(S.filters) + '. The day itself may still hold clips.'
        : 'No clip was recorded on ' + (d.date ? longDate(d.date) : 'this day') + '.',
      actions: filtersActive(S.filters) ? [{
        label: 'Clear filters', variant: 'primary', onClick: function () {
          applyFilters({ cameras: [], species: [], from: '', to: '', q: '', sort: S.filters.sort });
        }
      }] : []
    }));
    return;
  }

  var list = p.body.querySelector('.clipgrid');
  if (!list) {
    clear(p.body);
    list = h('ul.clipgrid.clipgrid--compact', { role: 'list' });
    p.body.appendChild(list);
  }
  list.className = 'clipgrid clipgrid--compact' + (S.selectMode ? ' clipgrid--select' : '');
  keyedList(list, d.clips, {
    key: function (c) { return c.path; },
    create: buildCard,
    update: updateCard
  });
}

function renderDayPanel() {
  if (!S.day.date) {
    if (S.els.dayPanel && S.els.dayPanel.parentNode) S.els.dayPanel.parentNode.removeChild(S.els.dayPanel);
    if (S.daySheet) { var sh = S.daySheet; S.daySheet = null; sh.close(null); }
    return;
  }

  if (isDesktop()) {
    if (S.daySheet) { var sh2 = S.daySheet; S.daySheet = null; sh2.close(null); }
    if (!S.els.dayPanel) S.els.dayPanel = buildDayPanel();
    if (S.els.dayPanel.parentNode !== S.els.monthRow) S.els.monthRow.appendChild(S.els.dayPanel);
    renderDayPanelInto(S.els.dayPanel);
    return;
  }

  if (S.els.dayPanel && S.els.dayPanel.parentNode) {
    S.els.dayPanel.parentNode.removeChild(S.els.dayPanel);
  }
  if (!S.daySheet) {
    var panel = buildDayPanel();
    S.sheetPanel = panel;
    var handle = sheet({
      title: longDate(S.day.date),
      snap: 'half',
      content: function (body) { body.appendChild(panel); },
      onClose: function () {
        S.daySheet = null;
        S.sheetPanel = null;
        if (S.day.date) closeDay();
      }
    });
    S.daySheet = handle;
    S.sheets.push(handle);
  }
  if (S.sheetPanel) renderDayPanelInto(S.sheetPanel);
}

/* --- month keyboard grid ------------------------------------------------- */

function moveMonthFocus(delta, unit) {
  var current = S.monthFocus ? dateFromKey(S.monthFocus) : new Date(S.year, S.month - 1, 1);
  if (!current) current = new Date(S.year, S.month - 1, 1);
  var next = new Date(current.getTime());
  if (unit === 'day') next.setDate(next.getDate() + delta);
  else if (unit === 'week') next.setDate(next.getDate() + delta * 7);
  else if (unit === 'month') next.setMonth(next.getMonth() + delta);
  else if (unit === 'year') next.setFullYear(next.getFullYear() + delta);

  var key = keyFromDate(next);
  S.monthFocus = key;
  if (next.getFullYear() !== S.year || next.getMonth() + 1 !== S.month) {
    S.pendingMonthFocus = key;
    router.setQuery({ year: String(next.getFullYear()), month: String(next.getMonth() + 1) });
    return;
  }
  renderMonth();
  var cell = S.els.calendar.grid.querySelector('.calendar__cell[data-date="' + cssEscape(key) + '"]');
  if (cell) cell.focus();
}

function monthKeydown(ev) {
  var map = {
    ArrowLeft: [-1, 'day'], ArrowRight: [1, 'day'],
    ArrowUp: [-1, 'week'], ArrowDown: [1, 'week']
  };
  if (map[ev.key]) {
    ev.preventDefault();
    moveMonthFocus(map[ev.key][0], map[ev.key][1]);
    return;
  }
  if (ev.key === 'PageUp' || ev.key === 'PageDown') {
    ev.preventDefault();
    var dir = ev.key === 'PageUp' ? -1 : 1;
    moveMonthFocus(dir, ev.shiftKey ? 'year' : 'month');
    return;
  }
  if (ev.key === 'Home' || ev.key === 'End') {
    ev.preventDefault();
    var cur = S.monthFocus ? dateFromKey(S.monthFocus) : new Date(S.year, S.month - 1, 1);
    var dow = cur.getDay();
    moveMonthFocus(ev.key === 'Home' ? -dow : (6 - dow), 'day');
  }
}

/* ============================================================================
   GRID KEYBOARD
   ========================================================================= */

function gridColumns() {
  var list = S.els.groups.querySelector('.clipgrid');
  if (!list) return 1;
  var cols = window.getComputedStyle(list).getPropertyValue('grid-template-columns');
  var n = String(cols).trim().split(/\s+/).filter(function (x) { return !!x; }).length;
  return Math.max(1, n);
}

function focusCard(path) {
  S.focusKey = path;
  renderGrid();
  var link = S.els.groups.querySelector('.clip__open[data-path="' + cssEscape(path) + '"]');
  if (link) { link.tabIndex = 0; link.focus(); }
}

function gridKeydown(ev) {
  var clips = visibleClips();
  if (!clips.length) return;
  var idx = -1;
  for (var i = 0; i < clips.length; i++) if (clips[i].path === S.focusKey) idx = i;
  if (idx < 0) idx = 0;

  var cols = gridColumns();
  var next = null;

  if (ev.key === 'ArrowRight') next = idx + 1;
  else if (ev.key === 'ArrowLeft') next = idx - 1;
  else if (ev.key === 'ArrowDown') next = idx + cols;
  else if (ev.key === 'ArrowUp') next = idx - cols;
  else if (ev.key === 'PageDown') next = idx + cols * 3;
  else if (ev.key === 'PageUp') next = idx - cols * 3;
  else if (ev.key === 'Home') next = 0;
  else if (ev.key === 'End') next = clips.length - 1;
  else if (ev.key === ' ' || ev.key === 'Spacebar') {
    ev.preventDefault();
    toggleSelection(clips[idx].path, ev.shiftKey);
    return;
  } else if (ev.key === 'Escape' && S.selectMode) {
    ev.preventDefault();
    setSelectMode(false);
    return;
  } else if ((ev.metaKey || ev.ctrlKey) && (ev.key === 'a' || ev.key === 'A')) {
    ev.preventDefault();
    if (!S.selectMode) setSelectMode(true);
    clips.forEach(function (c) { S.selected.add(c.path); });
    renderGrid();
    return;
  } else {
    return;
  }

  ev.preventDefault();
  next = Math.max(0, Math.min(clips.length - 1, next));
  if (ev.shiftKey && (ev.key.indexOf('Arrow') === 0)) {
    if (!S.selectMode) setSelectMode(true);
    if (!S.anchor) S.anchor = clips[idx].path;
    S.selected.add(clips[idx].path);
    S.selected.add(clips[next].path);
  }
  focusCard(clips[next].path);
  if (next >= clips.length - cols && S.hasMore) loadMore();
}

/* ============================================================================
   CHROME ACTIONS
   ========================================================================= */

function renderChromeActions() {
  var filterBtn = S.els.filterBtn;
  var activeCount = S.filters.cameras.length + S.filters.species.length +
    (S.filters.from || S.filters.to ? 1 : 0) + (S.filters.q ? 1 : 0);
  var badge = filterBtn.querySelector('.btn__count');
  if (activeCount) {
    if (!badge) { badge = h('span.btn__count'); filterBtn.appendChild(badge); }
    badge.textContent = String(activeCount);
    filterBtn.setAttribute('aria-label', 'Filters, ' + activeCount + ' active');
  } else {
    if (badge) badge.remove();
    filterBtn.setAttribute('aria-label', 'Filters');
  }
  clear(S.els.selectBtn).appendChild(h('span.btn__label', {
    text: S.selectMode ? 'Cancel' : 'Select'
  }));
  S.els.selectBtn.setAttribute('aria-pressed', S.selectMode ? 'true' : 'false');
  renderCount();
}

/* ============================================================================
   VIEW SWITCHING
   ========================================================================= */

function renderMode() {
  var isMonth = S.view === 'month';
  S.els.gridHost.hidden = isMonth;
  S.els.monthHost.hidden = !isMonth;
  S.els.segGrid.setAttribute('aria-pressed', isMonth ? 'false' : 'true');
  S.els.segMonth.setAttribute('aria-pressed', isMonth ? 'true' : 'false');
  S.els.densityBtn.hidden = isMonth;
  S.els.sortBtn.hidden = isMonth;
  clear(S.els.sortBtn).appendChild(h('span.btn__label', { text: sortLabel(S.filters.sort) }));
  S.els.sortBtn.appendChild(icon('chevron-down', { size: 'sm', 'class': 'btn__icon' }));
  S.els.densityBtn.setAttribute('aria-pressed', S.density === 'compact' ? 'true' : 'false');
}

function sortLabel(value) {
  for (var i = 0; i < SORTS.length; i++) if (SORTS[i].value === value) return SORTS[i].label;
  return 'Newest first';
}

function openSortMenu(anchor) {
  menu(anchor, SORTS.map(function (s) {
    return {
      label: s.label,
      checked: S.filters.sort === s.value,
      onSelect: function () { router.setQuery({ sort: s.value === 'newest' ? null : s.value }); }
    };
  }), { label: 'Sort clips' });
}

/* ============================================================================
   MOUNT
   ========================================================================= */

function buildShell(root) {
  var live = h('p.visually-hidden', { 'aria-live': 'polite', 'aria-atomic': 'true' });

  var segGrid = h('button.seg__btn', { type: 'button', 'aria-pressed': 'true' },
    icon('grid', { size: 'sm' }), h('span', 'Grid'));
  var segMonth = h('button.seg__btn', { type: 'button', 'aria-pressed': 'false' },
    icon('calendar', { size: 'sm' }), h('span', 'Month'));
  var seg = h('div.seg', { role: 'group', 'aria-label': 'Archive layout' }, segGrid, segMonth);

  var sortBtn = h('button.btn.btn--secondary.btn--sm', { type: 'button', 'aria-haspopup': 'menu' },
    h('span.btn__label', 'Newest first'));
  var densityBtn = h('button.icon-btn', {
    type: 'button', 'aria-pressed': 'false', 'aria-label': 'Toggle compact density'
  }, icon('list', { size: 'sm' }));
  var count = h('span.count', { 'aria-live': 'polite' });

  var toolbar = h('div.row.row--between.row--wrap', seg,
    h('span.spacer'), sortBtn, densityBtn, count);

  var chipRow = h('div.chip-row.chip-row--wrap', { role: 'group', 'aria-label': 'Active filters' });
  chipRow.hidden = true;

  var groups = h('div', { id: 'at-rec-groups' });
  var emptyHost = h('div');
  var more = h('button.btn.btn--secondary.btn--block', { type: 'button' },
    h('span.btn__label', 'Load more'));
  var sentinel = h('div', { 'aria-hidden': 'true', style: { height: '1px' } });
  var gridHost = h('div.stack', groups, emptyHost, sentinel, more);

  var monthPrev = h('button.calendar__nav', { type: 'button', 'aria-label': 'Previous month' },
    icon('chevron-left', { size: 'sm' }));
  var monthNext = h('button.calendar__nav', { type: 'button', 'aria-label': 'Next month' },
    icon('chevron-right', { size: 'sm' }));
  var monthBtn = h('button.calendar__month', { type: 'button', 'aria-haspopup': 'menu' }, 'Month');
  var dow = h('div.calendar__dow', { 'aria-hidden': 'true' });
  DAY_NAMES_SHORT.forEach(function (d) { dow.appendChild(h('span', { text: d })); });
  var grid = h('div.calendar__grid', { role: 'group', 'aria-label': 'Days of the month' });
  var monthStats = h('p.t-sm.t-3', { 'aria-live': 'polite' });
  var monthEmpty = h('div');
  monthEmpty.hidden = true;
  var calendar = h('div.calendar',
    h('div.calendar__head', monthPrev, monthBtn, monthNext),
    monthStats, dow, grid, monthEmpty);
  var monthRow = h('div', { style: { display: 'flex', gap: '16px', alignItems: 'stretch' } },
    h('div', { style: { flex: '1 1 auto', minWidth: '0' } }, calendar));
  var monthHost = h('div.stack', monthRow);
  monthHost.hidden = true;

  root.appendChild(h('h1.visually-hidden', { tabIndex: -1 }, 'Recordings'));
  root.appendChild(live);
  root.appendChild(h('div.stack', toolbar, chipRow, gridHost, monthHost));

  S.els.live = live;
  S.els.seg = seg;
  S.els.segGrid = segGrid;
  S.els.segMonth = segMonth;
  S.els.sortBtn = sortBtn;
  S.els.densityBtn = densityBtn;
  S.els.count = count;
  S.els.chipRow = chipRow;
  S.els.groups = groups;
  S.els.emptyHost = emptyHost;
  S.els.more = more;
  S.els.sentinel = sentinel;
  S.els.gridHost = gridHost;
  S.els.monthHost = monthHost;
  S.els.monthRow = monthRow;
  S.els.monthStats = monthStats;
  S.els.monthEmpty = monthEmpty;
  S.els.monthBtn = monthBtn;
  S.els.calendar = { grid: grid, prev: monthPrev, next: monthNext, btn: monthBtn };
}

function wireShell() {
  var els = S.els;

  track(on(els.segGrid, 'click', function () { router.setQuery({ view: null, date: null }); }));
  track(on(els.segMonth, 'click', function () { router.setQuery({ view: 'month' }); }));
  track(on(els.sortBtn, 'click', function () { openSortMenu(els.sortBtn); }));
  track(on(els.densityBtn, 'click', function () {
    router.setQuery({ density: S.density === 'compact' ? null : 'compact' });
  }));
  track(on(els.more, 'click', loadMore));

  track(on(els.calendar.prev, 'click', function () { stepMonth(-1); }));
  track(on(els.calendar.next, 'click', function () { stepMonth(1); }));
  track(on(els.monthBtn, 'click', function () { openMonthPicker(els.monthBtn); }));
  track(delegate(els.calendar.grid, 'click', '.calendar__cell', function (ev, cell) {
    var date = cell.getAttribute('data-date');
    if (!date) return;
    S.monthFocus = date;
    openDay(date);
  }));
  track(on(els.calendar.grid, 'keydown', monthKeydown));
  track(delegate(els.calendar.grid, 'focusin', '.calendar__cell', function (ev, cell) {
    var d = cell.getAttribute('data-date');
    if (d) S.monthFocus = d;
  }));

  /* --- card interaction: one delegated handler per event ---------------- */
  track(delegate(els.groups, 'click', '.clip__open', function (ev, link) {
    var path = link.getAttribute('data-path');
    if (S.suppressClick) { ev.preventDefault(); S.suppressClick = false; return; }
    if (S.selectMode) { ev.preventDefault(); toggleSelection(path, ev.shiftKey); return; }
    if (ev.shiftKey) { ev.preventDefault(); toggleSelection(path, false); return; }
    S.focusKey = path;   /* cmd-click and middle-click fall through to the browser */
  }));

  track(delegate(els.groups, 'change', '.clip__check', function (ev, box) {
    var path = box.getAttribute('data-path');
    if (box.checked) { S.selected.add(path); S.anchor = path; }
    else S.selected.delete(path);
    renderGrid();
  }));

  track(delegate(els.groups, 'click', '[data-play]', function (ev, btn) {
    ev.preventDefault();
    ev.stopPropagation();
    quickPlay(btn.getAttribute('data-play'));
  }));

  track(delegate(els.groups, 'click', '.dayhead__toggle', function (ev, btn) {
    var key = btn.getAttribute('data-day');
    if (S.collapsed.has(key)) S.collapsed.delete(key);
    else S.collapsed.add(key);
    renderGrid();
  }));

  track(delegate(els.groups, 'focusin', '.clip__open', function (ev, link) {
    var p = link.getAttribute('data-path');
    if (p) S.focusKey = p;
  }));

  track(on(els.groups, 'keydown', gridKeydown));

  /* Long-press: 500 ms enters selection mode and takes the pressed card. */
  track(delegate(els.groups, 'pointerdown', '.clip', function (ev, card) {
    if (ev.button !== undefined && ev.button !== 0) return;
    var link = card.querySelector('.clip__open');
    if (!link) return;
    var path = link.getAttribute('data-path');
    S.pressX = ev.clientX; S.pressY = ev.clientY;
    if (S.pressTimer) window.clearTimeout(S.pressTimer);
    S.pressTimer = window.setTimeout(function () {
      S.pressTimer = null;
      S.suppressClick = true;
      if (navigator.vibrate) { try { navigator.vibrate(10); } catch (e) {} }
      if (!S.selectMode) setSelectMode(true);
      toggleSelection(path, false);
    }, LONG_PRESS_MS);
  }));
  function cancelPress(ev) {
    if (!S.pressTimer) return;
    if (ev && ev.clientX !== undefined &&
        Math.abs(ev.clientX - S.pressX) < 10 && Math.abs(ev.clientY - S.pressY) < 10 &&
        ev.type === 'pointermove') return;
    window.clearTimeout(S.pressTimer);
    S.pressTimer = null;
  }
  track(on(els.groups, 'pointerup', cancelPress));
  track(on(els.groups, 'pointercancel', cancelPress));
  track(on(els.groups, 'pointermove', cancelPress));
  track(on(window, 'scroll', function () { cancelPress(null); }, { passive: true }));

  /* Day-panel cards live outside .groups; give them the same handlers. */
  track(delegate(document.body, 'click', '.daypanel .clip__open', function (ev, link) {
    if (!S.selectMode) return;
    ev.preventDefault();
    toggleSelection(link.getAttribute('data-path'), ev.shiftKey);
  }));
  track(delegate(document.body, 'click', '.daypanel [data-play]', function (ev, btn) {
    ev.preventDefault();
    ev.stopPropagation();
    quickPlay(btn.getAttribute('data-play'));
  }));
}

function installObserver() {
  if (typeof IntersectionObserver !== 'function') return;
  S.io = new IntersectionObserver(function (entries) {
    for (var i = 0; i < entries.length; i++) {
      if (entries[i].isIntersecting) { loadMore(); return; }
    }
  }, { rootMargin: '800px 0px' });
  S.io.observe(S.els.sentinel);
}

function installVisibility() {
  track(on(document, 'visibilitychange', function () {
    var visible = document.visibilityState !== 'hidden';
    store.set({ visible: visible });
    if (visible) {
      startRefresh();
      refreshGrid();
    } else {
      stopRefresh();
      /* No MJPEG lives in this view, but any <video> a quick-play opened is
         paused so a hidden tab is not decoding frames. */
      var vids = document.querySelectorAll('.dialog video');
      for (var i = 0; i < vids.length; i++) { try { vids[i].pause(); } catch (e) {} }
    }
  }));
}

function startRefresh() {
  stopRefresh();
  S.refreshTimer = window.setInterval(function () {
    if (S.view === 'grid') refreshGrid();
  }, REFRESH_MS);
  S.intervals.push(S.refreshTimer);
}

function stopRefresh() {
  if (S.refreshTimer) { window.clearInterval(S.refreshTimer); S.refreshTimer = null; }
}

function applyCtx(ctx, first) {
  var next = readState(ctx);
  var filtersChanged = first || !sameFilters(next.filters, S.filters);
  var viewChanged = first || next.view !== S.view;
  var monthChanged = first || next.year !== S.year || next.month !== S.month;
  var dateChanged = first || next.date !== S.date;

  S.ctx = ctx;
  S.filters = next.filters;
  S.view = next.view;
  S.density = next.density;
  S.year = next.year;
  S.month = next.month;
  S.date = next.date;

  if (filtersChanged) {
    S.selected.clear();
    S.anchor = null;
    /* The day panel is filtered too, so its payload is no longer valid. */
    S.day = { date: '', clips: null, loading: false, error: null };
  }

  renderMode();
  renderChromeActions();
  renderActiveChips();
  if (S.railFields) S.railFields.paint();

  if (S.view === 'grid') {
    if (filtersChanged) loadGrid(true);
    else if (viewChanged && !S.clips.length && !S.loading) loadGrid(true);
    else renderGrid();
  } else {
    loadCalendar();
    if (filtersChanged || monthChanged) loadMonth(true);
    else renderMonth();
    /* The grid model still backs the counts and the selection bar. */
    if (filtersChanged) loadGrid(true);
  }

  if (S.view !== 'month' || !S.date) {
    /* Leaving the month view (or the day) closes the panel and its sheet. */
    if (S.day.date) S.day = { date: '', clips: null, loading: false, error: null };
    renderDayPanel();
  } else if (dateChanged || filtersChanged) {
    loadDay(S.date);
  }
}

export const view = {
  mount: function (root, ctx) {
    S = {
      root: root,
      ctx: ctx,
      els: {},
      cleanups: [],
      timers: [],
      intervals: [],
      aborts: {},
      dialogs: [],
      sheets: [],
      daySheet: null,
      sheetPanel: null,
      io: null,
      refreshTimer: null,
      pressTimer: null,
      suppressClick: false,
      unmounting: false,
      pendingMonthFocus: null,

      filters: { cameras: [], species: [], from: '', to: '', q: '', sort: 'newest' },
      view: 'grid',
      density: 'comfortable',
      year: new Date().getFullYear(),
      month: new Date().getMonth() + 1,
      date: '',

      clips: [],
      total: 0,
      archiveTotal: 0,
      offset: 0,
      hasMore: false,
      loading: true,
      loadingMore: false,
      error: null,
      facets: { cameras: [], species: [] },
      universe: { cameras: [], species: [] },

      collapsed: new Set(),
      collapsing: new Set(),
      selected: new Set(),
      pendingDelete: new Set(),
      selectMode: false,
      anchor: null,
      focusKey: null,

      calendar: null,
      calendarLoading: false,
      calendarError: null,
      monthClips: null,
      monthLoading: false,
      monthError: null,
      monthToken: '',
      monthTruncated: false,
      monthFocus: null,
      day: { date: '', clips: null, loading: false, error: null }
    };

    buildShell(root);

    /* Chrome: the mobile actions, the desktop rail, the selection bar. */
    var filterBtn = h('button.btn.btn--secondary.btn--sm', { type: 'button' },
      icon('filter', { size: 'sm', 'class': 'btn__icon' }),
      h('span.btn__label', 'Filters'));
    var selectBtn = h('button.btn.btn--ghost.btn--sm', { type: 'button', 'aria-pressed': 'false' },
      h('span.btn__label', 'Select'));
    S.els.filterBtn = filterBtn;
    S.els.selectBtn = selectBtn;
    S.chromeActions = [filterBtn, selectBtn];
    track(on(filterBtn, 'click', openFilterSheet));
    track(on(selectBtn, 'click', function () { setSelectMode(!S.selectMode); }));

    S.railFields = buildFilterFields({ live: true });
    S.els.rail = S.railFields.el;
    S.els.selbar = buildSelbar();

    wireShell();
    installObserver();
    installVisibility();
    startRefresh();

    applyCtx(ctx, true);
  },

  /** Same route, new query: re-read and diff. Scroll and focus survive. */
  update: function (ctx) {
    if (!S) return;
    applyCtx(ctx, false);
  },

  unmount: function () {
    if (!S) return;
    S.unmounting = true;

    /* An undo bar cannot outlive the screen that offered it: every pending
       deadline fires now, which commits the deferred deletes. */
    toast.flush();

    S.cleanups.forEach(function (off) { try { off(); } catch (e) {} });
    S.timers.forEach(function (id) { window.clearTimeout(id); });
    S.intervals.forEach(function (id) { if (id) window.clearInterval(id); });
    if (S.refreshTimer) window.clearInterval(S.refreshTimer);
    if (S.pressTimer) window.clearTimeout(S.pressTimer);
    if (S.io) { try { S.io.disconnect(); } catch (e) {} }

    for (var name in S.aborts) {
      if (!Object.prototype.hasOwnProperty.call(S.aborts, name)) continue;
      var c = S.aborts[name];
      if (c) { try { c.abort(); } catch (e2) {} }
    }

    S.dialogs.forEach(function (d) { try { d.close(null); } catch (e3) {} });
    S.sheets.forEach(function (s) { try { s.close(null); } catch (e4) {} });

    var vids = document.querySelectorAll('.dialog video, .sheet video');
    for (var i = 0; i < vids.length; i++) {
      try { vids[i].pause(); vids[i].removeAttribute('src'); vids[i].load(); } catch (e5) {}
    }

    if (S.root) clear(S.root);
    S = null;
  }
};
