/* ============================================================================
   views/monitor.js — system health and logs.  Route: /app/monitor

   WHAT THIS SCREEN OWES THE OPERATOR
   · Resource gauges (CPU / memory / disk / GPU) with a 60-sample sparkline,
     a threshold WORD as well as a colour, and a once-per-crossing error toast
     rather than one toast every two seconds.
   · A per-camera pipeline card built from .readout + .meter: buffer fill,
     event state, active tracks, connection status.
   · Live reanalyze (reprocessing) progress. This is the ONLY page in the app
     that shows it today, so it is preserved verbatim.
   · The log viewer with EVERY filter the old page had — camera, level, type,
     relative range, custom from/to, entry limit and timezone — plus Refresh,
     Copy, per-row copy and a client-side text filter.

   THE TIMEZONE KEY IS LOAD-BEARING
   The old page wrote localStorage['logTimezone'] as a RAW string ('local',
   'UTC', 'America/Chicago'). core/store.js readLocal() JSON.parses, which
   throws on every one of those values and silently returns the fallback — so
   this file reads and writes that key by hand, raw, and stays bidirectionally
   compatible with the old page.

   POLLING
   Monitor every 2s, logs every 5s, countdown every 1s. All three stop dead on
   visibilitychange (via store.state.visible, which app.js maintains) and
   resume with an immediate refresh. There is no MJPEG on this screen, so
   there is no stream to detach.

   NEVER: location.reload(), innerHTML with model data, inline handlers, or a
   wholesale re-render of the log list.
   ========================================================================= */

import { h, svg, clear, on, delegate, keyedList } from '../core/dom.js';
import { icon } from '../core/icons.js';
import { store } from '../core/store.js';
import { router } from '../core/router.js';
import { api } from '../core/api.js';
import { toast } from '../core/toast.js';
import { plural, joinMeta } from '../core/format.js';

/* --- Tuning --------------------------------------------------------------- */

var MONITOR_MS = 2000;
var LOG_MS = 5000;
var SPARK_N = 60;          /* samples kept per gauge */
var ROW_CAP = 400;         /* hard ceiling on log rows held in the DOM */
var WARN_AT = 80;
var CRIT_AT = 92;
var TZ_KEY = 'logTimezone';

/* --- Filter option tables (verbatim from the old Monitor page) ------------ */

var LEVEL_OPTS = [
  ['all', 'All levels'],
  ['warning', 'Warnings & errors'],
  ['error', 'Errors only']
];

var TYPE_OPTS = [
  ['all', 'All types'],
  ['no-http', 'Hide HTTP traffic'],
  ['realtime', 'Realtime detections'],
  ['detection', 'All detections'],
  ['ptz', 'PTZ decisions'],
  ['tracking', 'Object tracking'],
  ['events', 'Events only'],
  ['clips', 'Clips only'],
  ['errors', 'Errors / warnings']
];

var RANGE_OPTS = [
  ['15', 'Last 15 min'],
  ['30', 'Last 30 min'],
  ['60', 'Last 1 hour'],
  ['120', 'Last 2 hours'],
  ['360', 'Last 6 hours'],
  ['720', 'Last 12 hours'],
  ['1440', 'Last 24 hours'],
  ['2880', 'Last 48 hours'],
  ['custom', 'Custom range…']
];

var LIMIT_OPTS = [
  ['100', '100 entries'],
  ['200', '200 entries'],
  ['500', '500 entries'],
  ['1000', '1000 entries'],
  ['2000', '2000 entries']
];

var TZ_OPTS = [
  ['local', 'Local browser time'],
  ['America/Chicago', 'Central (Chicago)'],
  ['America/New_York', 'Eastern (New York)'],
  ['America/Denver', 'Mountain (Denver)'],
  ['America/Los_Angeles', 'Pacific (Los Angeles)'],
  ['UTC', 'UTC'],
  ['Europe/London', 'London'],
  ['Europe/Paris', 'Paris'],
  ['Asia/Tokyo', 'Tokyo']
];

/* --- Small pure helpers --------------------------------------------------- */

function num(v) { var n = Number(v); return isFinite(n) ? n : null; }

/** Percentages here arrive as 0-100 already; format.pct() would rescale a
    sub-1% CPU reading to 90%, so this screen formats its own. */
function fmtPct(v) {
  var n = num(v);
  return n === null ? '--' : (Math.round(n * 10) / 10) + '';
}

function fmtGb(v) {
  var n = num(v);
  return n === null ? '--' : (n >= 100 ? Math.round(n) : Math.round(n * 10) / 10) + '';
}

function levelKey(level) {
  var l = String(level || '').toLowerCase();
  if (l.indexOf('err') === 0 || l === 'critical' || l === 'fatal') return 'error';
  if (l.indexOf('warn') === 0) return 'warn';
  if (l.indexOf('debug') === 0) return 'debug';
  return 'info';
}

/** 'mammalia_rodentia_sciuridae' -> 'Sciuridae'; 'white-tailed deer' -> as-is. */
function prettySpecies(name) {
  var s = String(name || '').trim();
  if (!s) return 'Unknown';
  if (s.indexOf('_') >= 0) {
    var parts = s.split('_');
    s = parts[parts.length - 1];
  }
  s = s.replace(/[_]+/g, ' ');
  return s.charAt(0).toUpperCase() + s.slice(1);
}

/** The 'logTimezone' key, read and written RAW so the old page still agrees. */
function readTz() {
  try {
    var raw = window.localStorage.getItem(TZ_KEY);
    if (raw === null || raw === '') return 'local';
    if (raw.charAt(0) === '"') { try { return JSON.parse(raw); } catch (e) { return raw; } }
    return raw;
  } catch (e) { return 'local'; }
}

function writeTz(value) {
  try { window.localStorage.setItem(TZ_KEY, String(value)); } catch (e) { /* private mode */ }
}

var tzCache = {};
function tzFormatter(tz) {
  if (tzCache[tz]) return tzCache[tz];
  var opts = { hour: '2-digit', minute: '2-digit', second: '2-digit', hour12: false };
  if (tz && tz !== 'local') opts.timeZone = tz;
  var f = null;
  try { f = new Intl.DateTimeFormat(undefined, opts); } catch (e) { f = null; }
  tzCache[tz] = f;
  return f;
}

/** Falls back to the server's pre-rendered `time` string if Intl refuses. */
function logClock(entry, tz) {
  var epoch = num(entry.timestamp);
  if (epoch === null) return String(entry.time || '--:--:--');
  var f = tzFormatter(tz);
  if (!f) return String(entry.time || '--:--:--');
  try { return f.format(new Date(epoch * 1000)); } catch (e) { return String(entry.time || '--:--:--'); }
}

function copyText(text) {
  if (navigator.clipboard && navigator.clipboard.writeText) {
    return navigator.clipboard.writeText(text);
  }
  return new Promise(function (resolve, reject) {
    try {
      var ta = document.createElement('textarea');
      ta.value = text;
      ta.setAttribute('readonly', '');
      ta.style.position = 'fixed';
      ta.style.opacity = '0';
      document.body.appendChild(ta);
      ta.select();
      var ok = document.execCommand('copy');
      document.body.removeChild(ta);
      ok ? resolve() : reject(new Error('The browser refused the copy.'));
    } catch (err) { reject(err); }
  });
}

/* --- Control factories ---------------------------------------------------- */

function labelledSelect(id, labelText, options, value) {
  var sel = h('select.select__el', { id: id });
  for (var i = 0; i < options.length; i++) {
    sel.appendChild(h('option', { value: options[i][0], text: options[i][1] }));
  }
  sel.value = value;
  var field = h('div.field',
    h('label.field__label', { 'for': id, text: labelText }),
    h('div.select', sel, h('span.select__chevron', { 'aria-hidden': 'true' },
      icon('chevron-down', { size: 'sm' }))));
  return { field: field, select: sel };
}

function labelledInput(id, labelText, type) {
  var input = h('input.input', { id: id, type: type });
  var field = h('div.field',
    h('label.field__label', { 'for': id, text: labelText }),
    input);
  return { field: field, input: input };
}

function button(labelText, iconName, variant) {
  return h('button.btn', { type: 'button', 'class': 'btn--' + (variant || 'secondary') },
    iconName ? icon(iconName, { size: 'sm', 'class': 'btn__icon' }) : null,
    h('span.btn__label', { text: labelText }));
}

/* --- The gauge (a .readout, a tall .meter and a 60-sample sparkline) ------ */

function makeGauge(label, unit) {
  /* The value is a real text node kept by reference, so a poll rewrites the
     number without touching the unit span beside it. */
  var valueText = document.createTextNode('--');
  var unitEl = h('span.readout__unit', { text: unit || '' });
  var valueEl = h('span.readout__value', valueText, unitEl);

  var readout = h('div.readout',
    h('span.readout__label', { text: label }),
    valueEl);

  var stateEl = h('span.gauge__state', { text: '' });
  var meterEl = h('div.gauge__meter.meter.meter--tall', {
    style: { '--n': '0', '--of': '100' }, 'aria-hidden': 'true'
  });
  var srEl = h('span.visually-hidden', { text: label + ' unavailable' });
  var poly = svg('polyline', { points: '' });
  var sparkEl = svg('svg.gauge__spark', {
    viewBox: '0 0 100 34', preserveAspectRatio: 'none',
    'aria-hidden': 'true', focusable: 'false'
  }, poly);
  var subEl = h('p.t-micro.t-3', { text: '' });

  var el = h('div.gauge.gauge--unavailable', { role: 'group', 'aria-label': label },
    h('div.gauge__head', readout, stateEl), meterEl, srEl, sparkEl, subEl);

  var samples = [];
  var lastState = '';

  return {
    el: el,
    /** state word is appended so the reading is never colour-only. */
    set: function (percent, subtitle, opts) {
      var o = opts || {};
      var p = num(percent);
      var available = p !== null && o.available !== false;

      if (available) {
        samples.push(Math.max(0, Math.min(100, p)));
        if (samples.length > SPARK_N) samples.shift();
      } else {
        samples.length = 0;
      }

      var word = 'nominal';
      var mod = 'nominal';
      var meterMod = '';
      if (!available) { word = 'unavailable'; mod = 'unavailable'; }
      else if (p >= CRIT_AT) { word = 'critical'; mod = 'critical'; meterMod = 'meter--danger'; }
      else if (p >= WARN_AT) { word = 'high'; mod = 'warn'; meterMod = 'meter--stale'; }

      el.className = 'gauge gauge--' + mod;
      meterEl.className = 'gauge__meter meter meter--tall' + (meterMod ? ' ' + meterMod : '');
      meterEl.style.setProperty('--n', String(available ? Math.max(0, Math.min(100, p)) : 0));
      meterEl.style.setProperty('--of', '100');

      valueText.nodeValue = available ? fmtPct(p) : '--';
      unitEl.textContent = available ? (unit || '') : '';

      stateEl.textContent = word;
      subEl.textContent = subtitle || '';
      srEl.textContent = available
        ? label + ' ' + fmtPct(p) + (unit || '') + ' · ' + word + (subtitle ? ' · ' + subtitle : '')
        : label + ' unavailable';

      /* Sparkline: 60 samples across a 100x34 box, stroke non-scaling. */
      if (samples.length > 1) {
        var pts = [];
        var span = samples.length > 1 ? samples.length - 1 : 1;
        for (var i = 0; i < samples.length; i++) {
          var x = (i / span) * 100;
          var y = 33 - (samples[i] / 100) * 32;
          pts.push((Math.round(x * 100) / 100) + ',' + (Math.round(y * 100) / 100));
        }
        poly.setAttribute('points', pts.join(' '));
      } else {
        poly.setAttribute('points', '');
      }

      var crossed = (mod === 'critical' && lastState !== 'critical');
      lastState = mod;
      return crossed ? word : null;
    }
  };
}

/* ==========================================================================
   THE VIEW
   Only one instance is ever mounted, so the live instance's teardown and
   query-sync hooks live at module scope rather than on `this` — that way a
   detached unmount() call still releases everything.
   ======================================================================== */

var active = null;

export const view = {

  mount: function (root, ctx) {

    /* --- per-mount state ------------------------------------------------- */

    var disposers = [];
    var timers = { monitor: 0, logs: 0, tick: 0 };
    var aborts = { monitor: null, logs: null };
    var dead = false;

    var monitorFailing = false;
    var logsFailing = false;
    var criticalAlerted = {};
    var cameraOptionSig = '';

    var logRows = [];          /* oldest-first, each carrying _key */
    var logMeta = null;
    var pinned = true;         /* auto-scroll pinned to the bottom */
    var unseen = 0;
    var renderedKeys = Object.create(null);
    var countdown = Math.round(LOG_MS / 1000);
    var autoRefresh = true;
    var syncingQuery = false;

    var q = ctx && ctx.query ? ctx.query : {};
    var filters = {
      camera: typeof q.camera === 'string' ? q.camera : '',
      level: LEVEL_OPTS.some(function (o) { return o[0] === q.level; }) ? q.level : 'all',
      type: TYPE_OPTS.some(function (o) { return o[0] === q.type; }) ? q.type : 'all',
      range: RANGE_OPTS.some(function (o) { return o[0] === q.minutes; }) ? q.minutes : '30',
      start: '',
      end: '',
      limit: LIMIT_OPTS.some(function (o) { return o[0] === q.limit; }) ? q.limit : '200',
      tz: readTz(),
      text: ''
    };

    /* --------------------------------------------------------------------
       STRUCTURE
       Every node is built once here; the poll only patches text, classes and
       custom properties. Nothing below is ever rebuilt wholesale.
       ------------------------------------------------------------------ */

    root.appendChild(h('h1.visually-hidden', { tabIndex: -1, text: 'Monitor' }));

    var page = h('div.stack.stack--loose');
    root.appendChild(page);

    /* --- 1 · resource gauges --------------------------------------------- */

    var gCpu = makeGauge('CPU', '%');
    var gMem = makeGauge('Memory', '%');
    var gDisk = makeGauge('Disk', '%');
    var gGpu = makeGauge('GPU', '%');

    var systemError = h('p.empty__cause.t-danger', { hidden: true, role: 'status' });

    page.appendChild(h('section.stack.stack--tight', { 'aria-label': 'System resources' },
      h('h2.overline.overline--strong', { text: 'System' }),
      systemError,
      h('div.gaugerow.stack', gCpu.el, gMem.el, gDisk.el, gGpu.el)));

    /* --- 2 · detector panel ---------------------------------------------- */

    function readout(label, initial) {
      var value = h('span.readout__value', { text: initial || '--' });
      return {
        el: h('div.readout', h('span.readout__label', { text: label }), value),
        value: value
      };
    }

    var rBackend = readout('Backend');
    var rRegion = readout('Region');
    var rCameras = readout('Cameras');
    var rUpdated = readout('Sampled');

    page.appendChild(h('section.stack.stack--tight', { 'aria-label': 'Detector' },
      h('h2.overline.overline--strong', { text: 'Detector' }),
      h('div.gauge',
        h('div.readout-strip', rBackend.el, rRegion.el, rCameras.el, rUpdated.el))));

    /* --- 3 · live reanalyze progress ------------------------------------- */

    var reproList = h('ul.stack.stack--tight', {
      'aria-live': 'polite',
      style: { listStyle: 'none', margin: '0', padding: '0' }
    });
    var reproSection = h('section.stack.stack--tight', {
      'aria-label': 'Active reanalysis', hidden: true
    },
      h('h2.overline.overline--strong', { text: 'Reanalyzing' }),
      reproList);
    page.appendChild(reproSection);

    /* --- 4 · per-camera pipeline cards ----------------------------------- */

    var camList = h('ul.stack', {
      style: { listStyle: 'none', margin: '0', padding: '0' }
    });
    var camEmpty = h('p.t-sm.t-3', { text: 'Waiting for the first sample…' });

    page.appendChild(h('section.stack.stack--tight', { 'aria-label': 'Camera pipelines' },
      h('h2.overline.overline--strong', { text: 'Pipelines' }),
      camEmpty,
      camList));

    /* --- 5 · recent clips ------------------------------------------------ */

    var clipList = h('ul.stack.stack--tight', {
      style: { listStyle: 'none', margin: '0', padding: '0' }
    });
    var clipEmpty = h('p.t-sm.t-3', { text: 'No clips recorded yet.' });

    page.appendChild(h('section.stack.stack--tight', { 'aria-label': 'Recent clips' },
      h('h2.overline.overline--strong', { text: 'Recent clips' }),
      clipEmpty,
      clipList));

    /* --- 6 · the log viewer ---------------------------------------------- */

    var selCamera = labelledSelect('mon-log-camera', 'Camera',
      [['', 'All cameras']], '');
    var selLevel = labelledSelect('mon-log-level', 'Level', LEVEL_OPTS, filters.level);
    var selType = labelledSelect('mon-log-type', 'Type', TYPE_OPTS, filters.type);
    var selRange = labelledSelect('mon-log-range', 'Time range', RANGE_OPTS, filters.range);
    var selLimit = labelledSelect('mon-log-limit', 'Entries', LIMIT_OPTS, filters.limit);
    var selTz = labelledSelect('mon-log-tz', 'Timezone', TZ_OPTS,
      TZ_OPTS.some(function (o) { return o[0] === filters.tz; }) ? filters.tz : 'local');

    var inStart = labelledInput('mon-log-start', 'From', 'datetime-local');
    var inEnd = labelledInput('mon-log-end', 'To', 'datetime-local');
    var customWrap = h('div.row.row--wrap', {
      hidden: filters.range !== 'custom',
      style: { padding: '0 12px 12px' }
    },
      inStart.field, inEnd.field,
      h('span.t-micro.t-3', { text: 'Server time' }));

    var textInput = h('input.search__input', {
      id: 'mon-log-text', type: 'search', placeholder: 'Filter messages',
      autocomplete: 'off', 'aria-label': 'Filter log messages'
    });
    var textClear = h('button.icon-btn.icon-btn--dense.search__clear', {
      type: 'button', 'aria-label': 'Clear the message filter', hidden: true
    }, icon('x', { size: 'sm' }));
    var searchBox = h('div.search.search--block',
      h('span.search__icon', { 'aria-hidden': 'true' }, icon('search', { size: 'sm' })),
      textInput, textClear);

    var btnRefresh = button('Refresh', 'refresh', 'secondary');
    var btnCopy = button('Copy', 'layers', 'secondary');
    var btnAuto = h('button.btn.btn--ghost', {
      type: 'button', 'aria-pressed': 'true',
      'aria-label': 'Pause automatic log refresh'
    }, h('span.btn__label', { text: 'Auto' }));

    /* The countdown rides INSIDE the head, in normal flow, so it can never
       cover a control the way the old fixed badge did. */
    var countdownEl = h('span.t-micro.mono.t-3', { 'aria-hidden': 'true', text: '' });

    var logCount = h('span.t-micro.t-3', { role: 'status', 'aria-live': 'polite', text: '' });

    var logHead = h('div.logview__head',
      selCamera.field, selLevel.field, selType.field,
      selRange.field, selLimit.field, selTz.field,
      h('span.spacer'),
      btnRefresh, btnCopy, btnAuto, countdownEl);

    /* Focusable so End / Home are a real keyboard path. It is deliberately
       NOT role="log": an implicit polite live region would announce every
       arriving line. The count line below is the single live region. */
    var logBody = h('div.logview__body', {
      tabIndex: 0, role: 'group', 'aria-label': 'Log entries',
      style: { maxHeight: '60vh' }
    });
    var rowHost = h('div');
    var overflowNote = h('div.logview__spacer.t-micro.t-3', {
      hidden: true, style: { padding: '8px 12px' }
    });
    var logStatus = h('div.table__empty', { hidden: true });
    var jumpCount = h('span.logview__jump-count', { text: '' });
    var jumpBtn = h('button.logview__jump', { type: 'button', hidden: true },
      icon('arrow-down', { size: 'sm' }),
      h('span', { text: 'Jump to latest' }),
      jumpCount);

    logBody.appendChild(overflowNote);
    logBody.appendChild(rowHost);
    logBody.appendChild(logStatus);
    logBody.appendChild(jumpBtn);

    var logView = h('div.logview',
      logHead,
      customWrap,
      h('div.row.row--wrap', { style: { padding: '0 12px 12px' } }, searchBox, logCount),
      logBody);

    page.appendChild(h('section.stack.stack--tight', { 'aria-label': 'System log' },
      h('h2.overline.overline--strong', { text: 'Log' }),
      logView));

    /* --------------------------------------------------------------------
       CHROME
       ------------------------------------------------------------------ */

    var chromeRefresh = h('button.icon-btn', {
      type: 'button', 'aria-label': 'Refresh now'
    }, icon('refresh', { size: 'sm' }));
    disposers.push(on(chromeRefresh, 'click', function () { refreshAll(true); }));

    var lastSubtitle = null;
    store.setChrome({
      title: 'Monitor', subtitle: 'Sampling…', actions: [chromeRefresh],
      toolbar: null, rail: null, norail: false, selbar: null, mods: []
    });

    function setSubtitle(text) {
      if (text === lastSubtitle) return;
      lastSubtitle = text;
      store.setChrome({ subtitle: text });
    }

    /* --------------------------------------------------------------------
       MONITOR POLL
       ------------------------------------------------------------------ */

    function renderSystem(data) {
      var sys = data.system || {};
      var gpu = data.gpu || {};

      var crossed = [];
      var c;

      c = gCpu.set(sys.cpu_percent, '', {});
      if (c) crossed.push('CPU');

      c = gMem.set(sys.memory_percent,
        joinMeta(fmtGb(sys.memory_used_gb) + ' of ' + fmtGb(sys.memory_total_gb) + ' GB'), {});
      if (c) crossed.push('Memory');

      c = gDisk.set(sys.disk_percent,
        joinMeta(fmtGb(sys.disk_used_gb) + ' of ' + fmtGb(sys.disk_total_gb) + ' GB'), {});
      if (c) crossed.push('Disk');

      gGpu.set(gpu.available ? gpu.utilization : null,
        gpu.available ? joinMeta(gpu.name, gpu.temperature ? gpu.temperature + ' °C' : '') : 'No GPU reported',
        { available: !!gpu.available });

      /* One persistent toast per crossing, not one every poll. */
      for (var i = 0; i < crossed.length; i++) {
        var name = crossed[i];
        if (criticalAlerted[name]) continue;
        criticalAlerted[name] = true;
        toast(name + ' is critical', {
          kind: 'error',
          detail: name + ' has crossed ' + CRIT_AT + '%. Recording may stall.'
        });
      }
      /* Reset the latch once the reading recovers, so a second excursion
         raises a second toast. */
      if (num(sys.cpu_percent) !== null && sys.cpu_percent < CRIT_AT) criticalAlerted.CPU = false;
      if (num(sys.memory_percent) !== null && sys.memory_percent < CRIT_AT) criticalAlerted.Memory = false;
      if (num(sys.disk_percent) !== null && sys.disk_percent < CRIT_AT) criticalAlerted.Disk = false;

      setSubtitle(joinMeta(
        'cpu ' + fmtPct(sys.cpu_percent) + '%',
        'mem ' + fmtPct(sys.memory_percent) + '%',
        'disk ' + fmtPct(sys.disk_percent) + '%',
        plural((data.cameras || []).length, 'camera', 'cameras')
      ));
    }

    function renderDetector(data) {
      var det = data.detector || {};
      rBackend.value.textContent = det.backend ? String(det.backend) : 'unknown';
      rRegion.value.textContent = det.country ? String(det.country) : 'unset';
      rCameras.value.textContent = String((data.cameras || []).length);
      var stamp = String(data.timestamp || '');
      rUpdated.value.textContent = stamp ? stamp.slice(11, 19) : '--';
    }

    function camState(cam) {
      var s = String(cam.status || '').toLowerCase();
      if (s === 'connected' || s === 'live' || s === 'running') return ['live', 'connected'];
      if (s === 'reconnecting' || s === 'connecting') return ['reconnecting', 'reconnecting'];
      if (s === 'stale') return ['stale', 'stale'];
      if (s === 'disconnected' || s === 'offline' || s === 'error') return ['offline', s || 'offline'];
      return ['unknown', s || 'unknown'];
    }

    function renderCameras(data) {
      var cams = data.cameras || [];
      camEmpty.hidden = cams.length > 0;

      keyedList(camList, cams, {
        key: function (cam) { return String(cam.id); },

        create: function () {
          var dot = h('span.status__dot', { 'aria-hidden': 'true' });
          var label = h('span.status__label', { text: '' });
          var status = h('span.status', dot, label);
          var name = h('h3.t-h4.truncate', { style: { margin: '0' }, text: '' });
          var sub = h('p.t-micro.t-3', { style: { margin: '0' }, text: '' });
          /* 20 segments at the meter's own pitch — the intrinsic width fits a
             narrow phone, and the ::after math here is pitch-based, not the
             percentage form .gauge__meter overrides it with. */
          var meter = h('div.meter.meter--tall', {
            style: { '--n': '0', '--of': '20' }, 'aria-hidden': 'true'
          });
          var meterSr = h('span.visually-hidden', { text: '' });

          var rBuffer = readout('Buffer');
          var rEvent = readout('Event');
          var rTracks = readout('Tracks');
          var rSpecies = readout('Species');
          var rConf = readout('Confidence');

          var viewLog = h('button.btn.btn--sm.btn--ghost', {
            type: 'button', 'data-camlog': ''
          }, h('span.btn__label', { text: 'View log' }));

          var li = h('li.gauge',
            h('div.row.row--between',
              h('div.row__grow', name, sub),
              status),
            meter, meterSr,
            h('div.readout-strip', rBuffer.el, rEvent.el, rTracks.el, rSpecies.el, rConf.el),
            h('div.btn-row', viewLog));

          li._parts = {
            status: status, dot: dot, label: label,
            name: name, sub: sub, meter: meter, meterSr: meterSr,
            buffer: rBuffer.value, event: rEvent.value, tracks: rTracks.value,
            species: rSpecies.value, conf: rConf.value, viewLog: viewLog
          };
          return li;
        },

        update: function (li, cam) {
          var p = li._parts;
          var st = camState(cam);

          p.status.className = 'status status--' + st[0];
          p.label.textContent = st[1];

          p.name.textContent = cam.name ? String(cam.name) : String(cam.id);
          p.sub.textContent = joinMeta(cam.id, cam.location);

          var seconds = num(cam.buffer_seconds) || 0;
          var maxSeconds = num(cam.buffer_max_seconds) || 30;
          var frames = num(cam.buffer_frames);
          var maxFrames = num(cam.buffer_max_frames);
          var fill = maxSeconds > 0 ? Math.max(0, Math.min(1, seconds / maxSeconds)) : 0;
          p.meter.style.setProperty('--n', String(Math.round(fill * 20)));
          p.meter.style.setProperty('--of', '20');
          p.meter.className = 'meter meter--tall' +
            (fill < 0.1 ? ' meter--stale' : ' meter--live');
          p.meterSr.textContent = 'Buffer ' + Math.round(fill * 100) + '% full';

          p.buffer.textContent = (Math.round(seconds * 10) / 10) + ' / ' + maxSeconds + ' s';
          if (frames !== null && maxFrames !== null) {
            p.buffer.textContent += '  (' + frames + '/' + maxFrames + 'f)';
          }

          var active = !!cam.event_active;
          p.event.textContent = active
            ? 'recording ' + (Math.round((num(cam.event_duration) || 0) * 10) / 10) + 's'
            : 'idle';

          p.tracks.textContent = String(num(cam.tracks_active) === null ? '--' : cam.tracks_active) +
            (cam.tracking_enabled ? '' : ' (off)');

          var sp = cam.event_species;
          if (Array.isArray(sp) && sp.length) {
            p.species.textContent = sp.map(prettySpecies).join(', ');
          } else {
            p.species.textContent = active ? 'pending' : '--';
          }

          var conf = num(cam.event_confidence);
          p.conf.textContent = conf === null || !active ? '--' : Math.round(conf * 100) + '%';

          p.viewLog.setAttribute('data-camlog', String(cam.id));
          p.viewLog.setAttribute('aria-label',
            'Filter the log to ' + (cam.name || cam.id));
        }
      });
    }

    function renderReprocessing(data) {
      var jobs = data.reprocessing_jobs || [];
      reproSection.hidden = jobs.length === 0;

      keyedList(reproList, jobs, {
        key: function (job, i) { return String(job.clip_name || i) + '|' + String(job.camera || ''); },

        create: function () {
          var name = h('span.t-sm', { text: '' });
          var meta = h('span.t-micro.t-3', { text: '' });
          var li = h('li.gauge',
            h('div.row',
              h('span.spinner', { 'aria-hidden': 'true' }),
              h('div.row__grow.stack.stack--tight', name, meta),
              h('span.status.status--reconnecting',
                h('span.status__dot', { 'aria-hidden': 'true' }),
                h('span.status__label', { text: 'analyzing' }))));
          li._parts = { name: name, meta: meta };
          return li;
        },

        update: function (li, job) {
          li._parts.name.textContent = String(job.clip_name || 'clip');
          var started = job.started ? ' · started ' + String(job.started).slice(11, 19) : '';
          li._parts.meta.textContent = String(job.camera || '') + started;
        }
      });
    }

    function renderRecentClips(data) {
      var clips = data.recent_clips || [];
      clipEmpty.hidden = clips.length > 0;

      keyedList(clipList, clips, {
        key: function (clip, i) { return String(clip.path || i); },

        create: function () {
          var name = h('span.camrow__name', { text: '' });
          var meta = h('span.camrow__meta', { text: '' });
          var link = h('a.camrow', { href: '#' },
            h('span.camrow__dot.camrow__dot--rec', { 'aria-hidden': 'true' }),
            name, meta);
          var li = h('li', link);
          li._parts = { link: link, name: name, meta: meta };
          return li;
        },

        update: function (li, clip) {
          var p = li._parts;
          var species = prettySpecies(clip.species);
          p.name.textContent = species;
          p.meta.textContent = joinMeta(clip.camera, clip.time);
          p.link.setAttribute('href',
            router.href('/clips/' + api.encodePath(String(clip.path || ''))));
          p.link.setAttribute('aria-label',
            species + ' on ' + (clip.camera || 'camera') + ' at ' + (clip.time || ''));
        }
      });
    }

    function loadMonitor(manual) {
      if (dead) return Promise.resolve();
      if (aborts.monitor) aborts.monitor.abort();
      var ctrl = typeof AbortController === 'function' ? new AbortController() : null;
      aborts.monitor = ctrl;

      return api.monitor({ signal: ctrl ? ctrl.signal : undefined, timeout: 8000 })
        .then(function (data) {
          if (dead) return;
          /* Only clear the slot if it is still OUR controller — a newer poll
             may already have claimed it, and unmount must still abort that. */
          if (aborts.monitor === ctrl) aborts.monitor = null;
          systemError.hidden = true;
          if (monitorFailing) {
            monitorFailing = false;
            toast('Monitor reconnected', { kind: 'success' });
          }
          renderSystem(data);
          renderDetector(data);
          renderCameras(data);
          renderReprocessing(data);
          renderRecentClips(data);
          syncCameraOptions(data.cameras || []);
        })
        .catch(function (err) {
          if (dead || api.isAbort(err)) return;
          if (aborts.monitor === ctrl) aborts.monitor = null;
          var msg = api.describe(err);
          systemError.hidden = false;
          systemError.textContent = 'System sample failed — ' + msg;
          /* Failure is always visible, but the toast fires on the transition
             only; a two-second poll must not produce a toast every two
             seconds. A manual press always gets an answer. */
          if (!monitorFailing || manual) {
            monitorFailing = true;
            toast('Could not read system health', { kind: 'error', detail: msg });
          }
        });
    }

    /* The camera <select> is rebuilt only when the camera SET changes, so a
       poll never clobbers an open dropdown or the current choice. */
    function syncCameraOptions(cams) {
      var ids = cams.map(function (c) { return String(c.id); });
      var sig = ids.join(',');
      if (sig === cameraOptionSig) return;
      cameraOptionSig = sig;

      var current = selCamera.select.value;
      clear(selCamera.select);
      selCamera.select.appendChild(h('option', { value: '', text: 'All cameras' }));
      for (var i = 0; i < cams.length; i++) {
        selCamera.select.appendChild(h('option', {
          value: String(cams[i].id),
          text: cams[i].name ? String(cams[i].name) : String(cams[i].id)
        }));
      }
      var want = filters.camera || current;
      selCamera.select.value = ids.indexOf(want) >= 0 ? want : '';
      if (selCamera.select.value !== filters.camera) {
        /* A deep link named a camera this server does not have. Say so by
           falling back to All cameras, and refetch so the rows match the
           control rather than a filter the user can no longer see. */
        filters.camera = selCamera.select.value;
        pushQuery();
        loadLogs(false);
      }
    }

    /* --------------------------------------------------------------------
       LOG POLL
       ------------------------------------------------------------------ */

    function logQuery() {
      var query = {
        level: filters.level,
        type: filters.type,
        limit: filters.limit
      };
      if (filters.camera) query.camera = filters.camera;
      if (filters.range === 'custom' && filters.start && filters.end) {
        query.start = filters.start;
        query.end = filters.end;
      } else if (filters.range === 'custom') {
        query.minutes = '30';   /* the old page's fallback, kept */
      } else {
        query.minutes = filters.range;
      }
      return query;
    }

    function keyOf(entry, tally) {
      var base = String(entry.timestamp) + '|' + levelKey(entry.level) + '|' +
        String(entry.camera || '') + '|' + String(entry.message || '');
      var n = tally[base] || 0;
      tally[base] = n + 1;
      return n ? base + '#' + n : base;
    }

    function loadLogs(manual) {
      if (dead) return Promise.resolve();
      if (aborts.logs) aborts.logs.abort();
      var ctrl = typeof AbortController === 'function' ? new AbortController() : null;
      aborts.logs = ctrl;

      return api.logs(logQuery(), { signal: ctrl ? ctrl.signal : undefined, timeout: 15000 })
        .then(function (data) {
          if (dead) return;
          if (aborts.logs === ctrl) aborts.logs = null;
          logsFailing = false;
          logMeta = data;

          /* The server ships newest-first; the viewer reads oldest-first so
             the newest line is the one at the bottom the eye rests on. */
          var incoming = (data.logs || []).slice().reverse();
          var tally = Object.create(null);
          for (var i = 0; i < incoming.length; i++) {
            incoming[i]._key = keyOf(incoming[i], tally);
          }
          logRows = incoming;
          renderLogs();
        })
        .catch(function (err) {
          if (dead || api.isAbort(err)) return;
          if (aborts.logs === ctrl) aborts.logs = null;
          var msg = api.describe(err);
          logStatus.hidden = false;
          logStatus.textContent = 'Log fetch failed — ' + msg;
          if (!logsFailing || manual) {
            logsFailing = true;
            toast('Could not read the log', { kind: 'error', detail: msg });
          }
        });
    }

    function visibleRows() {
      var needle = filters.text.trim().toLowerCase();
      if (!needle) return logRows;
      return logRows.filter(function (e) {
        return String(e.message || '').toLowerCase().indexOf(needle) >= 0;
      });
    }

    function renderLogs() {
      var rows = visibleRows();

      /* DOM ceiling. The entry limit reaches 2000; the DOM never holds more
         than ROW_CAP of them, and the count of what was dropped is stated
         rather than silently hidden. */
      var dropped = 0;
      if (rows.length > ROW_CAP) {
        dropped = rows.length - ROW_CAP;
        rows = rows.slice(dropped);
      }
      overflowNote.hidden = dropped === 0;
      if (dropped) {
        overflowNote.textContent = dropped + ' earlier ' +
          (dropped === 1 ? 'entry is' : 'entries are') +
          ' held back — narrow the range or lower the entry limit to see them.';
      }

      /* Counting the pill by net child change would undercount: the range is
         a sliding window, so old rows drop off the front as new ones arrive
         and the net stays near zero. Count keys that were not on screen. */
      var arrived = 0;
      if (!pinned) {
        for (var n = 0; n < rows.length; n++) {
          if (!renderedKeys[rows[n]._key]) arrived++;
        }
      }

      keyedList(rowHost, rows, {
        key: function (e) { return e._key; },

        create: function () {
          var ts = h('span.logrow__ts', { text: '' });
          var lvl = h('span.logrow__lvl', { text: '' });
          var msg = h('span.logrow__msg', { text: '' });
          var copy = h('button.logrow__copy', {
            type: 'button', 'data-logcopy': ''
          }, icon('layers', { size: 'sm' }));
          var row = h('div.logrow', ts, lvl, msg, copy);
          row._parts = { ts: ts, lvl: lvl, msg: msg, copy: copy };
          return row;
        },

        update: function (row, entry) {
          var p = row._parts;
          var lk = levelKey(entry.level);
          row.className = 'logrow logrow--' + lk;
          p.ts.textContent = logClock(entry, filters.tz);
          p.lvl.textContent = lk;
          p.msg.textContent = String(entry.message || '');
          p.copy.setAttribute('aria-label',
            'Copy the log line from ' + p.ts.textContent);
          row._text = p.ts.textContent + '  ' + lk.toUpperCase() + '  ' + p.msg.textContent;
        }
      });

      renderedKeys = Object.create(null);
      for (var m = 0; m < rows.length; m++) renderedKeys[rows[m]._key] = true;

      /* Count line — the ONE aria-live region on this screen. Writing the
         same string again would re-announce it every five seconds, so it is
         only assigned when it actually changed. */
      var total = logRows.length;
      var shown = rows.length;
      var parts = [plural(shown, 'entry', 'entries') + ' shown'];
      if (shown !== total) parts.push('of ' + total + ' fetched');
      if (logMeta) {
        if (logMeta.skipped) parts.push(logMeta.skipped + ' filtered out');
        if (logMeta.source) parts.push(String(logMeta.source));
        if (logMeta.error) parts.push(String(logMeta.error));
      }
      var countText = parts.join(' · ');
      if (countText !== logCount.textContent) logCount.textContent = countText;

      logStatus.hidden = shown > 0;
      if (!shown) {
        var emptyText = filters.text
          ? 'No log entry matches that text in the current range.'
          : 'No log entries in this range. Widen the time range or clear a filter.';
        if (emptyText !== logStatus.textContent) logStatus.textContent = emptyText;
      }

      if (pinned) {
        unseen = 0;
        jumpBtn.hidden = true;
        logBody.scrollTop = logBody.scrollHeight;
      } else {
        unseen += arrived;
        if (unseen > 0) {
          jumpBtn.hidden = false;
          jumpCount.textContent = unseen + ' new';
        }
      }
    }

    function jumpToLatest() {
      pinned = true;
      unseen = 0;
      jumpBtn.hidden = true;
      logBody.scrollTop = logBody.scrollHeight;
    }

    /* --------------------------------------------------------------------
       LISTENERS
       ------------------------------------------------------------------ */

    function onFilterChange(reload) {
      renderLogs();
      if (reload !== false) loadLogs(true);
    }

    disposers.push(on(selCamera.select, 'change', function () {
      filters.camera = selCamera.select.value;
      pushQuery();
      onFilterChange();
    }));

    disposers.push(on(selLevel.select, 'change', function () {
      filters.level = selLevel.select.value;
      pushQuery();
      onFilterChange();
    }));

    disposers.push(on(selType.select, 'change', function () {
      filters.type = selType.select.value;
      pushQuery();
      onFilterChange();
    }));

    disposers.push(on(selRange.select, 'change', function () {
      filters.range = selRange.select.value;
      customWrap.hidden = filters.range !== 'custom';
      pushQuery();
      if (filters.range === 'custom' && !(filters.start && filters.end)) {
        /* Do not fire a half-specified range at the server; wait for both. */
        inStart.input.focus();
        renderLogs();
        return;
      }
      onFilterChange();
    }));

    disposers.push(on(inStart.input, 'change', function () {
      filters.start = inStart.input.value;
      if (filters.start && filters.end) onFilterChange();
    }));

    disposers.push(on(inEnd.input, 'change', function () {
      filters.end = inEnd.input.value;
      if (filters.start && filters.end) onFilterChange();
    }));

    disposers.push(on(selLimit.select, 'change', function () {
      filters.limit = selLimit.select.value;
      pushQuery();
      onFilterChange();
    }));

    disposers.push(on(selTz.select, 'change', function () {
      filters.tz = selTz.select.value;
      writeTz(filters.tz);       /* raw, so the old page still reads it */
      renderLogs();              /* re-stamps every row; no refetch needed */
    }));

    var textTimer = 0;
    disposers.push(on(textInput, 'input', function () {
      filters.text = textInput.value;
      textClear.hidden = !filters.text;
      searchBox.classList.toggle('search--filled', !!filters.text);
      window.clearTimeout(textTimer);
      textTimer = window.setTimeout(function () { if (!dead) renderLogs(); }, 120);
    }));

    disposers.push(on(textClear, 'click', function () {
      textInput.value = '';
      filters.text = '';
      textClear.hidden = true;
      searchBox.classList.remove('search--filled');
      renderLogs();
      textInput.focus();
    }));

    disposers.push(on(btnRefresh, 'click', function () { refreshAll(true); }));

    disposers.push(on(btnAuto, 'click', function () {
      autoRefresh = !autoRefresh;
      btnAuto.setAttribute('aria-pressed', autoRefresh ? 'true' : 'false');
      btnAuto.setAttribute('aria-label', autoRefresh
        ? 'Pause automatic log refresh' : 'Resume automatic log refresh');
      btnAuto.className = 'btn ' + (autoRefresh ? 'btn--ghost' : 'btn--secondary');
      if (autoRefresh) startTimers(); else stopTimers();
      paintCountdown();
    }));

    disposers.push(on(btnCopy, 'click', function () {
      var rows = visibleRows();
      if (!rows.length) {
        toast('Nothing to copy', { kind: 'info', detail: 'The current filter matches no entries.' });
        return;
      }
      var text = rows.map(function (e) {
        return logClock(e, filters.tz) + '  ' + levelKey(e.level).toUpperCase() +
          '  ' + String(e.message || '');
      }).join('\n');
      copyText(text).then(function () {
        toast('Copied ' + plural(rows.length, 'entry', 'entries'), { kind: 'success' });
      }, function (err) {
        toast('Copy failed', { kind: 'error', detail: api.describe(err) });
      });
    }));

    disposers.push(delegate(rowHost, 'click', '[data-logcopy]', function (ev, node) {
      ev.preventDefault();
      var row = node.parentElement;
      var text = (row && row._text) || '';
      copyText(text).then(function () {
        toast('Log line copied', { kind: 'success' });
      }, function (err) {
        toast('Copy failed', { kind: 'error', detail: api.describe(err) });
      });
    }));

    disposers.push(delegate(camList, 'click', '[data-camlog]', function (ev, node) {
      ev.preventDefault();
      var id = node.getAttribute('data-camlog') || '';
      filters.camera = id;
      selCamera.select.value = id;
      pushQuery();
      onFilterChange();
      logView.scrollIntoView({ block: 'start' });
      selCamera.select.focus();
    }));

    disposers.push(on(jumpBtn, 'click', jumpToLatest));

    disposers.push(on(logBody, 'scroll', function () {
      var distance = logBody.scrollHeight - logBody.scrollTop - logBody.clientHeight;
      var nowPinned = distance < 24;
      if (nowPinned && !pinned) jumpToLatest();
      else pinned = nowPinned;
    }, { passive: true }));

    /* End jumps to the newest line; the viewer is focusable so this is a real
       keyboard path, not a mouse-only affordance. */
    disposers.push(on(logBody, 'keydown', function (ev) {
      if (ev.key === 'End') { ev.preventDefault(); jumpToLatest(); }
      else if (ev.key === 'Home') { ev.preventDefault(); pinned = false; logBody.scrollTop = 0; }
    }));

    /* --------------------------------------------------------------------
       URL — ?level=error&camera=cam2 round-trips
       ------------------------------------------------------------------ */

    function pushQuery() {
      syncingQuery = true;
      router.setQuery({
        camera: filters.camera || null,
        level: filters.level === 'all' ? null : filters.level,
        type: filters.type === 'all' ? null : filters.type,
        minutes: filters.range === '30' ? null : filters.range,
        limit: filters.limit === '200' ? null : filters.limit
      }, { replace: true });
      syncingQuery = false;
    }

    /* --------------------------------------------------------------------
       TIMERS + VISIBILITY
       ------------------------------------------------------------------ */

    function paintCountdown() {
      if (!autoRefresh) { countdownEl.textContent = 'paused'; return; }
      if (!store.state.visible) { countdownEl.textContent = 'idle'; return; }
      countdownEl.textContent = 'refresh in ' + countdown + 's';
    }

    function refreshAll(manual) {
      countdown = Math.round(LOG_MS / 1000);
      paintCountdown();
      loadMonitor(manual);
      loadLogs(manual);
    }

    function startTimers() {
      stopTimers();
      countdown = Math.round(LOG_MS / 1000);
      if (!autoRefresh || !store.state.visible) { paintCountdown(); return; }
      timers.monitor = window.setInterval(function () { loadMonitor(false); }, MONITOR_MS);
      timers.logs = window.setInterval(function () {
        countdown = Math.round(LOG_MS / 1000);
        loadLogs(false);
      }, LOG_MS);
      timers.tick = window.setInterval(function () {
        countdown = countdown > 1 ? countdown - 1 : Math.round(LOG_MS / 1000);
        paintCountdown();
      }, 1000);
      paintCountdown();
    }

    function stopTimers() {
      if (timers.monitor) { window.clearInterval(timers.monitor); timers.monitor = 0; }
      if (timers.logs) { window.clearInterval(timers.logs); timers.logs = 0; }
      if (timers.tick) { window.clearInterval(timers.tick); timers.tick = 0; }
    }

    /* app.js owns the visibilitychange listener and publishes the result. */
    disposers.push(store.select(['visible'], function (state) {
      if (dead) return;
      if (state.visible) { startTimers(); refreshAll(false); }
      else {
        stopTimers();
        if (aborts.monitor) aborts.monitor.abort();
        if (aborts.logs) aborts.logs.abort();
        paintCountdown();
      }
    }));

    /* --------------------------------------------------------------------
       GO
       ------------------------------------------------------------------ */

    /* Seed the camera dropdown from whatever the shell already loaded, so the
       control is usable before the first monitor sample lands. */
    if (store.state.cameras && store.state.cameras.length) {
      syncCameraOptions(store.state.cameras);
    }
    customWrap.hidden = filters.range !== 'custom';
    refreshAll(false);
    startTimers();

    /* Expose the teardown for unmount(). */
    var teardown = function () {
      dead = true;
      stopTimers();
      window.clearTimeout(textTimer);
      if (aborts.monitor) aborts.monitor.abort();
      if (aborts.logs) aborts.logs.abort();
      aborts.monitor = null;
      aborts.logs = null;
      for (var i = 0; i < disposers.length; i++) {
        try { disposers[i](); } catch (e) { /* a disposer must never block teardown */ }
      }
      disposers.length = 0;
      logRows = [];
      logMeta = null;
    };

    /* Same-route query changes (a deep link pasted while already here). */
    var applyQuery = function (nextQuery) {
      if (syncingQuery) return;
      var qq = nextQuery || {};
      var changed = false;

      var wantCamera = typeof qq.camera === 'string' ? qq.camera : '';
      if (wantCamera !== filters.camera) {
        filters.camera = wantCamera;
        selCamera.select.value = wantCamera;
        if (selCamera.select.value !== wantCamera) selCamera.select.value = '';
        changed = true;
      }

      var wantLevel = LEVEL_OPTS.some(function (o) { return o[0] === qq.level; }) ? qq.level : 'all';
      if (wantLevel !== filters.level) {
        filters.level = wantLevel; selLevel.select.value = wantLevel; changed = true;
      }

      var wantType = TYPE_OPTS.some(function (o) { return o[0] === qq.type; }) ? qq.type : 'all';
      if (wantType !== filters.type) {
        filters.type = wantType; selType.select.value = wantType; changed = true;
      }

      var wantRange = RANGE_OPTS.some(function (o) { return o[0] === qq.minutes; }) ? qq.minutes : '30';
      if (wantRange !== filters.range) {
        filters.range = wantRange; selRange.select.value = wantRange;
        customWrap.hidden = wantRange !== 'custom';
        changed = true;
      }

      var wantLimit = LIMIT_OPTS.some(function (o) { return o[0] === qq.limit; }) ? qq.limit : '200';
      if (wantLimit !== filters.limit) {
        filters.limit = wantLimit; selLimit.select.value = wantLimit; changed = true;
      }

      if (changed) loadLogs(false);
    };

    active = { teardown: teardown, applyQuery: applyQuery };
  },

  /** Same route, new query string — re-read ctx rather than remount. */
  update: function (ctx) {
    if (active) active.applyQuery(ctx && ctx.query);
  },

  unmount: function () {
    if (!active) return;
    var a = active;
    active = null;
    a.teardown();
  }
};

export default view;
