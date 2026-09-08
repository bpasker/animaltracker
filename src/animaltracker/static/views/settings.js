/* ============================================================================
   views/settings.js — global and per-camera configuration.

   Route: /app/settings   (optional query: ?section=global|<cameraId>)

   WHAT MAKES THIS SCREEN DIFFERENT FROM EVERY OTHER ONE
   Saving here rewrites the operator's config/cameras.yml. That file is
   gitignored, has no backup, and the server rewrites it wholesale with
   yaml.dump() — comments do not survive. So:

     · Edits are STAGED, never live. Nothing leaves the browser until Save.
     · The payload is always COMPLETE: every camera, every managed key, built
       from one serialiser that throws rather than emit a partial object. A
       half-object here is a truncated config on disk.
     · Every value is validated against the same ranges the old page enforced,
       plus the cross-field checks it never had (min_days <= max_days).
     · The save is optimistic — the baseline advances immediately so the UI is
       honest about intent — and rolls back field-for-field on failure, with a
       persistent danger toast naming the OS/HTTP error and a Retry. A failed
       write NEVER silently drops the operator's edits.
     · Fields whose value only takes effect after a restart say so, and the
       success toast repeats the list.

   The field inventory is a straight port of the old handle_settings_page:
   nothing was dropped, and the read-only rows (detector backends, PTZ) are
   still shown, still read-only, because they are how an operator confirms
   what the running process actually believes.

   Everything here is built with the core: h() for DOM (no innerHTML with
   model data — species names come off disk and out of ONVIF), keyedList for
   the species lists, delegate for their clicks, toast/dialog for reporting.
   ========================================================================= */

import { h, clear, on, delegate, keyedList } from '../core/dom.js';
import { icon } from '../core/icons.js';
import { api } from '../core/api.js';
import { store } from '../core/store.js';
import { toast } from '../core/toast.js';
import { dialog } from '../core/overlay.js';
import { router } from '../core/router.js';
import { speciesClass } from '../core/format.js';

/* --------------------------------------------------------------------------
   STATIC CATALOGUES  (ported verbatim from the old page)
   ------------------------------------------------------------------------ */

var SPECIES_CATALOG = [
  ['Common Wildlife', ['deer', 'coyote', 'fox', 'raccoon', 'opossum', 'skunk', 'rabbit',
    'squirrel', 'chipmunk', 'groundhog', 'armadillo', 'porcupine', 'beaver']],
  ['Large Mammals', ['bear', 'moose', 'elk', 'mountain lion', 'cougar', 'bobcat', 'lynx',
    'wolf', 'wild boar', 'javelina', 'bison', 'antelope']],
  ['Birds', ['bird', 'turkey', 'hawk', 'owl', 'eagle', 'vulture', 'crow',
    'heron', 'duck', 'goose', 'pheasant', 'quail', 'dove', 'woodpecker']],
  ['Farm Animals', ['horse', 'cow', 'sheep', 'goat', 'pig', 'chicken', 'donkey', 'llama']],
  ['Pets & Domestic', ['dog', 'cat', 'person']],
  ['Other', ['snake', 'turtle', 'frog', 'lizard', 'alligator', 'fish',
    'bat', 'mouse', 'rat', 'mole', 'weasel', 'otter', 'mink', 'badger']]
];

var HWACCEL_OPTIONS = [
  ['', 'None (CPU)'],
  ['nvdec', 'NVDEC (NVIDIA)'],
  ['cuda', 'CUDA (NVIDIA)'],
  ['vaapi', 'VAAPI (Intel/AMD)'],
  ['videotoolbox', 'VideoToolbox (macOS)']
];

var TRANSPORT_OPTIONS = [
  ['tcp', 'TCP (reliable)'],
  ['udp', 'UDP (lower latency)']
];

var PRIORITY_OPTIONS = [
  ['-2', 'Lowest'],
  ['-1', 'Low'],
  ['0', 'Normal'],
  ['1', 'High']
];

var SOUND_OPTIONS = [
  ['', 'Default'], ['pushover', 'Pushover'], ['bike', 'Bike'], ['bugle', 'Bugle'],
  ['cashregister', 'Cash Register'], ['classical', 'Classical'], ['cosmic', 'Cosmic'],
  ['falling', 'Falling'], ['gamelan', 'Gamelan'], ['incoming', 'Incoming'],
  ['intermission', 'Intermission'], ['magic', 'Magic'], ['mechanical', 'Mechanical'],
  ['pianobar', 'Piano Bar'], ['siren', 'Siren'], ['spacealarm', 'Space Alarm'],
  ['tugboat', 'Tugboat'], ['none', 'None (silent)']
];

/* Numeric ranges, exactly the min/max/step the old inputs carried. `int`
   forces a whole number; `label` is what a validation toast names. */
var GLOBAL_NUM_RULES = {
  'clip.pre_seconds': { min: 1, max: 60, step: 1, int: true, label: 'Pre-event buffer' },
  'clip.post_seconds': { min: 1, max: 60, step: 1, int: true, label: 'Post-event buffer' },
  'clip.max_concurrent_postprocess': { min: 1, max: 8, step: 1, int: true, label: 'Max concurrent post-processing' },
  'clip.max_event_seconds': { min: 30, max: 600, step: 10, int: true, label: 'Max event duration' },
  'clip.sample_rate': { min: 1, max: 30, step: 1, int: true, label: 'Sample rate' },
  'clip.track_merge_gap': { min: 10, max: 500, step: 10, int: true, label: 'Track merge gap' },
  'clip.post_analysis_confidence': { min: 0, max: 1, step: 0.05, label: 'Post-analysis species confidence' },
  'clip.post_analysis_generic_confidence': { min: 0, max: 1, step: 0.05, label: 'Post-analysis generic confidence' },
  'clip.spatial_merge_iou': { min: 0.1, max: 0.9, step: 0.05, label: 'Spatial overlap (IoU)' },
  'detector.generic_confidence': { min: 0, max: 1, step: 0.05, label: 'Default generic confidence' },
  'retention.min_days': { min: 1, max: 365, step: 1, int: true, label: 'Minimum retention' },
  'retention.max_days': { min: 1, max: 365, step: 1, int: true, label: 'Maximum retention' },
  'retention.max_utilization_pct': { min: 50, max: 95, step: 5, int: true, label: 'Max disk usage' }
};

var CAMERA_NUM_RULES = {
  'thresholds.confidence': { min: 0, max: 1, step: 0.05, label: 'Species confidence' },
  'thresholds.generic_confidence': { min: 0, max: 1, step: 0.05, label: 'Generic category confidence' },
  'thresholds.min_frames': { min: 1, max: 30, step: 1, int: true, label: 'Minimum frames' },
  'thresholds.min_duration': { min: 0, max: 30, step: 0.5, label: 'Minimum duration' },
  'rtsp.frame_skip': { min: 0, max: 30, step: 1, int: true, label: 'Frame skip' },
  'rtsp.latency_ms': { min: 0, max: 5000, step: 100, int: true, label: 'Latency' },
  'notification.priority': { min: -2, max: 1, step: 1, int: true, label: 'Notification priority' }
};

/* Paths whose new value the running process only picks up after a restart.
   Keyed by the tail of the path so one table serves every camera. */
var RESTART_TAILS = {
  'clip.max_concurrent_postprocess': 'the post-processing worker pool is sized at startup',
  'rtsp.frame_skip': 'the stream reader is built at startup',
  'rtsp.hwaccel': 'the decoder is chosen at startup',
  'rtsp.latency_ms': 'the stream buffer is sized at startup',
  'rtsp.transport': 'the RTSP session is negotiated at startup'
};

/* --------------------------------------------------------------------------
   TINY UTILITIES
   ------------------------------------------------------------------------ */

var uidN = 0;
function uid(prefix) { uidN += 1; return prefix + '-' + uidN; }

function clone(v) { return JSON.parse(JSON.stringify(v)); }

function isArray(v) { return Object.prototype.toString.call(v) === '[object Array]'; }

function getAt(obj, path) {
  var cur = obj;
  for (var i = 0; i < path.length; i++) {
    if (cur === null || cur === undefined) return undefined;
    cur = cur[path[i]];
  }
  return cur;
}

function setAt(obj, path, value) {
  var cur = obj;
  for (var i = 0; i < path.length - 1; i++) {
    if (cur[path[i]] === null || typeof cur[path[i]] !== 'object') cur[path[i]] = {};
    cur = cur[path[i]];
  }
  cur[path[path.length - 1]] = value;
}

function pathKey(path) { return path.join(''); }

/* Species lists are compared as case-insensitive sets: toggling a chip off and
   back on must not read as a change just because the order moved. */
function normList(v) {
  if (!isArray(v)) return '';
  var out = [];
  for (var i = 0; i < v.length; i++) out.push(String(v[i]).toLowerCase());
  out.sort();
  return out.join('');
}

function eqValue(a, b) {
  if (isArray(a) || isArray(b)) return normList(a) === normList(b);
  if (a === null || a === undefined) return (b === null || b === undefined);
  if (b === null || b === undefined) return false;
  if (typeof a === 'number' || typeof b === 'number') return Number(a) === Number(b);
  return String(a) === String(b);
}

function titleCase(s) {
  var str = String(s || '');
  return str.charAt(0).toUpperCase() + str.slice(1);
}

function plural(n, one, many) { return n === 1 ? one : (many || one + 's'); }

function toNumber(v) {
  if (v === null || v === undefined || v === '') return NaN;
  var n = Number(v);
  return isFinite(n) ? n : NaN;
}

function roundTo(value, step) {
  if (!step) return value;
  var r = Math.round(value / step) * step;
  /* Kill float noise: 0.30000000000000004 must print as 0.3. */
  return Math.round(r * 1e6) / 1e6;
}

/* --------------------------------------------------------------------------
   THE PATH INVENTORY
   Dirty state and validation are computed from this list, NOT from whatever
   happens to be rendered — so an edit in the Global section still counts
   while you are looking at cam2.
   ------------------------------------------------------------------------ */

var GLOBAL_PATHS = [
  ['global', 'detector', 'generic_confidence'],
  ['global', 'clip', 'pre_seconds'],
  ['global', 'clip', 'post_seconds'],
  ['global', 'clip', 'max_concurrent_postprocess'],
  ['global', 'clip', 'max_event_seconds'],
  ['global', 'clip', 'post_analysis'],
  ['global', 'clip', 'post_analysis_confidence'],
  ['global', 'clip', 'post_analysis_generic_confidence'],
  ['global', 'clip', 'delete_if_no_animal'],
  ['global', 'clip', 'sample_rate'],
  ['global', 'clip', 'tracking_enabled'],
  ['global', 'clip', 'track_merge_gap'],
  ['global', 'clip', 'spatial_merge_enabled'],
  ['global', 'clip', 'spatial_merge_iou'],
  ['global', 'clip', 'hierarchical_merge_enabled'],
  ['global', 'clip', 'single_animal_mode'],
  ['global', 'clip', 'thumbnail_cropped'],
  ['global', 'retention', 'min_days'],
  ['global', 'retention', 'max_days'],
  ['global', 'retention', 'max_utilization_pct'],
  ['global', 'exclusion_list']
];

var CAMERA_TAILS = [
  ['detect_enabled'],
  ['thresholds', 'confidence'],
  ['thresholds', 'generic_confidence'],
  ['thresholds', 'min_frames'],
  ['thresholds', 'min_duration'],
  ['rtsp', 'frame_skip'],
  ['rtsp', 'hwaccel'],
  ['rtsp', 'latency_ms'],
  ['rtsp', 'transport'],
  ['notification', 'priority'],
  ['notification', 'sound'],
  ['include_species'],
  ['exclude_species']
];

function cameraPaths(id) {
  var out = [];
  for (var i = 0; i < CAMERA_TAILS.length; i++) {
    out.push(['cameras', id].concat(CAMERA_TAILS[i]));
  }
  return out;
}

function allPaths(model) {
  var out = GLOBAL_PATHS.slice();
  var ids = cameraIds(model);
  for (var i = 0; i < ids.length; i++) out = out.concat(cameraPaths(ids[i]));
  return out;
}

function cameraIds(model) {
  var ids = [];
  if (!model || !model.cameras) return ids;
  for (var k in model.cameras) {
    if (Object.prototype.hasOwnProperty.call(model.cameras, k)) ids.push(k);
  }
  ids.sort();
  return ids;
}

/* --------------------------------------------------------------------------
   NORMALISATION
   The server hands back nulls where the UI wants a string, and omits keys on
   older configs. Normalising once on load means every control below can
   assume its value exists and has the right type.
   ------------------------------------------------------------------------ */

function num(v, fallback) {
  var n = Number(v);
  return isFinite(n) ? n : fallback;
}

function bool(v, fallback) {
  if (v === true || v === false) return v;
  if (v === null || v === undefined) return fallback;
  return !!v;
}

function strList(v) {
  if (!isArray(v)) return [];
  var out = [];
  for (var i = 0; i < v.length; i++) {
    var s = String(v[i] === null || v[i] === undefined ? '' : v[i]);
    if (s) out.push(s);
  }
  return out;
}

function normalize(raw) {
  var src = raw && typeof raw === 'object' ? raw : {};
  var g = src.global && typeof src.global === 'object' ? src.global : {};
  var det = g.detector && typeof g.detector === 'object' ? g.detector : {};
  var clip = g.clip && typeof g.clip === 'object' ? g.clip : {};
  var ret = g.retention && typeof g.retention === 'object' ? g.retention : {};

  var model = {
    global: {
      detector: {
        backend: det.backend === null || det.backend === undefined ? '' : String(det.backend),
        realtime_backend: det.realtime_backend ? String(det.realtime_backend) : String(det.backend || ''),
        postprocess_backend: det.postprocess_backend ? String(det.postprocess_backend) : 'speciesnet',
        speciesnet_version: det.speciesnet_version ? String(det.speciesnet_version) : '',
        country: det.country ? String(det.country) : '',
        admin1_region: det.admin1_region ? String(det.admin1_region) : '',
        generic_confidence: num(det.generic_confidence, 0.9)
      },
      clip: {
        pre_seconds: num(clip.pre_seconds, 5),
        post_seconds: num(clip.post_seconds, 5),
        max_event_seconds: num(clip.max_event_seconds, 300),
        max_concurrent_postprocess: num(clip.max_concurrent_postprocess, 1),
        post_analysis: bool(clip.post_analysis, true),
        post_analysis_confidence: num(clip.post_analysis_confidence, 0.3),
        post_analysis_generic_confidence: num(clip.post_analysis_generic_confidence, 0.5),
        delete_if_no_animal: bool(clip.delete_if_no_animal, true),
        sample_rate: num(clip.sample_rate, 3),
        tracking_enabled: bool(clip.tracking_enabled, true),
        track_merge_gap: num(clip.track_merge_gap, 120),
        spatial_merge_enabled: bool(clip.spatial_merge_enabled, true),
        spatial_merge_iou: num(clip.spatial_merge_iou, 0.3),
        hierarchical_merge_enabled: bool(clip.hierarchical_merge_enabled, true),
        single_animal_mode: bool(clip.single_animal_mode, false),
        thumbnail_cropped: bool(clip.thumbnail_cropped, true)
      },
      retention: {
        min_days: num(ret.min_days, 7),
        max_days: num(ret.max_days, 30),
        max_utilization_pct: num(ret.max_utilization_pct, 80)
      },
      exclusion_list: strList(g.exclusion_list)
    },
    cameras: {}
  };

  var cams = src.cameras && typeof src.cameras === 'object' ? src.cameras : {};
  for (var id in cams) {
    if (!Object.prototype.hasOwnProperty.call(cams, id)) continue;
    var c = cams[id] || {};
    var th = c.thresholds || {};
    var rt = c.rtsp || {};
    var nt = c.notification || {};
    var pt = c.ptz_tracking || {};
    model.cameras[id] = {
      id: String(c.id || id),
      name: c.name ? String(c.name) : String(id),
      location: c.location ? String(c.location) : '',
      detect_enabled: bool(c.detect_enabled, true),
      thresholds: {
        confidence: num(th.confidence, 0.5),
        generic_confidence: num(th.generic_confidence, 0.9),
        min_frames: num(th.min_frames, 3),
        min_duration: num(th.min_duration, 2)
      },
      rtsp: {
        frame_skip: num(rt.frame_skip, 0),
        hwaccel: rt.hwaccel ? String(rt.hwaccel) : '',
        transport: rt.transport ? String(rt.transport) : 'tcp',
        latency_ms: num(rt.latency_ms, 200)
      },
      notification: {
        priority: num(nt.priority, 0),
        sound: nt.sound ? String(nt.sound) : ''
      },
      /* Read-only mirror of what the process believes. Never sent back. */
      ptz_tracking: {
        enabled: bool(pt.enabled, false),
        target_camera_id: pt.target_camera_id ? String(pt.target_camera_id) : '',
        self_track: bool(pt.self_track, false),
        multi_camera_tracking: bool(pt.multi_camera_tracking, true),
        target_fill_pct: num(pt.target_fill_pct, 0.6),
        patrol_enabled: bool(pt.patrol_enabled, true),
        patrol_return_delay: num(pt.patrol_return_delay, 5)
      },
      include_species: strList(c.include_species),
      exclude_species: strList(c.exclude_species),
      recent_detections: c.recent_detections && typeof c.recent_detections === 'object'
        ? c.recent_detections : {}
    };
  }
  return model;
}

/* --------------------------------------------------------------------------
   THE PAYLOAD
   One serialiser, used for the POST and for the "Reveal in YAML" preview.
   It THROWS on anything it cannot represent — a partial body here is a
   truncated cameras.yml on disk, so failing loudly is the only safe move.
   ------------------------------------------------------------------------ */

function PayloadError(message) {
  var e = new Error(message);
  e.name = 'PayloadError';
  return e;
}

function requireNum(model, path, rule) {
  var v = getAt(model, path);
  var n = Number(v);
  if (!isFinite(n)) throw PayloadError((rule && rule.label ? rule.label : path.join('.')) + ' is not a number.');
  if (rule) {
    if (n < rule.min || n > rule.max) {
      throw PayloadError((rule.label || path.join('.')) + ' is outside its allowed range.');
    }
    if (rule.int && Math.round(n) !== n) return Math.round(n);
  }
  return n;
}

function requireBool(model, path) {
  var v = getAt(model, path);
  if (v !== true && v !== false) throw PayloadError(path.join('.') + ' is not a boolean.');
  return v;
}

function requireList(model, path) {
  var v = getAt(model, path);
  if (!isArray(v)) throw PayloadError(path.join('.') + ' is not a list.');
  return strList(v);
}

function buildGlobalPayload(model) {
  var g = ['global'];
  return {
    detector: {
      generic_confidence: requireNum(model, g.concat(['detector', 'generic_confidence']), GLOBAL_NUM_RULES['detector.generic_confidence'])
    },
    clip: {
      pre_seconds: requireNum(model, g.concat(['clip', 'pre_seconds']), GLOBAL_NUM_RULES['clip.pre_seconds']),
      post_seconds: requireNum(model, g.concat(['clip', 'post_seconds']), GLOBAL_NUM_RULES['clip.post_seconds']),
      max_event_seconds: requireNum(model, g.concat(['clip', 'max_event_seconds']), GLOBAL_NUM_RULES['clip.max_event_seconds']),
      max_concurrent_postprocess: requireNum(model, g.concat(['clip', 'max_concurrent_postprocess']), GLOBAL_NUM_RULES['clip.max_concurrent_postprocess']),
      post_analysis: requireBool(model, g.concat(['clip', 'post_analysis'])),
      post_analysis_confidence: requireNum(model, g.concat(['clip', 'post_analysis_confidence']), GLOBAL_NUM_RULES['clip.post_analysis_confidence']),
      post_analysis_generic_confidence: requireNum(model, g.concat(['clip', 'post_analysis_generic_confidence']), GLOBAL_NUM_RULES['clip.post_analysis_generic_confidence']),
      delete_if_no_animal: requireBool(model, g.concat(['clip', 'delete_if_no_animal'])),
      sample_rate: requireNum(model, g.concat(['clip', 'sample_rate']), GLOBAL_NUM_RULES['clip.sample_rate']),
      tracking_enabled: requireBool(model, g.concat(['clip', 'tracking_enabled'])),
      track_merge_gap: requireNum(model, g.concat(['clip', 'track_merge_gap']), GLOBAL_NUM_RULES['clip.track_merge_gap']),
      spatial_merge_enabled: requireBool(model, g.concat(['clip', 'spatial_merge_enabled'])),
      spatial_merge_iou: requireNum(model, g.concat(['clip', 'spatial_merge_iou']), GLOBAL_NUM_RULES['clip.spatial_merge_iou']),
      hierarchical_merge_enabled: requireBool(model, g.concat(['clip', 'hierarchical_merge_enabled'])),
      single_animal_mode: requireBool(model, g.concat(['clip', 'single_animal_mode'])),
      thumbnail_cropped: requireBool(model, g.concat(['clip', 'thumbnail_cropped']))
    },
    retention: {
      min_days: requireNum(model, g.concat(['retention', 'min_days']), GLOBAL_NUM_RULES['retention.min_days']),
      max_days: requireNum(model, g.concat(['retention', 'max_days']), GLOBAL_NUM_RULES['retention.max_days']),
      max_utilization_pct: requireNum(model, g.concat(['retention', 'max_utilization_pct']), GLOBAL_NUM_RULES['retention.max_utilization_pct'])
    },
    exclusion_list: requireList(model, g.concat(['exclusion_list']))
  };
}

function buildCameraPayload(model, id) {
  var p = ['cameras', id];
  var hw = getAt(model, p.concat(['rtsp', 'hwaccel']));
  var sound = getAt(model, p.concat(['notification', 'sound']));
  var transport = String(getAt(model, p.concat(['rtsp', 'transport'])) || '');
  if (transport !== 'tcp' && transport !== 'udp') {
    throw PayloadError('Camera ' + id + ': transport must be tcp or udp.');
  }
  return {
    detect_enabled: requireBool(model, p.concat(['detect_enabled'])),
    thresholds: {
      confidence: requireNum(model, p.concat(['thresholds', 'confidence']), CAMERA_NUM_RULES['thresholds.confidence']),
      generic_confidence: requireNum(model, p.concat(['thresholds', 'generic_confidence']), CAMERA_NUM_RULES['thresholds.generic_confidence']),
      min_frames: requireNum(model, p.concat(['thresholds', 'min_frames']), CAMERA_NUM_RULES['thresholds.min_frames']),
      min_duration: requireNum(model, p.concat(['thresholds', 'min_duration']), CAMERA_NUM_RULES['thresholds.min_duration'])
    },
    rtsp: {
      frame_skip: requireNum(model, p.concat(['rtsp', 'frame_skip']), CAMERA_NUM_RULES['rtsp.frame_skip']),
      /* '' means "no hardware decoder"; the server stores null and drops the
         key from the YAML, which is what the old page did. */
      hwaccel: hw ? String(hw) : null,
      latency_ms: requireNum(model, p.concat(['rtsp', 'latency_ms']), CAMERA_NUM_RULES['rtsp.latency_ms']),
      transport: transport
    },
    notification: {
      priority: requireNum(model, p.concat(['notification', 'priority']), CAMERA_NUM_RULES['notification.priority']),
      sound: sound ? String(sound) : null
    },
    include_species: requireList(model, p.concat(['include_species'])),
    exclude_species: requireList(model, p.concat(['exclude_species']))
  };
}

function buildPayload(model, expectedIds) {
  var cams = {};
  var ids = cameraIds(model);
  if (!ids.length) throw PayloadError('No cameras are loaded — refusing to write an empty camera set.');
  if (expectedIds && expectedIds.length !== ids.length) {
    throw PayloadError('The camera set changed while you were editing. Reload before saving.');
  }
  for (var i = 0; i < ids.length; i++) cams[ids[i]] = buildCameraPayload(model, ids[i]);
  var payload = { cameras: cams, global: buildGlobalPayload(model) };
  /* Last gate before the wire: the server merges whatever arrives, so an
     absent sub-object silently keeps stale values rather than erroring. */
  if (!payload.global.clip || !payload.global.retention || !payload.global.detector ||
      !isArray(payload.global.exclusion_list)) {
    throw PayloadError('The settings payload came out incomplete — nothing was sent.');
  }
  return payload;
}

/* Cross-field checks the individual controls cannot see. */
function crossValidate(model) {
  var out = [];
  var minD = Number(getAt(model, ['global', 'retention', 'min_days']));
  var maxD = Number(getAt(model, ['global', 'retention', 'max_days']));
  if (isFinite(minD) && isFinite(maxD) && minD > maxD) {
    out.push({
      path: ['global', 'retention', 'min_days'],
      message: 'Minimum retention (' + minD + 'd) cannot exceed maximum retention (' + maxD + 'd).'
    });
  }
  return out;
}

function validateModel(model) {
  var out = [];
  var i, rule, v, n;

  for (i = 0; i < GLOBAL_PATHS.length; i++) {
    var gp = GLOBAL_PATHS[i];
    rule = GLOBAL_NUM_RULES[gp.slice(1).join('.')];
    if (!rule) continue;
    n = toNumber(getAt(model, gp));
    if (!isFinite(n)) { out.push({ path: gp, message: rule.label + ' must be a number.' }); continue; }
    if (n < rule.min || n > rule.max) {
      out.push({ path: gp, message: rule.label + ' must be between ' + rule.min + ' and ' + rule.max + '.' });
    }
  }

  var ids = cameraIds(model);
  for (var c = 0; c < ids.length; c++) {
    for (var t = 0; t < CAMERA_TAILS.length; t++) {
      var tail = CAMERA_TAILS[t];
      rule = CAMERA_NUM_RULES[tail.join('.')];
      if (!rule) continue;
      var cp = ['cameras', ids[c]].concat(tail);
      n = toNumber(getAt(model, cp));
      if (!isFinite(n)) { out.push({ path: cp, message: model.cameras[ids[c]].name + ' — ' + rule.label + ' must be a number.' }); continue; }
      if (n < rule.min || n > rule.max) {
        out.push({
          path: cp,
          message: model.cameras[ids[c]].name + ' — ' + rule.label +
            ' must be between ' + rule.min + ' and ' + rule.max + '.'
        });
      }
    }
  }

  return out.concat(crossValidate(model));
}

/* --------------------------------------------------------------------------
   A VERY SMALL YAML WRITER  (preview only — never parsed back)
   ------------------------------------------------------------------------ */

function yamlScalar(v) {
  if (v === null || v === undefined) return 'null';
  if (v === true) return 'true';
  if (v === false) return 'false';
  if (typeof v === 'number') return String(v);
  var s = String(v);
  if (s === '' || /[:#\-{}\[\],&*?|>'"%@`]/.test(s) || /^\s|\s$/.test(s) || /^[0-9.]+$/.test(s)) {
    return '"' + s.replace(/\\/g, '\\\\').replace(/"/g, '\\"') + '"';
  }
  return s;
}

function toYaml(value, indent) {
  var pad = new Array(indent + 1).join('  ');
  var lines = [];
  var k, i;
  if (isArray(value)) {
    if (!value.length) return pad + '[]\n';
    for (i = 0; i < value.length; i++) lines.push(pad + '- ' + yamlScalar(value[i]));
    return lines.join('\n') + '\n';
  }
  if (value && typeof value === 'object') {
    for (k in value) {
      if (!Object.prototype.hasOwnProperty.call(value, k)) continue;
      var v = value[k];
      if (v && typeof v === 'object') {
        if (isArray(v) && !v.length) { lines.push(pad + k + ': []'); continue; }
        lines.push(pad + k + ':');
        lines.push(toYaml(v, indent + 1).replace(/\n$/, ''));
      } else {
        lines.push(pad + k + ': ' + yamlScalar(v));
      }
    }
    return lines.join('\n') + '\n';
  }
  return pad + yamlScalar(value) + '\n';
}

/* ==========================================================================
   SESSION STATE
   ========================================================================= */

var S = null;

function newSession() {
  return {
    root: null,
    baseline: null,      /* what the server last confirmed */
    draft: null,         /* what the operator is editing */
    section: 'global',
    fields: [],          /* controllers for the CURRENTLY RENDERED section */
    fieldByKey: {},
    offs: [],            /* every listener this view attached */
    abort: null,
    refreshAbort: null,
    saving: false,
    destroyed: false,
    navEl: null,
    panelEl: null,
    selbar: null,
    saveBtn: null,
    resetBtn: null,
    countEl: null,
    countDetailEl: null,
    selbarShown: false,
    navList: null,
    layoutEl: null,
    contentEl: null,
    savedTimers: []
  };
}

function track(off) { if (off) S.offs.push(off); return off; }

/* ==========================================================================
   FIELD CONTROLLERS
   Each returns { path, el, setDirty, setError, setSaved, focus, sync }.
   `sync` re-reads the draft into the control (used after Reset / reload).
   ========================================================================= */

function fieldShell(o) {
  var labelId = uid('lbl');
  var hintId = uid('hint');
  var el = h('div.field');
  var dot = h('span.field__dirty', { hidden: true, 'aria-hidden': 'true' });
  var labelEl = h(o.labelTag || 'label.field__label#' + labelId, { text: o.label });
  labelEl.appendChild(dot);
  if (o.restart) {
    labelEl.appendChild(h('span.badge.sp--unknown', { text: 'Restart' }));
  }
  var statusEl = h('span.field__status', { 'aria-live': 'polite' });
  var hintEl = h('p.field__hint#' + hintId, { text: o.hint || '' });
  var errEl = h('p.field__error', { hidden: true, role: 'alert' });
  return {
    el: el, dot: dot, labelEl: labelEl, labelId: labelId,
    hintEl: hintEl, hintId: hintId, errEl: errEl, statusEl: statusEl
  };
}

function baseController(shell, path, extra) {
  var savedTimer = null;
  var ctl = {
    path: path,
    key: pathKey(path),
    el: shell.el,
    label: extra.label,
    focus: extra.focus,
    sync: extra.sync,
    setDirty: function (on_) {
      shell.dot.hidden = !on_;
      if (on_) shell.dot.setAttribute('title', 'Changed — not yet saved');
    },
    setError: function (msg) {
      if (msg) {
        shell.errEl.hidden = false;
        clear(shell.errEl);
        shell.errEl.appendChild(icon('alert', { size: 'sm' }));
        shell.errEl.appendChild(h('span', { text: msg }));
        shell.hintEl.hidden = true;
        if (extra.setInvalid) extra.setInvalid(true);
      } else {
        shell.errEl.hidden = true;
        clear(shell.errEl);
        shell.hintEl.hidden = false;
        if (extra.setInvalid) extra.setInvalid(false);
      }
    },
    setStatus: function (kind, text) {
      clear(shell.statusEl);
      shell.statusEl.className = 'field__status';
      if (!kind) return;
      if (kind === 'saved') {
        shell.statusEl.classList.add('field__saved');
        shell.statusEl.appendChild(icon('check', { size: 'sm' }));
        shell.statusEl.appendChild(h('span', { text: text || 'Saved' }));
        if (savedTimer) clearTimeout(savedTimer);
        savedTimer = setTimeout(function () {
          if (shell.statusEl.isConnected) { clear(shell.statusEl); shell.statusEl.className = 'field__status'; }
        }, 2400);
        S.savedTimers.push(savedTimer);
      } else if (kind === 'error') {
        shell.statusEl.appendChild(icon('alert', { size: 'sm' }));
        shell.statusEl.appendChild(h('span.t-danger', { text: text || 'Not saved' }));
      }
    }
  };
  return ctl;
}

/* --- read-only mirror of a running value --------------------------------- */

function readonlyField(o) {
  var shell = fieldShell({ label: o.label, hint: o.hint });
  var input = h('input.input', {
    type: 'text', value: o.value === null || o.value === undefined ? '' : String(o.value),
    disabled: true, readonly: true, 'aria-describedby': shell.hintId
  });
  shell.labelEl.setAttribute('for', input.id || (input.id = uid('ro')));
  shell.el.appendChild(shell.labelEl);
  shell.el.appendChild(input);
  shell.el.appendChild(shell.hintEl);
  return shell.el;
}

/* --- number + stepper ---------------------------------------------------- */

function numberField(o) {
  var shell = fieldShell({ label: o.label, hint: o.hint, restart: o.restart });
  var inputId = uid('num');
  var input = h('input.input#' + inputId, {
    type: 'number', inputmode: 'decimal',
    min: String(o.min), max: String(o.max), step: String(o.step),
    'aria-describedby': shell.hintId
  });
  shell.labelEl.setAttribute('for', inputId);

  var dec = h('button.icon-btn.icon-btn--dense', {
    type: 'button', 'aria-label': 'Decrease ' + o.label
  }, icon('minus'));
  var inc = h('button.icon-btn.icon-btn--dense', {
    type: 'button', 'aria-label': 'Increase ' + o.label
  }, icon('plus'));

  var stepper = h('div.stepper', dec, input, inc);
  shell.el.appendChild(shell.labelEl);
  shell.el.appendChild(stepper);
  shell.el.appendChild(shell.hintEl);
  shell.el.appendChild(shell.errEl);
  shell.el.appendChild(shell.statusEl);

  function currentValue() {
    var n = toNumber(getAt(S.draft, o.path));
    return isFinite(n) ? n : o.min;
  }

  function render() {
    var v = currentValue();
    input.value = String(v);
    dec.disabled = v <= o.min;
    inc.disabled = v >= o.max;
  }

  function commit(next, fromTyping) {
    var n = toNumber(next);
    if (!isFinite(n)) {
      ctl.setError(o.label + ' must be a number.');
      return;
    }
    if (n < o.min || n > o.max) {
      ctl.setError(o.label + ' must be between ' + o.min + ' and ' + o.max + '.');
      /* Do NOT write an out-of-range value into the draft: the model stays
         sendable at all times, and the error tells the user why. */
      return;
    }
    ctl.setError(null);
    if (o.int) n = Math.round(n);
    else n = roundTo(n, 0.01);
    setAt(S.draft, o.path, n);
    if (!fromTyping) render();
    else { dec.disabled = n <= o.min; inc.disabled = n >= o.max; }
    onModelChanged();
  }

  function nudge(dir, mult) {
    var step = o.step * (mult || 1);
    var v = currentValue() + dir * step;
    v = Math.min(o.max, Math.max(o.min, roundTo(v, o.int ? 1 : 0.01)));
    commit(v, false);
    input.focus();
  }

  track(on(dec, 'click', function () { nudge(-1, 1); }));
  track(on(inc, 'click', function () { nudge(1, 1); }));
  track(on(input, 'input', function () {
    /* An empty box is mid-edit, not an error: leave the draft alone and say
       nothing until blur puts a value back. */
    if (input.value === '' || input.value === '-') { ctl.setError(null); return; }
    commit(input.value, true);
  }));
  track(on(input, 'blur', function () {
    var n = toNumber(input.value);
    if (!isFinite(n)) { render(); ctl.setError(null); return; }
    commit(Math.min(o.max, Math.max(o.min, n)), false);
  }));
  track(on(input, 'keydown', function (ev) {
    if (ev.key !== 'ArrowUp' && ev.key !== 'ArrowDown') return;
    if (!ev.shiftKey) return;                 /* plain arrows: native step */
    ev.preventDefault();
    nudge(ev.key === 'ArrowUp' ? 1 : -1, 10);
  }));

  var ctl = baseController(shell, o.path, {
    label: o.label,
    focus: function () { input.focus(); },
    sync: function () { render(); ctl.setError(null); },
    setInvalid: function (bad) {
      if (bad) input.setAttribute('aria-invalid', 'true');
      else input.removeAttribute('aria-invalid');
    }
  });
  render();
  return ctl;
}

/* --- slider -------------------------------------------------------------- */

function sliderField(o) {
  /* o.min/max/step are in DISPLAY units. toModel/fromModel bridge to the
     stored value (a 0..1 fraction for every confidence in this config). */
  var shell = fieldShell({ label: o.label, hint: o.hint, restart: o.restart });
  var toModel = o.toModel || function (v) { return v; };
  var fromModel = o.fromModel || function (v) { return v; };
  var unit = o.unit || '';

  var fill = h('div.slider__fill');
  var knob = h('div.slider__knob');
  var rail = h('div.slider__rail', fill, knob);
  var out = h('output.slider__output');
  var el = h('div.slider', {
    role: 'slider', tabIndex: 0,
    'aria-labelledby': shell.labelId,
    'aria-describedby': shell.hintId,
    'aria-valuemin': String(o.min),
    'aria-valuemax': String(o.max)
  }, rail, out);

  shell.labelEl = h('span.field__label#' + shell.labelId, { text: o.label });
  var dot = h('span.field__dirty', { hidden: true, 'aria-hidden': 'true' });
  shell.labelEl.appendChild(dot);
  if (o.restart) shell.labelEl.appendChild(h('span.badge.sp--unknown', { text: 'Restart' }));
  shell.dot = dot;

  shell.el.appendChild(shell.labelEl);
  shell.el.appendChild(el);
  shell.el.appendChild(shell.hintEl);
  shell.el.appendChild(shell.errEl);
  shell.el.appendChild(shell.statusEl);

  function display() {
    var m = toNumber(getAt(S.draft, o.path));
    if (!isFinite(m)) m = toModel(o.min);
    var v = fromModel(m);
    return Math.min(o.max, Math.max(o.min, v));
  }

  function render() {
    var v = display();
    var frac = (v - o.min) / (o.max - o.min || 1);
    el.style.setProperty('--v', String(frac));
    el.setAttribute('aria-valuenow', String(v));
    el.setAttribute('aria-valuetext', v + unit);
    out.textContent = v + unit;
    /* A value already on disk can sit outside the slider's range. Say so
       rather than let the knob quietly lie about what will be saved. */
    if (ctl) {
      var raw = toNumber(getAt(S.draft, o.path));
      var shown = fromModel(isFinite(raw) ? raw : toModel(o.min));
      if (isFinite(shown) && (shown < o.min || shown > o.max)) {
        ctl.setError(o.label + ' is ' + shown + unit + ' on disk, outside the allowed ' +
          o.min + unit + '–' + o.max + unit + '. Move the slider to correct it.');
      } else {
        ctl.setError(null);
      }
    }
  }

  function commit(v) {
    var next = Math.min(o.max, Math.max(o.min, roundTo(v, o.step)));
    var model = roundTo(toModel(next), 0.0001);
    if (eqValue(model, getAt(S.draft, o.path))) { render(); return; }
    setAt(S.draft, o.path, model);
    render();
    onModelChanged();
  }

  function fromPointer(clientX) {
    var r = rail.getBoundingClientRect();
    if (!r.width) return;
    var frac = (clientX - r.left) / r.width;
    frac = Math.min(1, Math.max(0, frac));
    commit(o.min + frac * (o.max - o.min));
  }

  var dragging = false;
  track(on(el, 'pointerdown', function (ev) {
    if (ev.button !== undefined && ev.button !== 0) return;
    dragging = true;
    el.classList.add('slider--dragging');
    if (el.setPointerCapture) { try { el.setPointerCapture(ev.pointerId); } catch (e) {} }
    el.focus();
    fromPointer(ev.clientX);
    ev.preventDefault();
  }));
  track(on(el, 'pointermove', function (ev) {
    if (!dragging) return;
    fromPointer(ev.clientX);
  }));
  function endDrag() {
    if (!dragging) return;
    dragging = false;
    el.classList.remove('slider--dragging');
  }
  track(on(el, 'pointerup', endDrag));
  track(on(el, 'pointercancel', endDrag));
  track(on(el, 'blur', endDrag));

  track(on(el, 'keydown', function (ev) {
    var v = display();
    var k = ev.key;
    var handled = true;
    if (k === 'ArrowRight' || k === 'ArrowUp') v += ev.shiftKey ? Math.min(1, o.step) : o.step;
    else if (k === 'ArrowLeft' || k === 'ArrowDown') v -= ev.shiftKey ? Math.min(1, o.step) : o.step;
    else if (k === 'PageUp') v += o.step * 4;
    else if (k === 'PageDown') v -= o.step * 4;
    else if (k === 'Home') v = o.min;
    else if (k === 'End') v = o.max;
    else handled = false;
    if (!handled) return;
    ev.preventDefault();
    commit(v);
  }));

  var ctl = baseController(shell, o.path, {
    label: o.label,
    focus: function () { el.focus(); },
    sync: function () { render(); }
  });
  ctl.setDirty = function (on_) { dot.hidden = !on_; };
  render();
  return ctl;
}

/* --- switch -------------------------------------------------------------- */

function switchField(o) {
  var titleId = uid('sw');
  var hintId = uid('swh');
  var dot = h('span.field__dirty', { hidden: true, 'aria-hidden': 'true' });
  var titleEl = h('span.switch-row__title#' + titleId, { text: o.label });
  titleEl.appendChild(dot);
  if (o.restart) titleEl.appendChild(h('span.badge.sp--unknown', { text: 'Restart' }));
  var hintEl = h('span.switch-row__hint#' + hintId, { text: o.hint || '' });
  var stateEl = h('span.switch-row__state');
  var row = h('div.switch-row', {
    role: 'switch', tabIndex: o.disabled ? -1 : 0,
    'aria-labelledby': titleId,
    'aria-describedby': hintId,
    'aria-disabled': o.disabled ? 'true' : null
  },
    h('span.switch-row__text', titleEl, hintEl),
    stateEl,
    h('span.switch', h('span.switch__knob')));

  var statusEl = h('span.field__status', { 'aria-live': 'polite' });
  var el = h('div.field', row, statusEl);

  function value() {
    if (o.disabled) return !!o.value;
    return !!getAt(S.draft, o.path);
  }

  function render() {
    var v = value();
    row.setAttribute('aria-checked', v ? 'true' : 'false');
    stateEl.textContent = v ? 'On' : 'Off';
  }

  function toggle() {
    if (o.disabled) return;
    setAt(S.draft, o.path, !value());
    render();
    onModelChanged();
  }

  if (!o.disabled) {
    track(on(row, 'click', toggle));
    track(on(row, 'keydown', function (ev) {
      if (ev.key !== ' ' && ev.key !== 'Enter' && ev.key !== 'Spacebar') return;
      ev.preventDefault();
      toggle();
    }));
  }

  render();

  if (o.disabled) return { el: el, readonly: true };

  var savedTimer = null;
  return {
    path: o.path,
    key: pathKey(o.path),
    el: el,
    label: o.label,
    focus: function () { row.focus(); },
    sync: render,
    setDirty: function (on_) { dot.hidden = !on_; },
    setError: function () {},
    setStatus: function (kind, text) {
      clear(statusEl);
      statusEl.className = 'field__status';
      if (kind === 'saved') {
        statusEl.classList.add('field__saved');
        statusEl.appendChild(icon('check', { size: 'sm' }));
        statusEl.appendChild(h('span', { text: text || 'Saved' }));
        if (savedTimer) clearTimeout(savedTimer);
        savedTimer = setTimeout(function () {
          if (statusEl.isConnected) { clear(statusEl); statusEl.className = 'field__status'; }
        }, 2400);
        S.savedTimers.push(savedTimer);
      } else if (kind === 'error') {
        statusEl.appendChild(icon('alert', { size: 'sm' }));
        statusEl.appendChild(h('span.t-danger', { text: text || 'Not saved' }));
      }
    }
  };
}

/* --- select -------------------------------------------------------------- */

function selectField(o) {
  var shell = fieldShell({ label: o.label, hint: o.hint, restart: o.restart });
  var selId = uid('sel');
  var sel = h('select.select__el#' + selId, { 'aria-describedby': shell.hintId });
  for (var i = 0; i < o.options.length; i++) {
    sel.appendChild(h('option', { value: o.options[i][0] }, o.options[i][1]));
  }
  shell.labelEl.setAttribute('for', selId);
  var wrap = h('div.select', sel, h('span.select__chevron', icon('chevron-down', { size: 'sm' })));

  shell.el.appendChild(shell.labelEl);
  shell.el.appendChild(wrap);
  shell.el.appendChild(shell.hintEl);
  shell.el.appendChild(shell.errEl);
  shell.el.appendChild(shell.statusEl);

  function render() {
    var v = getAt(S.draft, o.path);
    sel.value = v === null || v === undefined ? '' : String(v);
  }

  track(on(sel, 'change', function () {
    var raw = sel.value;
    setAt(S.draft, o.path, o.numeric ? Number(raw) : raw);
    onModelChanged();
  }));

  var ctl = baseController(shell, o.path, {
    label: o.label,
    focus: function () { sel.focus(); },
    sync: render
  });
  render();
  return ctl;
}

/* --- species multi-select ------------------------------------------------ */

function speciesField(o) {
  /* o: { path, label, hint, recent: {name: count} | null } */
  var labelId = uid('spl');
  var el = h('section.field', { 'aria-labelledby': labelId });
  var dot = h('span.field__dirty', { hidden: true, 'aria-hidden': 'true' });
  var head = h('h3.field__label#' + labelId, { text: o.label });
  head.appendChild(dot);
  el.appendChild(head);
  el.appendChild(h('p.field__hint', { text: o.hint || '' }));

  var searchId = uid('spq');
  var searchInput = h('input.search__input#' + searchId, {
    type: 'search', placeholder: 'Search species…',
    'aria-label': 'Filter ' + o.label, autocomplete: 'off',
    autocapitalize: 'off', spellcheck: 'false'
  });
  var searchBox = h('div.search.search--block',
    h('span.search__icon', icon('search', { size: 'sm' })),
    searchInput);

  var clearBtn = h('button.chip.chip--clear', { type: 'button' },
    h('span', { text: 'Clear all' }));

  var countEl = h('p.field__hint', { role: 'status', 'aria-live': 'polite' });
  var groupsHost = h('div.stack.stack--tight');
  var emptyNote = h('p.field__hint', { hidden: true, text: 'No species match that filter.' });

  el.appendChild(h('div.row.row--wrap', h('div.row__grow', searchBox), clearBtn));
  el.appendChild(countEl);
  el.appendChild(groupsHost);
  el.appendChild(emptyNote);

  /* Groups are stable; only their contents are reconciled. */
  var groups = [];
  function addGroup(id, label, items) {
    var listEl = h('div.chip-row.chip-row--wrap', { role: 'group', 'aria-label': label });
    var headEl = h('p.overline', { text: label });
    var wrap = h('div', headEl, listEl);
    groups.push({ id: id, items: items, wrap: wrap, listEl: listEl });
    groupsHost.appendChild(wrap);
  }

  function currentList() {
    var v = getAt(S.draft, o.path);
    return isArray(v) ? v : [];
  }

  /* Anything already selected but absent from the catalogue and from recent
     detections still has to be visible — otherwise the UI would quietly hide
     a filter that is actually in force. */
  var seen = {};
  var recentItems = [];
  var k;
  if (o.recent) {
    var recentKeys = [];
    for (k in o.recent) if (Object.prototype.hasOwnProperty.call(o.recent, k)) recentKeys.push(k);
    recentKeys.sort(function (a, b) { return (o.recent[b] || 0) - (o.recent[a] || 0); });
    for (var r = 0; r < recentKeys.length; r++) {
      recentItems.push({ name: recentKeys[r], count: o.recent[recentKeys[r]] });
      seen[recentKeys[r].toLowerCase()] = 1;
    }
  }
  var catalogSeen = {};
  var cat;
  for (cat = 0; cat < SPECIES_CATALOG.length; cat++) {
    for (var s = 0; s < SPECIES_CATALOG[cat][1].length; s++) {
      catalogSeen[SPECIES_CATALOG[cat][1][s].toLowerCase()] = 1;
    }
  }
  var customItems = [];
  var startList = currentList();
  for (var q = 0; q < startList.length; q++) {
    var low = String(startList[q]).toLowerCase();
    if (seen[low] || catalogSeen[low]) continue;
    seen[low] = 1;
    customItems.push({ name: startList[q], count: null });
  }

  if (recentItems.length) addGroup('recent', 'Recent detections', recentItems);
  if (customItems.length) addGroup('custom', 'In this list', customItems);
  for (cat = 0; cat < SPECIES_CATALOG.length; cat++) {
    var names = SPECIES_CATALOG[cat][1];
    var items = [];
    for (var n2 = 0; n2 < names.length; n2++) items.push({ name: names[n2], count: null });
    addGroup('cat' + cat, SPECIES_CATALOG[cat][0], items);
  }

  var filter = '';

  function selectedSet() {
    var set = {};
    var list = currentList();
    for (var i = 0; i < list.length; i++) set[String(list[i]).toLowerCase()] = 1;
    return set;
  }

  function render() {
    var set = selectedSet();
    var shown = 0;
    for (var g = 0; g < groups.length; g++) {
      var grp = groups[g];
      var visible = [];
      for (var i = 0; i < grp.items.length; i++) {
        var it = grp.items[i];
        if (filter && it.name.toLowerCase().indexOf(filter) < 0) continue;
        visible.push(it);
      }
      shown += visible.length;
      grp.wrap.hidden = visible.length === 0;
      keyedList(grp.listEl, visible, {
        key: function (item) { return item.name.toLowerCase(); },
        create: function (item) {
          var chip = h('button.chip', {
            type: 'button',
            'class': speciesClass(item.name),
            dataset: { species: item.name }
          },
            h('span.chip__dot', { 'aria-hidden': 'true' }),
            h('span.chip__label'),
            item.count === null || item.count === undefined ? null
              : h('span.chip__count', { text: String(item.count) }));
          return chip;
        },
        update: function (node, item) {
          var labelNode = node.querySelector('.chip__label');
          if (labelNode) labelNode.textContent = titleCase(item.name);
          var on_ = !!set[item.name.toLowerCase()];
          node.setAttribute('aria-pressed', on_ ? 'true' : 'false');
          node.setAttribute('aria-label',
            titleCase(item.name) + (on_ ? ' — in list' : ' — not in list'));
        }
      });
    }
    emptyNote.hidden = shown !== 0;
    var count = currentList().length;
    countEl.textContent = count === 0
      ? (o.emptyMeans || 'Nothing selected.')
      : plural(count, 'species', 'species') + ' selected';
    clearBtn.disabled = count === 0;
  }

  function toggle(name) {
    var list = currentList().slice();
    var low = String(name).toLowerCase();
    var idx = -1;
    for (var i = 0; i < list.length; i++) {
      if (String(list[i]).toLowerCase() === low) { idx = i; break; }
    }
    if (idx >= 0) list.splice(idx, 1);
    else list.push(name);
    setAt(S.draft, o.path, list);
    render();
    onModelChanged();
  }

  track(delegate(groupsHost, 'click', '.chip[data-species]', function (ev, node) {
    ev.preventDefault();
    toggle(node.dataset.species);
  }));
  track(on(searchInput, 'input', function () {
    filter = searchInput.value.trim().toLowerCase();
    if (filter) searchBox.classList.add('search--filled');
    else searchBox.classList.remove('search--filled');
    render();
  }));
  track(on(clearBtn, 'click', function () {
    if (!currentList().length) return;
    setAt(S.draft, o.path, []);
    render();
    onModelChanged();
  }));

  render();

  return {
    path: o.path,
    key: pathKey(o.path),
    el: el,
    label: o.label,
    focus: function () { searchInput.focus(); },
    sync: render,
    setDirty: function (on_) { dot.hidden = !on_; },
    setError: function () {},
    setStatus: function () {}
  };
}

/* ==========================================================================
   SECTION RENDERING
   ========================================================================= */

function fieldset(legend, children) {
  var fs = h('fieldset.fieldset', h('legend.fieldset__legend', { text: legend }));
  for (var i = 0; i < children.length; i++) {
    if (!children[i]) continue;
    fs.appendChild(children[i].el || children[i]);
  }
  return fs;
}

function reg(ctl) {
  if (!ctl || ctl.readonly || !ctl.path) return ctl;
  S.fields.push(ctl);
  S.fieldByKey[ctl.key] = ctl;
  return ctl;
}

function pctSlider(o) {
  return sliderField({
    path: o.path, label: o.label, hint: o.hint, restart: o.restart,
    min: o.min === undefined ? 0 : o.min,
    max: o.max === undefined ? 100 : o.max,
    step: o.step === undefined ? 5 : o.step,
    unit: '%',
    toModel: function (v) { return roundTo(v / 100, 0.0001); },
    fromModel: function (m) { return Math.round(m * 100); }
  });
}

function renderGlobalSection(host) {
  var d = S.draft.global.detector;

  host.appendChild(fieldset('Detector', [
    readonlyField({
      label: 'Real-time detector', value: d.realtime_backend || d.backend || '—',
      hint: 'Fast detector for live streaming and PTZ tracking (MegaDetector ~50–150 ms). Edit cameras.yml to change; requires a restart.'
    }),
    readonlyField({
      label: 'Post-processing detector', value: d.postprocess_backend || 'speciesnet',
      hint: 'Accurate detector for clip analysis after recording (SpeciesNet ~200–500 ms). Edit cameras.yml to change; requires a restart.'
    }),
    readonlyField({
      label: 'SpeciesNet version', value: d.speciesnet_version || '—',
      hint: 'Model release the post-processor loads at startup.'
    }),
    readonlyField({
      label: 'Location', value: ((d.country || '') + ' ' + (d.admin1_region || '')).trim() || '—',
      hint: 'Geographic priors for species filtering (e.g. USA MN).'
    }),
    reg(pctSlider({
      path: ['global', 'detector', 'generic_confidence'],
      label: 'Default generic confidence',
      hint: 'Fallback threshold for vague labels (animal, bird, mammal) where a camera sets none.'
    }))
  ]));

  host.appendChild(fieldset('Clips', [
    reg(numberField({
      path: ['global', 'clip', 'pre_seconds'], label: 'Pre-event buffer (seconds)',
      min: 1, max: 60, step: 1, int: true,
      hint: 'Seconds of video to keep before the detection trigger.'
    })),
    reg(numberField({
      path: ['global', 'clip', 'post_seconds'], label: 'Post-event buffer (seconds)',
      min: 1, max: 60, step: 1, int: true,
      hint: 'Seconds of video to record after the detection ends.'
    })),
    reg(numberField({
      path: ['global', 'clip', 'max_concurrent_postprocess'], label: 'Max concurrent post-processing',
      min: 1, max: 8, step: 1, int: true, restart: true,
      hint: 'Concurrent post-processing jobs (lower = less RAM). The worker pool is sized at startup.'
    })),
    reg(numberField({
      path: ['global', 'clip', 'max_event_seconds'], label: 'Max event duration (seconds)',
      min: 30, max: 600, step: 10, int: true,
      hint: 'Force-close events after this duration (prevents a memory leak).'
    })),
    reg(switchField({
      path: ['global', 'clip', 'post_analysis'], label: 'Post-analysis enabled',
      hint: 'Re-analyse clips after recording for better species ID.'
    })),
    reg(pctSlider({
      path: ['global', 'clip', 'post_analysis_confidence'], label: 'Post-analysis species confidence',
      hint: 'Species threshold for post-analysis (lower catches more).'
    })),
    reg(pctSlider({
      path: ['global', 'clip', 'post_analysis_generic_confidence'], label: 'Post-analysis generic confidence',
      hint: 'Generic category threshold for post-analysis (animal, bird, etc.).'
    })),
    reg(switchField({
      path: ['global', 'clip', 'delete_if_no_animal'], label: 'Delete false positives',
      hint: 'Automatically delete clips where post-analysis finds no animal (leaves, shadows).'
    })),
    reg(numberField({
      path: ['global', 'clip', 'sample_rate'], label: 'Sample rate',
      min: 1, max: 30, step: 1, int: true,
      hint: 'Analyse every Nth frame (lower = more thorough, slower).'
    })),
    reg(switchField({
      path: ['global', 'clip', 'tracking_enabled'], label: 'Object tracking',
      hint: 'Track the same animal across frames for a consistent species ID (ByteTrack).'
    })),
    reg(numberField({
      path: ['global', 'clip', 'track_merge_gap'], label: 'Track merge gap',
      min: 10, max: 500, step: 10, int: true,
      hint: 'Maximum frame gap when merging same-species tracks.'
    })),
    reg(switchField({
      path: ['global', 'clip', 'spatial_merge_enabled'], label: 'Spatial merge',
      hint: 'Merge tracks in the same location (ignores species misclassifications).'
    })),
    reg(pctSlider({
      path: ['global', 'clip', 'spatial_merge_iou'], label: 'Spatial overlap (IoU)',
      min: 10, max: 90, step: 5,
      hint: 'Minimum bounding-box overlap to merge (30% recommended).'
    })),
    reg(switchField({
      path: ['global', 'clip', 'hierarchical_merge_enabled'], label: 'Hierarchical merging',
      hint: 'Merge generic "animal" tracks into specific species tracks.'
    })),
    reg(switchField({
      path: ['global', 'clip', 'single_animal_mode'], label: 'Single animal mode',
      hint: 'Force-merge ALL non-overlapping tracks into one. Use only when you are certain there is one animal.'
    })),
    reg(switchField({
      path: ['global', 'clip', 'thumbnail_cropped'], label: 'Cropped thumbnails',
      hint: 'Zoom thumbnails to the detection area (off = full frame with a bounding box). This server build accepts the value but does not persist it to cameras.yml.'
    }))
  ]));

  host.appendChild(fieldset('Storage & retention', [
    reg(numberField({
      path: ['global', 'retention', 'min_days'], label: 'Minimum retention (days)',
      min: 1, max: 365, step: 1, int: true,
      hint: 'Keep clips for at least this many days.'
    })),
    reg(numberField({
      path: ['global', 'retention', 'max_days'], label: 'Maximum retention (days)',
      min: 1, max: 365, step: 1, int: true,
      hint: 'Delete clips older than this, unless space is needed sooner.'
    })),
    reg(sliderField({
      path: ['global', 'retention', 'max_utilization_pct'], label: 'Max disk usage (%)',
      min: 50, max: 95, step: 5, unit: '%',
      hint: 'Start deleting old clips when disk usage exceeds this.'
    }))
  ]));

  host.appendChild(fieldset('Global species exclusions', [
    reg(speciesField({
      path: ['global', 'exclusion_list'],
      label: 'Never notify for these species',
      hint: 'Applies to every camera, on top of each camera’s own exclude list.',
      emptyMeans: 'No global exclusions — every species notifies.',
      recent: null
    }))
  ]));
}

function renderCameraSection(host, id) {
  var cam = S.draft.cameras[id];
  if (!cam) {
    host.appendChild(h('p.field__hint', { text: 'That camera is no longer in the configuration.' }));
    return;
  }
  var p = ['cameras', id];

  host.appendChild(fieldset('Identity', [
    readonlyField({ label: 'Camera ID', value: cam.id, hint: 'The key used in cameras.yml, in clip paths and in the API.' }),
    readonlyField({ label: 'Name', value: cam.name, hint: 'Display name. Edit cameras.yml to change; requires a restart.' }),
    readonlyField({ label: 'Location', value: cam.location || '—', hint: 'Free-text placement note shown on Live and Monitor.' })
  ]));

  host.appendChild(fieldset('Detection', [
    reg(switchField({
      path: p.concat(['detect_enabled']), label: 'Detection enabled',
      hint: 'Enable or disable detection for this camera.'
    })),
    reg(pctSlider({
      path: p.concat(['thresholds', 'confidence']), label: 'Species confidence',
      hint: 'Threshold for specific species (cardinal, deer, and so on).'
    })),
    reg(pctSlider({
      path: p.concat(['thresholds', 'generic_confidence']), label: 'Generic category confidence',
      hint: 'Higher threshold for vague labels (animal, bird, mammal).'
    })),
    reg(numberField({
      path: p.concat(['thresholds', 'min_frames']), label: 'Minimum frames',
      min: 1, max: 30, step: 1, int: true,
      hint: 'Consecutive frames with a detection required before an event opens.'
    })),
    reg(numberField({
      path: p.concat(['thresholds', 'min_duration']), label: 'Minimum duration (seconds)',
      min: 0, max: 30, step: 0.5,
      hint: 'Minimum event duration before a clip is saved and a notification sent.'
    }))
  ]));

  host.appendChild(fieldset('Stream', [
    reg(numberField({
      path: p.concat(['rtsp', 'frame_skip']), label: 'Frame skip',
      min: 0, max: 30, step: 1, int: true, restart: true,
      hint: 'Skip N frames between detections (reduces CPU; 0 analyses every frame).'
    })),
    reg(selectField({
      path: p.concat(['rtsp', 'hwaccel']), label: 'Hardware acceleration',
      options: HWACCEL_OPTIONS, restart: true,
      hint: 'Hardware decoder for the stream (platform-specific).'
    })),
    reg(numberField({
      path: p.concat(['rtsp', 'latency_ms']), label: 'Latency (ms)',
      min: 0, max: 5000, step: 100, int: true, restart: true,
      hint: 'Stream latency buffer in milliseconds.'
    })),
    reg(selectField({
      path: p.concat(['rtsp', 'transport']), label: 'Transport',
      options: TRANSPORT_OPTIONS, restart: true,
      hint: 'RTSP transport protocol.'
    }))
  ]));

  host.appendChild(fieldset('Notifications', [
    reg(selectField({
      path: p.concat(['notification', 'priority']), label: 'Priority',
      options: PRIORITY_OPTIONS, numeric: true,
      hint: 'Pushover notification priority.'
    })),
    reg(selectField({
      path: p.concat(['notification', 'sound']), label: 'Sound',
      options: SOUND_OPTIONS,
      hint: 'Notification sound.'
    }))
  ]));

  var ptz = cam.ptz_tracking;
  var ptzFields = [
    readonlyField({
      label: 'PTZ tracking enabled', value: ptz.enabled ? 'Yes' : 'No',
      hint: 'Automatic pan-tilt-zoom tracking. Edit cameras.yml to change; requires a restart.'
    })
  ];
  if (ptz.enabled) {
    ptzFields.push(readonlyField({
      label: 'Target camera', value: ptz.target_camera_id || 'self',
      hint: 'Camera whose PTZ head this camera drives.'
    }));
    ptzFields.push(readonlyField({
      label: 'Self track', value: ptz.self_track ? 'Yes' : 'No',
      hint: 'This camera’s own detections may contribute to tracking.'
    }));
    ptzFields.push(readonlyField({
      label: 'Multi-camera tracking', value: ptz.multi_camera_tracking ? 'Yes' : 'No',
      hint: 'Allow the target camera’s detections to take over for finer control.'
    }));
    ptzFields.push(readonlyField({
      label: 'Target fill', value: Math.round(ptz.target_fill_pct * 100) + '%',
      hint: 'How much of the frame the animal should fill.'
    }));
    ptzFields.push(readonlyField({
      label: 'Patrol mode', value: ptz.patrol_enabled ? 'On' : 'Off',
      hint: 'Sweep to scan for objects when nothing is detected.'
    }));
    ptzFields.push(readonlyField({
      label: 'Patrol return delay', value: ptz.patrol_return_delay + ' s',
      hint: 'Seconds of quiet before the head returns to its patrol path.'
    }));
  }
  host.appendChild(fieldset('PTZ tracking (read-only)', ptzFields));

  host.appendChild(fieldset('Include species', [
    reg(speciesField({
      path: p.concat(['include_species']),
      label: 'Detect only these species',
      hint: 'Leave empty to detect everything.',
      emptyMeans: 'Nothing selected — every species is detected.',
      recent: null
    }))
  ]));

  host.appendChild(fieldset('Exclude species', [
    reg(speciesField({
      path: p.concat(['exclude_species']),
      label: 'Always ignore these species',
      hint: 'Ignored even when detected. Recent detections on this camera are listed first, with their clip counts.',
      emptyMeans: 'Nothing excluded on this camera.',
      recent: cam.recent_detections
    }))
  ]));
}

/* ==========================================================================
   DIRTY STATE
   ========================================================================= */

function dirtyKeys() {
  var out = {};
  var list = [];
  if (!S.baseline || !S.draft) return { map: out, list: list };
  var paths = allPaths(S.draft);
  for (var i = 0; i < paths.length; i++) {
    var p = paths[i];
    if (!eqValue(getAt(S.draft, p), getAt(S.baseline, p))) {
      out[pathKey(p)] = p;
      list.push(p);
    }
  }
  return { map: out, list: list };
}

function sectionOf(path) {
  if (path[0] === 'global') return 'global';
  return path[1];
}

function restartNotesFor(paths) {
  var notes = {};
  for (var i = 0; i < paths.length; i++) {
    var tail = path_tail(paths[i]);
    if (RESTART_TAILS[tail]) notes[tail] = RESTART_TAILS[tail];
  }
  var out = [];
  for (var k in notes) if (Object.prototype.hasOwnProperty.call(notes, k)) out.push(k);
  return out;
}

function path_tail(path) {
  return path[0] === 'global' ? path.slice(1).join('.') : path.slice(2).join('.');
}

function onModelChanged() {
  refreshDirty();
}

function refreshDirty() {
  var d = dirtyKeys();
  var n = d.list.length;

  /* per-field dots (only the rendered section has controllers) */
  for (var i = 0; i < S.fields.length; i++) {
    var ctl = S.fields[i];
    ctl.setDirty(!!d.map[ctl.key]);
  }

  /* per-section dots in the nav */
  var bySection = {};
  for (var j = 0; j < d.list.length; j++) {
    var sec = sectionOf(d.list[j]);
    bySection[sec] = (bySection[sec] || 0) + 1;
  }
  if (S.navEl) {
    var buttons = S.navEl.querySelectorAll('[data-section]');
    for (var b = 0; b < buttons.length; b++) {
      var el = buttons[b];
      var count = bySection[el.dataset.section] || 0;
      var dot = el.querySelector('.field__dirty');
      if (dot) {
        dot.hidden = count === 0;
        if (count) dot.setAttribute('title', count + ' unsaved ' + plural(count, 'change'));
      }
    }
  }

  /* the save bar */
  if (S.countEl) {
    S.countEl.firstChild.nodeValue = n + ' unsaved ' + plural(n, 'change');
  }
  if (S.countDetailEl) {
    var restarts = restartNotesFor(d.list);
    S.countDetailEl.textContent = restarts.length
      ? restarts.length + ' ' + plural(restarts.length, 'field') + ' need a restart'
      : 'Writes config/cameras.yml';
  }
  if (S.saveBtn) S.saveBtn.disabled = n === 0 || S.saving;
  if (S.resetBtn) S.resetBtn.disabled = n === 0 || S.saving;

  /* setChrome re-renders the whole chrome slice, so only speak when the bar's
     presence actually flips — not on every keystroke. */
  var show = n > 0;
  if (show !== S.selbarShown) {
    S.selbarShown = show;
    store.setChrome({ selbar: show ? S.selbar : null });
  }
  return d;
}

/* ==========================================================================
   SAVE / RESET
   ========================================================================= */

function setSaveBusy(busy) {
  if (!S.saveBtn) return;
  if (busy) S.saveBtn.setAttribute('aria-busy', 'true');
  else S.saveBtn.removeAttribute('aria-busy');
  S.saveBtn.disabled = busy;
  if (S.resetBtn) S.resetBtn.disabled = busy;
  if (S.selbar) {
    if (busy) S.selbar.classList.add('selbar--acting');
    else S.selbar.classList.remove('selbar--acting');
  }
}

function focusPath(path) {
  var sec = sectionOf(path);
  if (sec !== S.section) {
    S.section = sec;
    renderSection();
  }
  var ctl = S.fieldByKey[pathKey(path)];
  if (!ctl) return;
  if (ctl.el && ctl.el.scrollIntoView) ctl.el.scrollIntoView({ block: 'center' });
  if (ctl.focus) ctl.focus();
  if (ctl.setError) ctl.setError(null);
}

function save() {
  if (S.saving || !S.draft || !S.baseline) return;

  var problems = validateModel(S.draft);
  if (problems.length) {
    var first = problems[0];
    focusPath(first.path);
    var ctl = S.fieldByKey[pathKey(first.path)];
    if (ctl && ctl.setError) ctl.setError(first.message);
    toast.danger('Nothing was saved — ' + problems.length + ' ' + plural(problems.length, 'field') + ' failed validation.', {
      detail: first.message
    });
    return;
  }

  var payload;
  try {
    payload = buildPayload(S.draft, cameraIds(S.baseline));
  } catch (err) {
    toast.error('Nothing was sent — the settings payload could not be built.', {
      detail: String(err && err.message ? err.message : err)
    });
    return;
  }

  var d = dirtyKeys();
  var changed = d.list.slice();
  var n = changed.length;
  if (!n) return;

  var prevBaseline = clone(S.baseline);
  S.saving = true;
  setSaveBusy(true);

  /* Optimistic: the baseline advances now, so the form reads as saved while
     the write is in flight. `prevBaseline` is the rollback. */
  S.baseline = clone(S.draft);
  refreshDirty();

  var progress = toast.progress('Writing config/cameras.yml…', {
    detail: n + ' ' + plural(n, 'change') + ' · ' + cameraIds(S.draft).length + ' cameras'
  });

  api.saveSettings(payload, { timeout: 30000, signal: S.abort.signal }).then(function () {
    if (S.destroyed) return;
    progress.close();
    S.saving = false;
    setSaveBusy(false);

    for (var i = 0; i < changed.length; i++) {
      var ctl = S.fieldByKey[pathKey(changed[i])];
      if (ctl && ctl.setStatus) ctl.setStatus('saved', 'Saved');
    }

    var restarts = restartNotesFor(changed);
    toast.success(n + ' ' + plural(n, 'change') + ' saved to config/cameras.yml', {
      detail: restarts.length
        ? restarts.join(', ') + ' — ' + plural(restarts.length, 'this field takes', 'these fields take') +
          ' effect after a restart'
        : 'Applied to the running process immediately'
    });
    refreshDirty();
  }, function (err) {
    if (S.destroyed) return;
    progress.close();
    S.saving = false;
    setSaveBusy(false);
    if (api.isAbort(err)) return;

    /* Roll the baseline back: every edit becomes dirty again, still in the
       controls, still editable. A failed YAML write must never eat work. */
    S.baseline = prevBaseline;
    refreshDirty();
    for (var i = 0; i < changed.length; i++) {
      var ctl = S.fieldByKey[pathKey(changed[i])];
      if (ctl && ctl.setStatus) ctl.setStatus('error', 'Not saved');
    }
    toast.error('config/cameras.yml was NOT written — your edits are still here.', {
      detail: api.describe(err),
      retry: save
    });
  });
}

function resetDraft() {
  var d = dirtyKeys();
  if (!d.list.length) return;
  var n = d.list.length;
  var dlg = dialog({
    role: 'alertdialog',
    tone: 'danger',
    title: 'Discard ' + n + ' unsaved ' + plural(n, 'change') + '?',
    body: 'The form returns to the values the server last confirmed. Nothing on disk changes.',
    stakes: n + ' ' + plural(n, 'field') + ' across ' + describeSections(d.list),
    actions: [
      { label: 'Keep editing', variant: 'secondary', value: false, focus: true },
      { label: 'Discard changes', variant: 'danger', value: true }
    ]
  });
  dlg.result.then(function (v) {
    if (v !== true || S.destroyed) return;
    S.draft = clone(S.baseline);
    renderSection();
    refreshDirty();
    toast.info(n + ' ' + plural(n, 'change') + ' discarded');
  });
}

function describeSections(paths) {
  var seen = {};
  var names = [];
  for (var i = 0; i < paths.length; i++) {
    var sec = sectionOf(paths[i]);
    if (seen[sec]) continue;
    seen[sec] = 1;
    names.push(sec === 'global' ? 'Global'
      : (S.draft.cameras[sec] ? S.draft.cameras[sec].name : sec));
  }
  return names.join(', ');
}

/* ==========================================================================
   YAML PREVIEW
   ========================================================================= */

function revealYaml() {
  var text;
  try {
    if (S.section === 'global') {
      text = 'general:\n' + toYaml(buildGlobalPayload(S.draft), 1);
    } else {
      var cam = buildCameraPayload(S.draft, S.section);
      text = 'cameras:\n  - id: ' + yamlScalar(S.section) + '\n' +
        toYaml(cam, 2);
    }
  } catch (err) {
    text = '# This section cannot be serialised yet:\n# ' +
      String(err && err.message ? err.message : err);
  }
  var pre = h('pre.code.mono', {
    tabIndex: 0,
    style: { overflowX: 'auto', whiteSpace: 'pre', margin: '0' }
  });
  pre.textContent = text;
  dialog({
    role: 'dialog',
    title: 'What this section writes',
    body: 'Read-only. The server rewrites config/cameras.yml with yaml.dump(), so comments and blank lines in the file are not preserved.',
    width: 640,
    content: pre,
    actions: [{ label: 'Close', variant: 'secondary', value: null, focus: true }]
  });
}

/* ==========================================================================
   NAV + LAYOUT
   ========================================================================= */

function buildNav() {
  var nav = h('nav.stack.stack--tight', { 'aria-label': 'Settings sections' });
  var list = h('div.row.row--wrap', { style: { gap: 'var(--s-2)' } });
  nav.appendChild(list);
  S.navList = list;
  return nav;
}

function renderNav() {
  var items = [{ id: 'global', label: 'Global', iconName: 'settings' }];
  var ids = cameraIds(S.draft);
  for (var i = 0; i < ids.length; i++) {
    items.push({ id: ids[i], label: S.draft.cameras[ids[i]].name || ids[i], iconName: 'camera' });
  }
  keyedList(S.navList, items, {
    key: function (item) { return item.id; },
    create: function (item) {
      var btn = h('button.tab', {
        type: 'button',
        dataset: { section: item.id }
      },
        h('span', { 'aria-hidden': 'true' }, icon(item.iconName, { size: 'sm' })),
        h('span.truncate', { dataset: { role: 'label' } }),
        h('span.field__dirty', { hidden: true, 'aria-hidden': 'true' }));
      return btn;
    },
    update: function (node, item) {
      var lab = node.querySelector('[data-role="label"]');
      if (lab) lab.textContent = item.label;
      var active = item.id === S.section;
      if (active) node.setAttribute('aria-current', 'page');
      else node.removeAttribute('aria-current');
      node.setAttribute('aria-label', item.label + ' settings');
    }
  });
}

function renderSection() {
  S.fields = [];
  S.fieldByKey = {};
  clear(S.panelEl);

  var titleId = uid('sec');
  var title = S.section === 'global'
    ? 'Global'
    : (S.draft.cameras[S.section] ? S.draft.cameras[S.section].name : S.section);

  var yamlBtn = h('button.btn.btn--secondary.btn--sm', { type: 'button' },
    h('span.btn__icon', { 'aria-hidden': 'true' }, icon('layers', { size: 'sm' })),
    h('span.btn__label', 'Reveal in YAML'));
  track(on(yamlBtn, 'click', revealYaml));

  var head = h('div.row.row--between',
    h('h2.t-h3#' + titleId, { tabIndex: -1, text: title }),
    yamlBtn);

  S.panelEl.setAttribute('aria-labelledby', titleId);
  S.panelEl.appendChild(head);

  var body = h('div.stack.stack--loose.field-form');
  S.panelEl.appendChild(body);

  if (S.section === 'global') renderGlobalSection(body);
  else renderCameraSection(body, S.section);

  renderNav();
  refreshDirty();
}

function selectSection(id) {
  if (id === S.section) return;
  S.section = id;
  renderSection();
  var heading = S.panelEl.querySelector('h2');
  if (heading && heading.focus) heading.focus();
}

/* ==========================================================================
   LOADING
   ========================================================================= */

function skeleton() {
  var host = h('div.stack.stack--loose');
  for (var i = 0; i < 6; i++) {
    host.appendChild(h('div.stack.stack--tight',
      h('span.skel.skel--text', { style: { width: '30%' } }),
      h('span.skel.skel--row')));
  }
  host.setAttribute('aria-hidden', 'true');
  return host;
}

function errorState(err, retry) {
  var box = h('div.empty.empty--error',
    h('div.empty__art', icon('alert', { size: 'lg' })),
    h('h2.empty__title', 'Settings could not be loaded'),
    h('p.empty__body', { text: api.describe(err) }),
    h('p.empty__endpoint', { text: '/api/settings' }),
    h('div.empty__actions'));
  var again = h('button.btn.btn--primary', { type: 'button' },
    h('span.btn__icon', { 'aria-hidden': 'true' }, icon('refresh', { size: 'sm' })),
    h('span.btn__label', 'Try again'));
  track(on(again, 'click', retry));
  box.querySelector('.empty__actions').appendChild(again);
  return box;
}

function load(opts) {
  var o = opts || {};
  if (!o.quiet) {
    clear(S.contentEl);
    S.contentEl.appendChild(skeleton());
  }
  if (S.refreshAbort) { try { S.refreshAbort.abort(); } catch (e) {} S.refreshAbort = null; }
  var ctrl = null;
  if (typeof AbortController === 'function') {
    ctrl = new AbortController();
    S.refreshAbort = ctrl;
  }
  return api.settings({ signal: ctrl ? ctrl.signal : undefined, timeout: 20000 }).then(function (raw) {
    if (S.destroyed) return;
    S.refreshAbort = null;
    var model = normalize(raw);
    if (!cameraIds(model).length && !o.quiet) {
      clear(S.contentEl);
      S.contentEl.appendChild(h('div.empty',
        h('div.empty__art', icon('camera', { size: 'lg' })),
        h('h2.empty__title', 'No cameras are configured'),
        h('p.empty__body', 'The running process has no camera workers, so there is nothing to configure. Add a camera to cameras.yml and restart.')));
      return;
    }
    if (o.quiet) {
      /* A background refresh must never overwrite work in progress. */
      if (dirtyKeys().list.length) return;
      if (!S.panelEl) return;
      if (S.panelEl.contains(document.activeElement)) return;
      var sameShape = JSON.stringify(cameraIds(model)) === JSON.stringify(cameraIds(S.draft || {}));
      S.baseline = model;
      S.draft = clone(model);
      if (!sameShape && !S.draft.cameras[S.section] && S.section !== 'global') S.section = 'global';
      renderSection();
      return;
    }
    S.baseline = model;
    S.draft = clone(model);
    if (S.section !== 'global' && !S.draft.cameras[S.section]) S.section = 'global';
    buildBody();
  }, function (err) {
    if (S.destroyed || api.isAbort(err)) return;
    S.refreshAbort = null;
    if (o.quiet) {
      toast.danger('Could not refresh settings', { detail: api.describe(err) });
      return;
    }
    clear(S.contentEl);
    S.contentEl.appendChild(errorState(err, function () { load({}); }));
    toast.error('Settings could not be loaded.', { detail: api.describe(err), retry: function () { load({}); } });
  });
}

/* ==========================================================================
   BODY
   ========================================================================= */

function buildBody() {
  clear(S.contentEl);

  var warn = h('p.field__hint', { style: { margin: '0' } },
    icon('alert', { size: 'sm' }),
    h('span', { text: ' Saving rewrites config/cameras.yml in place. That file is not in version control and the server does not keep a backup; comments in it are not preserved.' }));

  var nav = buildNav();
  S.navEl = nav;

  S.panelEl = h('section', { role: 'region', tabIndex: -1 });

  var layout = h('div', {
    style: {
      display: 'grid',
      gap: 'var(--s-6)',
      alignItems: 'start',
      gridTemplateColumns: '1fr'
    }
  }, nav, S.panelEl);
  S.layoutEl = layout;
  applyLayout();

  S.contentEl.appendChild(h('div.stack.stack--loose', warn, layout));

  track(delegate(nav, 'click', '[data-section]', function (ev, node) {
    ev.preventDefault();
    selectSection(node.dataset.section);
  }));

  renderSection();
}

function applyLayout() {
  if (!S.layoutEl) return;
  var wide = window.matchMedia && window.matchMedia('(min-width: 1024px)').matches;
  S.layoutEl.style.gridTemplateColumns = wide ? '220px minmax(0, 1fr)' : '1fr';
  if (S.navList) {
    S.navList.style.flexDirection = wide ? 'column' : 'row';
    S.navList.style.alignItems = wide ? 'stretch' : 'center';
  }
}

/* ==========================================================================
   SAVE BAR
   ========================================================================= */

function buildSelbar() {
  var count = h('div.selbar__count', { role: 'status', 'aria-live': 'polite' });
  count.appendChild(document.createTextNode('0 unsaved changes'));
  var detail = h('span');
  count.appendChild(detail);
  S.countEl = count;
  S.countDetailEl = detail;

  var reset = h('button.btn.btn--secondary', { type: 'button', disabled: true },
    h('span.btn__icon', { 'aria-hidden': 'true' }, icon('undo', { size: 'sm' })),
    h('span.btn__label', 'Reset'));
  var saveBtn = h('button.btn.btn--primary', { type: 'button', disabled: true },
    h('span.btn__icon', { 'aria-hidden': 'true' }, icon('check', { size: 'sm' })),
    h('span.btn__spinner', { 'aria-hidden': 'true' }, h('span.spinner')),
    h('span.btn__label', 'Save changes'));
  S.resetBtn = reset;
  S.saveBtn = saveBtn;

  track(on(reset, 'click', resetDraft));
  track(on(saveBtn, 'click', save));

  return h('div.selbar', count, h('div.spacer'),
    h('div.selbar__actions', reset, h('span.selbar__sep'), saveBtn));
}

/* ==========================================================================
   NAVIGATION GUARD
   ========================================================================= */

function installGuards() {
  track(on(window, 'beforeunload', function (ev) {
    if (S.destroyed || !dirtyKeys().list.length) return;
    ev.preventDefault();
    ev.returnValue = '';
    return '';
  }));

  /* In-app links (tab bar, app bar, rail) are plain anchors that app.js
     intercepts. We run first, in the capture phase, and only when there is
     something to lose. */
  track(on(document, 'click', function (ev) {
    if (S.destroyed || S.saving) return;
    if (ev.defaultPrevented || ev.button !== 0) return;
    if (ev.metaKey || ev.ctrlKey || ev.shiftKey || ev.altKey) return;
    var node = ev.target;
    while (node && node !== document && node.tagName !== 'A') node = node.parentElement;
    if (!node || node === document || !node.getAttribute) return;
    var href = node.getAttribute('href');
    if (!href || href.charAt(0) === '#') return;
    if (node.target && node.target !== '_self') return;
    if (node.host && node.host !== window.location.host) return;
    var here = window.location.pathname + window.location.search;
    if (href === here) return;
    var d = dirtyKeys();
    if (!d.list.length) return;

    ev.preventDefault();
    ev.stopPropagation();
    var n = d.list.length;
    var dlg = dialog({
      role: 'alertdialog',
      tone: 'danger',
      title: 'Discard ' + n + ' unsaved ' + plural(n, 'change') + '?',
      body: 'Leaving this screen throws away edits that were never written to config/cameras.yml.',
      stakes: n + ' ' + plural(n, 'field') + ' across ' + describeSections(d.list),
      actions: [
        { label: 'Stay here', variant: 'secondary', value: 'stay', focus: true },
        { label: 'Save and leave', variant: 'primary', value: 'save' },
        { label: 'Discard and leave', variant: 'danger', value: 'go' }
      ]
    });
    dlg.result.then(function (v) {
      if (v === 'go') {
        S.baseline = clone(S.draft);   /* silence the guard, then navigate */
        refreshDirty();
        router.navigate(href);
      } else if (v === 'save') {
        save();
      }
    });
  }, true));

  track(on(document, 'visibilitychange', function () {
    if (S.destroyed) return;
    if (document.hidden) {
      if (S.refreshAbort) { S.refreshAbort.abort(); S.refreshAbort = null; }
      return;
    }
    if (!S.baseline) return;
    if (dirtyKeys().list.length) return;   /* never clobber pending edits */
    load({ quiet: true });
  }));

  if (window.matchMedia) {
    var mq = window.matchMedia('(min-width: 1024px)');
    var onMq = function () { applyLayout(); };
    if (mq.addEventListener) {
      mq.addEventListener('change', onMq);
      track(function () { mq.removeEventListener('change', onMq); });
    } else if (mq.addListener) {
      mq.addListener(onMq);
      track(function () { mq.removeListener(onMq); });
    }
  }
}

/* ==========================================================================
   THE VIEW
   ========================================================================= */

export var view = {
  mount: function (root, ctx) {
    S = newSession();
    S.root = root;
    S.destroyed = false;
    if (typeof AbortController === 'function') S.abort = new AbortController();
    else S.abort = { signal: undefined, abort: function () {} };

    var q = (ctx && ctx.query) || {};
    if (q.section) S.section = String(q.section);
    else if (q.camera) S.section = String(q.camera);

    S.selbar = buildSelbar();

    store.setChrome({
      title: 'Settings',
      subtitle: 'Detection, clips, retention and cameras',
      actions: [],
      toolbar: null,
      rail: null,
      norail: true,
      selbar: null,
      mods: []
    });

    root.appendChild(h('h1.visually-hidden', { tabIndex: -1, text: 'Settings' }));
    S.contentEl = h('div');
    root.appendChild(S.contentEl);

    installGuards();
    load({});
  },

  unmount: function () {
    if (!S) return;
    S.destroyed = true;
    if (S.abort && S.abort.abort) { try { S.abort.abort(); } catch (e) {} }
    if (S.refreshAbort) { try { S.refreshAbort.abort(); } catch (e2) {} }
    for (var i = 0; i < S.offs.length; i++) {
      try { S.offs[i](); } catch (e3) {}
    }
    for (var t = 0; t < S.savedTimers.length; t++) clearTimeout(S.savedTimers[t]);
    S.offs = [];
    S.fields = [];
    S.fieldByKey = {};
    store.setChrome({ selbar: null });
    S = null;
  }
};
