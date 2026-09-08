/* ============================================================================
   core/format.js — every string the user reads that is derived from a number
   or a timestamp.

   TIMEZONE POLICY (load-bearing)
   The server formats clip timestamps in ITS configured timezone and ships them
   as ISO-8601 WITH an offset: "2026-09-07T03:49:30-05:00". A phone in another
   timezone must still see the camera's wall clock, so wall-clock formatting
   reads the fields out of the string and ignores the browser's zone. Only
   elapsed time ("4 minutes ago") uses the absolute instant, because elapsed
   time is zone-independent.

   PUBLIC API
     parseServerTime(iso)    -> { at: Date, y, mo, d, h, mi, s, offset }
     clockTime(iso, opts?)   -> "3:49 AM"
     dayKey(iso)             -> "2026-09-07"
     longDate(dateStr)       -> "Monday, 7 September 2026"
     dayLabel(dateStr, now?) -> { relative: "Today", full: "Monday, 7 September" }
     monthLabel(y, m)        -> "September 2026"
     timeAgo(iso|epoch, now?)-> "Just now" | "12 minutes ago" | "Sep 3"
     shortAgo(seconds)       -> "42s" | "4m 12s" | "2h"
     fileSize(bytes)         -> "79.7 KB"
     mb(sizeMb)              -> "0.08 MB"
     durationClock(seconds)  -> "0:18"
     plural(n, one, many?)   -> "3 recordings"
     speciesClass(name)      -> "sp--deer"
     speciesKey(name)        -> "deer"
     isUnclassified(name)    -> boolean
     filmClass(iso)          -> "film--night"
     pct(x)                  -> "94%"
     confidenceSegments(x)   -> 0..8
   ========================================================================= */

var MONTHS = ['January', 'February', 'March', 'April', 'May', 'June', 'July',
  'August', 'September', 'October', 'November', 'December'];
var MONTHS_SHORT = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul',
  'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
var DAYS = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday'];
var DAYS_SHORT = ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat'];

export var MONTH_NAMES = MONTHS;
export var MONTH_NAMES_SHORT = MONTHS_SHORT;
export var DAY_NAMES = DAYS;
export var DAY_NAMES_SHORT = DAYS_SHORT;

var ISO_RE = /^(\d{4})-(\d{2})-(\d{2})[T ](\d{2}):(\d{2}):(\d{2})/;

/**
 * Split a server timestamp into its wall-clock fields plus the absolute
 * instant. Returns null for anything unparseable — every caller checks.
 */
export function parseServerTime(iso) {
  if (!iso) return null;
  if (typeof iso === 'number') {
    var d = new Date(iso * 1000);
    return {
      at: d, y: d.getFullYear(), mo: d.getMonth() + 1, d: d.getDate(),
      h: d.getHours(), mi: d.getMinutes(), s: d.getSeconds(), offset: null
    };
  }
  var m = ISO_RE.exec(String(iso));
  if (!m) return null;
  var at = new Date(Date.parse(iso));
  if (isNaN(at.getTime())) at = null;
  return {
    at: at,
    y: +m[1], mo: +m[2], d: +m[3],
    h: +m[4], mi: +m[5], s: +m[6],
    offset: (/([+-]\d{2}:?\d{2}|Z)$/.exec(String(iso)) || [null])[0]
  };
}

function pad2(n) { return n < 10 ? '0' + n : String(n); }

/** "3:49 AM" in the CAMERA's timezone. Pass {seconds:true} for "3:49:30 AM". */
export function clockTime(iso, opts) {
  var t = parseServerTime(iso);
  if (!t) return '--:--';
  var h = t.h % 12;
  if (h === 0) h = 12;
  var suffix = t.h < 12 ? 'AM' : 'PM';
  var out = h + ':' + pad2(t.mi);
  if (opts && opts.seconds) out += ':' + pad2(t.s);
  return out + ' ' + suffix;
}

/** "2026-09-07" from a server timestamp, in the camera's timezone. */
export function dayKey(iso) {
  var t = parseServerTime(iso);
  if (!t) return '';
  return t.y + '-' + pad2(t.mo) + '-' + pad2(t.d);
}

/* A YYYY-MM-DD string is a calendar date, not an instant: build it as a LOCAL
   Date so no timezone can shift it onto the neighbouring day. */
function dateFromKey(dateStr) {
  var p = /^(\d{4})-(\d{2})-(\d{2})$/.exec(String(dateStr || ''));
  if (!p) return null;
  return new Date(+p[1], +p[2] - 1, +p[3]);
}
export { dateFromKey };

export function keyFromDate(d) {
  return d.getFullYear() + '-' + pad2(d.getMonth() + 1) + '-' + pad2(d.getDate());
}

/** "Monday, 7 September 2026" */
export function longDate(dateStr) {
  var d = dateFromKey(dateStr);
  if (!d) return String(dateStr || '');
  return DAYS[d.getDay()] + ', ' + d.getDate() + ' ' + MONTHS[d.getMonth()] + ' ' + d.getFullYear();
}

/**
 * The archive spine's header: "Today · Monday, 7 September".
 * `relative` is null on any day older than yesterday.
 */
export function dayLabel(dateStr, now) {
  var d = dateFromKey(dateStr);
  if (!d) return { relative: null, full: String(dateStr || 'Undated') };
  var today = now ? new Date(now.getTime()) : new Date();
  today.setHours(0, 0, 0, 0);
  var diff = Math.round((today.getTime() - d.getTime()) / 86400000);
  var relative = null;
  if (diff === 0) relative = 'Today';
  else if (diff === 1) relative = 'Yesterday';
  var full = DAYS[d.getDay()] + ', ' + d.getDate() + ' ' + MONTHS[d.getMonth()];
  if (d.getFullYear() !== today.getFullYear()) full += ' ' + d.getFullYear();
  return { relative: relative, full: full };
}

export function monthLabel(year, month) {
  return MONTHS[Math.max(0, Math.min(11, month - 1))] + ' ' + year;
}

/**
 * Relative age. Buckets match the ladder the old UI shipped, so an operator's
 * reading habits survive the rewrite.
 */
export function timeAgo(value, now) {
  var t = typeof value === 'number' ? new Date(value * 1000)
        : (value instanceof Date ? value : (parseServerTime(value) || {}).at);
  if (!t || isNaN(t.getTime())) return '';
  var nowMs = now ? now.getTime() : Date.now();
  var sec = Math.round((nowMs - t.getTime()) / 1000);
  if (sec < 0) sec = 0;
  if (sec < 45) return 'Just now';
  var min = Math.round(sec / 60);
  if (min < 60) return min + (min === 1 ? ' minute ago' : ' minutes ago');
  var hr = Math.floor(sec / 3600);
  if (hr < 24) return hr + (hr === 1 ? ' hour ago' : ' hours ago');
  var days = Math.floor(sec / 86400);
  if (days === 1) return 'Yesterday';
  if (days < 7) return days + ' days ago';
  var p = parseServerTime(value);
  if (p) return MONTHS_SHORT[p.mo - 1] + ' ' + p.d;
  return MONTHS_SHORT[t.getMonth()] + ' ' + t.getDate();
}

/** Compact machine age for a readout: "42s", "4m 12s", "2h 06m". */
export function shortAgo(seconds) {
  if (seconds === null || seconds === undefined || isNaN(seconds)) return '--';
  var s = Math.max(0, Math.floor(seconds));
  if (s < 60) return s + 's';
  if (s < 3600) return Math.floor(s / 60) + 'm ' + pad2(s % 60) + 's';
  if (s < 86400) return Math.floor(s / 3600) + 'h ' + pad2(Math.floor((s % 3600) / 60)) + 'm';
  return Math.floor(s / 86400) + 'd';
}

export function fileSize(bytes) {
  var b = Number(bytes);
  if (!isFinite(b) || b < 0) return '--';
  if (b < 1024) return b + ' B';
  if (b < 1024 * 1024) return (b / 1024).toFixed(1) + ' KB';
  if (b < 1024 * 1024 * 1024) return (b / 1048576).toFixed(1) + ' MB';
  return (b / 1073741824).toFixed(2) + ' GB';
}

export function mb(sizeMb) {
  var v = Number(sizeMb);
  if (!isFinite(v)) return '--';
  if (v >= 1024) return (v / 1024).toFixed(2) + ' GB';
  return v.toFixed(v < 10 ? 2 : 1) + ' MB';
}

export function durationClock(seconds) {
  var s = Number(seconds);
  if (!isFinite(s) || s < 0) return '0:00';
  var m = Math.floor(s / 60);
  var r = Math.floor(s % 60);
  return m + ':' + pad2(r);
}

export function plural(n, one, many) {
  var word = n === 1 ? one : (many || one + 's');
  return n + ' ' + word;
}

/* --- Species colour channel ------------------------------------------------
   The stylesheet ships nine species token trios. Anything outside them lands
   on --unknown, which is italic and differently weighted, not merely a
   different hue. Matching is on the DISPLAY name the server already derived
   (common names via get_common_name), lowercased. */
var SPECIES_RULES = [
  ['deer', /deer|elk|moose|cervid/],
  ['squirrel', /squirrel|chipmunk|marmot|sciurid|groundhog|woodchuck/],
  ['rabbit', /rabbit|cottontail|hare|lagomorph/],
  ['coyote', /coyote|wolf|dog|canid|domestic dog/],
  ['raccoon', /raccoon|procyon/],
  ['opossum', /opossum|possum|didelph|skunk|badger|weasel|mustelid/],
  ['bird', /bird|cardinal|crow|jay|owl|hawk|robin|sparrow|finch|wren|dove|turkey|goose|duck|heron|aves/],
  ['fox', /fox|vulpes|bobcat|lynx|cat|felid/]
];

var UNCLASSIFIED = /^(unknown|unclassified|animal|blank|empty|no cv result|manual clip)$/;

export function isUnclassified(name) {
  return UNCLASSIFIED.test(String(name || '').trim().toLowerCase());
}

export function speciesKey(name) {
  var n = String(name || '').toLowerCase();
  if (!n || isUnclassified(n)) return 'unknown';
  for (var i = 0; i < SPECIES_RULES.length; i++) {
    if (SPECIES_RULES[i][1].test(n)) return SPECIES_RULES[i][0];
  }
  return 'unknown';
}

export function speciesClass(name) {
  return 'sp--' + speciesKey(name);
}

/**
 * Film stock behind a thumbnail: a plausible poster while the real frame is in
 * flight, chosen from the capture hour so a night clip does not flash daylight.
 */
export function filmClass(iso) {
  var t = parseServerTime(iso);
  if (!t) return 'film--night';
  var h = t.h;
  if (h >= 22 || h < 5) return 'film--ir';
  if (h < 7) return 'film--dawn';
  if (h < 17) return 'film--day';
  if (h < 20) return 'film--dusk';
  return 'film--night';
}

export function pct(x) {
  var v = Number(x);
  if (!isFinite(v)) return '--';
  if (v <= 1) v = v * 100;
  return Math.round(v) + '%';
}

/** Lit segments for an 8-segment confidence .meter. */
export function confidenceSegments(x, of) {
  var total = of || 8;
  var v = Number(x);
  if (!isFinite(v)) return 0;
  if (v > 1) v = v / 100;
  return Math.max(0, Math.min(total, Math.round(v * total)));
}

/** "3 cameras · 11 species · 11.4 GB" style joiner that drops empty parts. */
export function joinMeta() {
  var out = [];
  for (var i = 0; i < arguments.length; i++) {
    if (arguments[i] !== null && arguments[i] !== undefined && arguments[i] !== '') {
      out.push(String(arguments[i]));
    }
  }
  return out.join(' · ');
}
