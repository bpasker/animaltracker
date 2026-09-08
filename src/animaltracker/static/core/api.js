/* ============================================================================
   core/api.js — every call to the aiohttp server.

   No view calls fetch() directly. This module owns:
     · URL building (query strings, path encoding that survives '+' and spaces)
     · timeouts and cancellation (every call takes an AbortSignal)
     · one error type, ApiError, carrying status, endpoint and the server's
       own words — the diagnostic empty states and error toasts quote them
     · the connection flag: the first failure that looks like "server gone"
       flips store.connected, and the first success flips it back.

   NOTHING here shows UI. A caller that swallows an ApiError without showing
   something is a bug (constraint 10).

   PUBLIC API
     api.cameras(opts)                       -> { cameras, timezone }
     api.recordings(query, opts)             -> { clips, total, archive_total,
                                                  offset, limit, has_more, facets }
     api.clip(path, opts)                    -> clip detail
     api.calendar(opts)                      -> { years, filters }
     api.day(date, query, opts)              -> { date, clips, summary }
     api.monitor(opts)                       -> monitor payload
     api.logs(query, opts)                   -> logs payload
     api.settings(opts) / api.saveSettings(body, opts)
     api.deleteClip(path, opts)              -> true
     api.bulkDelete(paths, opts)             -> { deleted_count, ... }
     api.reprocess(path, settings, opts)     -> server payload
     api.processingLog(path, opts)           -> { exists, ... }
     api.saveClip(cameraId, opts)            -> server payload
     api.ptz.*                               -> PTZ control surface
     api.clipUrl(path) / api.thumbUrl(path)  -> media URLs (never fetched here)

     ApiError                                 { status, endpoint, detail }
     api.describe(err)                        -> one sentence a human can act on

   Every `opts` accepts { signal, timeout } — timeout defaults to 15s, or 0 to
   wait forever (used by nothing today; reprocess passes its own).
   ========================================================================= */

import { store } from './store.js';

var DEFAULT_TIMEOUT = 15000;

export function ApiError(message, info) {
  this.name = 'ApiError';
  this.message = message;
  this.status = (info && info.status) || 0;
  this.endpoint = (info && info.endpoint) || '';
  this.detail = (info && info.detail) || '';
  this.cause = info && info.cause;
  /* Not a subclass of Error on purpose: Safari 15 loses the prototype chain
     through some transpiled paths, and every consumer checks .name. */
  this.stack = (info && info.cause && info.cause.stack) || new Error(message).stack;
}
ApiError.prototype.toString = function () { return this.name + ': ' + this.message; };

export function isAbort(err) {
  return !!err && (err.name === 'AbortError' || err.code === 20);
}

/* --- URLs ---------------------------------------------------------------- */

/** Encode a clip path ("cam1/2026/09/07/x.mp4") for a path segment position. */
export function encodePath(path) {
  return String(path || '').split('/').map(encodeURIComponent).join('/');
}

function qs(params) {
  if (!params) return '';
  var parts = [];
  for (var k in params) {
    if (!Object.prototype.hasOwnProperty.call(params, k)) continue;
    var v = params[k];
    if (v === null || v === undefined || v === '') continue;
    if (Array.isArray(v)) {
      if (!v.length) continue;
      v = v.join(',');
    }
    parts.push(encodeURIComponent(k) + '=' + encodeURIComponent(String(v)));
  }
  return parts.length ? '?' + parts.join('&') : '';
}
export { qs };

/* --- The one fetch ------------------------------------------------------- */

function request(method, endpoint, opts) {
  var o = opts || {};
  var timeout = o.timeout === undefined ? DEFAULT_TIMEOUT : o.timeout;
  var ctrl = null;
  var signal = o.signal;
  var timer = null;
  var timedOut = false;

  /* Compose the caller's signal with our timeout. AbortSignal.any() is far
     newer than the iOS 15 floor, so it is done by hand, and `timedOut`
     separates "we gave up" from "the caller cancelled". */
  if (timeout > 0 && typeof AbortController === 'function') {
    ctrl = new AbortController();
    timer = setTimeout(function () { timedOut = true; ctrl.abort(); }, timeout);
    if (signal) {
      if (signal.aborted) ctrl.abort();
      else signal.addEventListener('abort', function () { ctrl.abort(); });
    }
    signal = ctrl.signal;
  }

  var init = { method: method, signal: signal, credentials: 'same-origin' };
  init.headers = { 'Accept': 'application/json' };
  if (o.body !== undefined) {
    init.headers['Content-Type'] = 'application/json';
    init.body = JSON.stringify(o.body);
  }
  if (o.keepalive) init.keepalive = true;

  return fetch(endpoint, init).then(function (res) {
    if (timer) clearTimeout(timer);
    store.set({ connected: true });
    if (!res.ok) {
      return res.text().then(function (text) {
        throw new ApiError(
          httpMessage(res.status, endpoint),
          { status: res.status, endpoint: endpoint, detail: (text || '').slice(0, 400) }
        );
      }, function () {
        throw new ApiError(httpMessage(res.status, endpoint),
          { status: res.status, endpoint: endpoint });
      });
    }
    if (o.raw) return res;
    var type = res.headers.get('Content-Type') || '';
    if (type.indexOf('application/json') < 0) return res.text();
    return res.json().catch(function (err) {
      throw new ApiError('The server sent a malformed response.',
        { status: res.status, endpoint: endpoint, cause: err });
    });
  }, function (err) {
    if (timer) clearTimeout(timer);
    if (isAbort(err) && !timedOut) throw err;         /* caller cancelled */
    if (timedOut) {
      throw new ApiError('The request timed out after ' + Math.round(timeout / 1000) + 's.',
        { status: 0, endpoint: endpoint, cause: err });
    }
    /* A network-level failure is the "server gone" case. */
    store.set({ connected: false });
    throw new ApiError('Could not reach the Animal Tracker server.',
      { status: 0, endpoint: endpoint, cause: err });
  });
}

function httpMessage(status, endpoint) {
  if (status === 403) return 'The server refused that path.';
  if (status === 404) return 'Not found on the server.';
  if (status === 409) return 'That job is already running.';
  if (status >= 500) return 'The server failed while handling ' + endpoint + '.';
  return 'The server rejected the request (HTTP ' + status + ').';
}

/**
 * One sentence naming the cause and, where there is one, the remedy.
 * Used by every toast and every .empty--error.
 */
function describe(err) {
  if (!err) return 'Something failed, and the failure carried no detail.';
  if (isAbort(err)) return 'The request was cancelled.';
  if (err.name === 'ApiError') {
    var text = err.message;
    if (err.detail) text += ' ' + err.detail;
    return text;
  }
  return String(err.message || err);
}

/* --- The typed surface --------------------------------------------------- */

export var api = {
  ApiError: ApiError,
  describe: describe,
  isAbort: isAbort,
  encodePath: encodePath,

  /** Media URLs. Never fetched here — they go into <img src> / <video src>. */
  clipUrl: function (path) { return '/clips/' + encodePath(path); },
  thumbUrl: function (pathOrUrl) {
    var s = String(pathOrUrl || '');
    if (s.indexOf('/') === 0 || s.indexOf('http') === 0) return s;
    return '/clips/' + encodePath(s);
  },

  cameras: function (opts) {
    return request('GET', '/api/cameras', opts);
  },

  /**
   * query: { camera, species, from, to, q, sort, limit, offset }
   * camera/species may be arrays; they are joined with commas (OR within a
   * category, AND across categories — the server's semantics).
   */
  recordings: function (query, opts) {
    return request('GET', '/api/recordings' + qs(query), opts);
  },

  clip: function (path, opts) {
    return request('GET', '/api/clip/' + encodePath(path), opts);
  },

  calendar: function (opts) {
    return request('GET', '/api/recordings/calendar', opts);
  },

  /** query: { camera, species } — single values; the server matches species
      as a case-insensitive substring, so multi-select is refined client-side. */
  day: function (date, query, opts) {
    return request('GET', '/api/recordings/day/' + encodeURIComponent(date) + qs(query), opts);
  },

  monitor: function (opts) {
    return request('GET', '/api/monitor', opts);
  },

  logs: function (query, opts) {
    return request('GET', '/api/logs' + qs(query), opts);
  },

  settings: function (opts) {
    return request('GET', '/api/settings', opts);
  },

  saveSettings: function (body, opts) {
    return request('POST', '/api/settings', Object.assign({ body: body }, opts || {}));
  },

  /* The server deletes for real — there is no soft-delete window on disk.
     The undo affordance in the UI is therefore a DEFERRED delete: the view
     holds the request until the toast's deadline expires. */
  deleteClip: function (path, opts) {
    return request('DELETE', '/recordings?path=' + encodeURIComponent(path), opts)
      .then(function () { return true; });
  },

  bulkDelete: function (paths, opts) {
    return request('POST', '/recordings/bulk_delete',
      Object.assign({ body: { paths: paths } }, opts || {}));
  },

  reprocess: function (path, settings, opts) {
    var body = Object.assign({ path: path }, settings || {});
    return request('POST', '/recordings/reprocess',
      Object.assign({ body: body, timeout: 0 }, opts || {}));
  },

  processingLog: function (path, opts) {
    return request('GET', '/recordings/log/' + encodePath(path), opts);
  },

  saveClip: function (cameraId, opts) {
    return request('POST', '/save_clip/' + encodeURIComponent(cameraId),
      Object.assign({ timeout: 30000 }, opts || {}));
  },

  ptz: {
    move: function (id, vec, opts) {
      return request('POST', '/ptz/' + encodeURIComponent(id),
        Object.assign({ body: Object.assign({ action: 'move' }, vec) }, opts || {}));
    },
    stop: function (id, opts) {
      return request('POST', '/ptz/' + encodeURIComponent(id),
        Object.assign({ body: { action: 'stop' } }, opts || {}));
    },
    position: function (id, opts) {
      return request('GET', '/ptz/' + encodeURIComponent(id) + '/position', opts);
    },
    mode: function (id, opts) {
      return request('GET', '/ptz/' + encodeURIComponent(id) + '/mode', opts);
    },
    presets: function (id, opts) {
      return request('GET', '/ptz/' + encodeURIComponent(id) + '/presets', opts);
    },
    setPatrolPresets: function (id, tokens, opts) {
      return request('POST', '/ptz/' + encodeURIComponent(id) + '/presets',
        Object.assign({ body: { presets: tokens } }, opts || {}));
    },
    gotoPreset: function (id, token, opts) {
      return request('POST', '/ptz/' + encodeURIComponent(id) + '/goto_preset',
        Object.assign({ body: { token: token } }, opts || {}));
    },
    savePreset: function (id, name, opts) {
      return request('POST', '/ptz/' + encodeURIComponent(id) + '/save_preset',
        Object.assign({ body: { name: name } }, opts || {}));
    },
    patrol: function (id, enabled, opts) {
      return request('POST', '/ptz/' + encodeURIComponent(id) + '/patrol',
        Object.assign({ body: { enabled: !!enabled } }, opts || {}));
    },
    track: function (id, enabled, opts) {
      return request('POST', '/ptz/' + encodeURIComponent(id) + '/track',
        Object.assign({ body: { enabled: !!enabled } }, opts || {}));
    },
    returnDelay: function (id, seconds, opts) {
      return request('POST', '/ptz/' + encodeURIComponent(id) + '/return_delay',
        Object.assign({ body: { delay: seconds } }, opts || {}));
    },
    debug: function (opts) { return request('GET', '/ptz/debug', opts); },
    setDebug: function (enabled, opts) {
      return request('POST', '/ptz/debug',
        Object.assign({ body: { enabled: !!enabled } }, opts || {}));
    }
  },

  /** Escape hatch for anything not yet wrapped. Same error contract. */
  request: request
};
