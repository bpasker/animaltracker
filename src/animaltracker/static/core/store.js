/* ============================================================================
   core/store.js — the app's shared state, and the factory views use for their
   own local state.

   The store is deliberately dumb: a plain object, a shallow merge, and a
   synchronous notify carrying the list of keys that actually changed. There is
   no reactivity magic and no proxy — a view subscribes to the keys it cares
   about and patches its own DOM.

   THE CHROME SLICE
   A view never reaches into the app bar, the rail or the tab bar directly.
   It DECLARES its chrome and app.js renders it:

     store.setChrome({
       title:    'Recordings',
       subtitle: '250 clips · 11 species · 11.4 GB',
       rail:     railElement,      // DOM node, or null for no rail
       toolbar:  chipStripElement, // rides under the mobile top bar
       actions:  [buttonEl, ...],  // trailing controls in the mobile top bar
       selbar:   selbarElement,    // floating selection bar, or null
       mods:     ['toolbar']       // shell modifiers, without the 'shell--'
     })

   Anything omitted from a setChrome() patch is left alone; app.js resets the
   whole slice between views, so a view only declares what it needs.

   PUBLIC API
     store.state                  the live state object (read, never mutate)
     store.get(key)
     store.set(patch)             shallow merge + notify
     store.setChrome(patch)       merge into state.chrome + notify ['chrome']
     store.subscribe(fn)          fn(state, changedKeys) -> unsubscribe()
     store.select(keys, fn)       notified only when one of `keys` changed
     createStore(initial)         same API, for a view's local state
   ========================================================================= */

function shallowEqual(a, b) {
  if (a === b) return true;
  if (a instanceof Set && b instanceof Set) {
    if (a.size !== b.size) return false;
    var same = true;
    a.forEach(function (v) { if (!b.has(v)) same = false; });
    return same;
  }
  return false;
}

export function createStore(initial) {
  var state = Object.assign({}, initial || {});
  var subs = [];

  function notify(keys) {
    if (!keys.length) return;
    /* Copy first: a subscriber may unsubscribe during the walk. */
    var list = subs.slice();
    for (var i = 0; i < list.length; i++) {
      try {
        list[i](state, keys);
      } catch (err) {
        /* One broken subscriber must not stop the rest of the app. */
        if (window.console) console.error('[store] subscriber failed', err);
      }
    }
  }

  var api = {
    state: state,

    get: function (key) { return state[key]; },

    set: function (patch) {
      var changed = [];
      for (var k in patch) {
        if (!Object.prototype.hasOwnProperty.call(patch, k)) continue;
        if (!shallowEqual(state[k], patch[k])) {
          state[k] = patch[k];
          changed.push(k);
        }
      }
      notify(changed);
      return api;
    },

    /** Force a notify even when the reference did not change (mutated array). */
    touch: function () {
      notify(Array.prototype.slice.call(arguments));
      return api;
    },

    setChrome: function (patch) {
      var next = Object.assign({}, state.chrome || {}, patch);
      state.chrome = next;
      notify(['chrome']);
      return api;
    },

    subscribe: function (fn) {
      subs.push(fn);
      var live = true;
      return function unsubscribe() {
        if (!live) return;
        live = false;
        var i = subs.indexOf(fn);
        if (i >= 0) subs.splice(i, 1);
      };
    },

    select: function (keys, fn) {
      var want = Array.isArray(keys) ? keys : [keys];
      return api.subscribe(function (s, changed) {
        for (var i = 0; i < changed.length; i++) {
          if (want.indexOf(changed[i]) >= 0) { fn(s, changed); return; }
        }
      });
    }
  };

  return api;
}

/* --- The application store -------------------------------------------------
   Keys owned here:
     theme        'light' | 'dark' | 'auto'
     density      'comfortable' | 'compact'
     railCollapsed boolean, persisted
     cameras      [] from /api/cameras
     timezone     server timezone label
     camerasError Error | null — the rail renders a diagnostic row from it
     connected    false once a fetch has failed; the app bar says so
     route        { path, params }
     chrome       see above
     savedViews   [{ id, name, query, count }] — persisted in localStorage
     visible      document visibility, so pollers can consult it
   ------------------------------------------------------------------------- */
export var store = createStore({
  theme: 'auto',
  density: 'comfortable',
  railCollapsed: false,
  cameras: [],
  timezone: '',
  camerasError: null,
  connected: true,
  route: null,
  visible: true,
  savedViews: [],
  chrome: {}
});

/* --- localStorage helpers --------------------------------------------------
   Storage throws in a private window on iOS; every access is guarded and the
   app degrades to defaults rather than failing to boot. */
export function readLocal(key, fallback) {
  try {
    var raw = window.localStorage.getItem(key);
    if (raw === null) return fallback;
    return JSON.parse(raw);
  } catch (e) {
    return fallback;
  }
}

export function writeLocal(key, value) {
  try {
    window.localStorage.setItem(key, JSON.stringify(value));
    return true;
  } catch (e) {
    return false;
  }
}
