/* ============================================================================
   core/router.js — history routing for the /app shell.

   The old UI's query string is the only thing users could bookmark, so the
   URL is the state: /app/recordings?view=month&year=2026&month=9&cameras=cam1
   round-trips exactly, Back and Forward walk months and filter sets, and
   nothing in this app ever calls location.reload().

   ROUTE TABLE
     router.register('/recordings', view)          exact + child paths
     router.register('/clips/:path*', view)        ':name' = one segment,
                                                   ':name*' = the rest, slashes
                                                   included and decoded
   A view is { mount(root, ctx), unmount(), update(ctx)? }.

   ctx = {
     path:   '/recordings',           the matched app path (no /app prefix)
     params: { path: 'cam1/…mp4' },   decoded route params
     query:  { view:'month', … },     decoded query string, always an object
     url:    '/app/recordings?…',     the full URL
     first:  true                     first mount of this view instance
   }

   WHEN THE URL CHANGES
     · different route      -> unmount() the old view, mount() the new one
     · same route, new query-> update(ctx) if the view exports it, else
                               unmount()+mount(). update() is what lets a
                               filter change keep scroll position and focus.
   Views must therefore treat update() as "re-read ctx and diff".

   PUBLIC API
     router.register(pattern, view)
     router.start(root)                 begins routing, mounts the first view
     router.navigate(url, opts)         opts { replace, state }
     router.setQuery(patch, opts)       merge/remove params on the current URL
                                        (null or '' removes a key)
     router.go(path, query, opts)       navigate to an app path with a query
     router.href(path, query)           build a URL without navigating
     router.current                     the live ctx
     router.subscribe(fn)               fn(ctx) after every navigation
     router.back()
   ========================================================================= */

var BASE = '/app';

function parseQuery(search) {
  var out = {};
  var s = String(search || '');
  if (s.charAt(0) === '?') s = s.slice(1);
  if (!s) return out;
  var parts = s.split('&');
  for (var i = 0; i < parts.length; i++) {
    if (!parts[i]) continue;
    var eq = parts[i].indexOf('=');
    var k = eq < 0 ? parts[i] : parts[i].slice(0, eq);
    var v = eq < 0 ? '' : parts[i].slice(eq + 1);
    try { k = decodeURIComponent(k.replace(/\+/g, ' ')); } catch (e) {}
    try { v = decodeURIComponent(v.replace(/\+/g, ' ')); } catch (e2) {}
    out[k] = v;
  }
  return out;
}

function buildQuery(obj) {
  var parts = [];
  for (var k in obj) {
    if (!Object.prototype.hasOwnProperty.call(obj, k)) continue;
    var v = obj[k];
    if (v === null || v === undefined || v === '') continue;
    parts.push(encodeURIComponent(k) + '=' + encodeURIComponent(String(v)));
  }
  return parts.length ? '?' + parts.join('&') : '';
}

export { parseQuery, buildQuery };

function compile(pattern) {
  var segs = pattern.split('/').filter(Boolean);
  var names = [];
  var src = '';
  for (var i = 0; i < segs.length; i++) {
    var seg = segs[i];
    if (seg.charAt(0) === ':') {
      var greedy = seg.charAt(seg.length - 1) === '*';
      names.push(greedy ? seg.slice(1, -1) : seg.slice(1));
      src += greedy ? '/(.+)' : '/([^/]+)';
    } else {
      src += '/' + seg.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
    }
  }
  if (!src) src = '/';
  return { re: new RegExp('^' + src + '/?$'), names: names, pattern: pattern };
}

function appPath(pathname) {
  var p = String(pathname || '/');
  if (p.indexOf(BASE) === 0) p = p.slice(BASE.length);
  if (!p) p = '/';
  if (p.charAt(0) !== '/') p = '/' + p;
  return p;
}

function createRouter() {
  var routes = [];
  var fallback = null;
  var root = null;
  var current = null;          /* ctx */
  var mounted = null;          /* { view, route } */
  var subs = [];

  function match(path) {
    for (var i = 0; i < routes.length; i++) {
      var m = routes[i].re.exec(path);
      if (!m) continue;
      var params = {};
      for (var n = 0; n < routes[i].names.length; n++) {
        var raw = m[n + 1] || '';
        try { raw = decodeURIComponent(raw); } catch (e) {}
        params[routes[i].names[n]] = raw;
      }
      return { route: routes[i], params: params };
    }
    return null;
  }

  function contextFor() {
    var path = appPath(window.location.pathname);
    var found = match(path);
    return {
      path: path,
      matched: found ? found.route.pattern : null,
      params: found ? found.params : {},
      query: parseQuery(window.location.search),
      url: window.location.pathname + window.location.search,
      view: found ? found.route.view : (fallback && fallback.view)
    };
  }

  function notify(ctx) {
    var list = subs.slice();
    for (var i = 0; i < list.length; i++) {
      try { list[i](ctx); } catch (err) {
        if (window.console) console.error('[router] subscriber failed', err);
      }
    }
  }

  function resolve() {
    var ctx = contextFor();
    var prev = current;
    current = ctx;
    api.current = ctx;

    var sameRoute = mounted && prev && prev.matched === ctx.matched && mounted.view === ctx.view;
    /* A route param change (a different clip) is a different page even though
       the pattern matched — force a remount unless the view opts out. */
    if (sameRoute && prev) {
      for (var k in ctx.params) {
        if (ctx.params[k] !== prev.params[k]) { sameRoute = false; break; }
      }
    }

    if (sameRoute && mounted.view.update) {
      ctx.first = false;
      try {
        mounted.view.update(ctx);
      } catch (err) {
        if (window.console) console.error('[router] update failed', err);
      }
      notify(ctx);
      return;
    }

    if (mounted) {
      try { mounted.view.unmount(); } catch (err2) {
        if (window.console) console.error('[router] unmount failed', err2);
      }
      mounted = null;
    }
    while (root && root.firstChild) root.removeChild(root.firstChild);

    if (!ctx.view) {
      notify(ctx);
      return;
    }
    ctx.first = true;
    mounted = { view: ctx.view, route: ctx.matched };
    try {
      ctx.view.mount(root, ctx);
    } catch (err3) {
      if (window.console) console.error('[router] mount failed', err3);
      mounted = null;
    }
    notify(ctx);
  }

  var api = {
    current: null,

    register: function (pattern, view) {
      var compiled = compile(pattern);
      compiled.view = view;
      routes.push(compiled);
      return api;
    },

    /** The view used when nothing matches (a 404 inside the app). */
    setFallback: function (view) { fallback = { view: view }; return api; },

    start: function (mountRoot) {
      root = mountRoot;
      window.addEventListener('popstate', function () { resolve(); });
      resolve();
      return api;
    },

    subscribe: function (fn) {
      subs.push(fn);
      return function () {
        var i = subs.indexOf(fn);
        if (i >= 0) subs.splice(i, 1);
      };
    },

    href: function (path, query) {
      var p = path.charAt(0) === '/' ? path : '/' + path;
      return BASE + (p === '/' ? '' : p) + buildQuery(query || {});
    },

    navigate: function (url, opts) {
      var o = opts || {};
      var target = String(url);
      var here = window.location.pathname + window.location.search;
      if (target === here && !o.force) return;
      if (o.replace) window.history.replaceState(o.state || null, '', target);
      else window.history.pushState(o.state || null, '', target);
      resolve();
    },

    go: function (path, query, opts) {
      api.navigate(api.href(path, query), opts);
    },

    /**
     * Merge `patch` into the current query string. A null / undefined / ''
     * value removes the key. This is how every filter, sort and view-mode
     * change is written, so all of them are bookmarkable and undo with Back.
     */
    setQuery: function (patch, opts) {
      var next = Object.assign({}, current ? current.query : {});
      for (var k in patch) {
        if (!Object.prototype.hasOwnProperty.call(patch, k)) continue;
        var v = patch[k];
        if (v === null || v === undefined || v === '') delete next[k];
        else next[k] = String(v);
      }
      api.navigate((current ? BASE + current.path : window.location.pathname) + buildQuery(next), opts);
    },

    back: function () { window.history.back(); },

    /** Intercept in-app anchors so a normal <a href> does not reload. */
    interceptLinks: function (container) {
      container.addEventListener('click', function (ev) {
        if (ev.defaultPrevented || ev.button !== 0) return;
        if (ev.metaKey || ev.ctrlKey || ev.shiftKey || ev.altKey) return;
        var node = ev.target;
        while (node && node !== container) {
          if (node.tagName === 'A') break;
          node = node.parentElement;
        }
        if (!node || node.tagName !== 'A') return;
        if (node.target || node.hasAttribute('download')) return;
        var href = node.getAttribute('href') || '';
        if (!href || href.charAt(0) === '#') return;
        if (href.indexOf(BASE) !== 0) return;
        ev.preventDefault();
        api.navigate(href);
      });
    },

    base: BASE
  };

  return api;
}

export var router = createRouter();
