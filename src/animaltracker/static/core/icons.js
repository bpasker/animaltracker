/* ============================================================================
   core/icons.js — the inline SVG sprite.

   No emoji does load-bearing icon work anywhere in this app, and no icon
   costs a network request. One <svg class="icon-sprite"> of <symbol>s is
   injected into <body> at boot; every icon on screen is a 20px <use>.

   PUBLIC API
     icon(name, opts?)  -> SVGElement   opts: { size:'sm'|'lg', label, class }
     installSprite()    -> void         idempotent; app.js calls it once
     ICON_NAMES         -> [string]

   An icon that carries meaning takes a `label` (it becomes role="img" plus a
   <title>); everything else is aria-hidden, because the text beside it is the
   accessible name.
   ========================================================================= */

import { svg } from './dom.js';

var NS = 'http://www.w3.org/2000/svg';

/* Stroke geometry on a 24 grid, stroke-width 1.6, round caps — the values
   live in app.css (.i / .i-sm / .i-lg) so nothing here sets presentation. */
var PATHS = {
  'search':      ['M11 4a7 7 0 1 0 0 14 7 7 0 0 0 0-14z', 'M20 20l-4-4'],
  'x':           ['M6 6l12 12', 'M18 6L6 18'],
  'check':       ['M5 13l4 4L19 7'],
  'chevron-left':  ['M15 5l-7 7 7 7'],
  'chevron-right': ['M9 5l7 7-7 7'],
  'chevron-down':  ['M5 9l7 7 7-7'],
  'chevron-up':    ['M5 15l7-7 7 7'],
  'arrow-left':  ['M20 12H4', 'M10 6l-6 6 6 6'],
  'arrow-down':  ['M12 4v16', 'M6 14l6 6 6-6'],
  'grid':        ['M4 4h7v7H4z', 'M13 4h7v7h-7z', 'M4 13h7v7H4z', 'M13 13h7v7h-7z'],
  'calendar':    ['M4 6h16v14H4z', 'M4 10h16', 'M8 3v4', 'M16 3v4'],
  'list':        ['M8 6h12', 'M8 12h12', 'M8 18h12', 'M4 6h.01', 'M4 12h.01', 'M4 18h.01'],
  'filter':      ['M3 5h18l-7 8v6l-4 2v-8z'],
  'sort':        ['M7 4v16', 'M4 8l3-4 3 4', 'M17 20V4', 'M14 16l3 4 3-4'],
  'play':        ['M8 5l11 7-11 7z'],
  'trash':       ['M4 7h16', 'M9 7V4h6v3', 'M6 7l1 13h10l1-13', 'M10 11v6', 'M14 11v6'],
  'camera':      ['M3 7h4l2-3h6l2 3h4v13H3z', 'M12 16a3.5 3.5 0 1 0 0-7 3.5 3.5 0 0 0 0 7z'],
  'film':        ['M3 5h18v14H3z', 'M3 9h18', 'M3 15h18', 'M8 5v14', 'M16 5v14'],
  'monitor':     ['M3 5h18v11H3z', 'M9 20h6', 'M12 16v4'],
  'settings':    ['M12 15a3 3 0 1 0 0-6 3 3 0 0 0 0 6z',
                  'M4.5 12a7.5 7.5 0 0 1 .1-1.2l-1.8-1.4 2-3.4 2.1.8a7.5 7.5 0 0 1 2.1-1.2L9.4 3h4.2l.4 2.2c.8.3 1.5.7 2.1 1.2l2.1-.8 2 3.4-1.8 1.4a7.5 7.5 0 0 1 0 2.4l1.8 1.4-2 3.4-2.1-.8a7.5 7.5 0 0 1-2.1 1.2l-.4 2.2H9.4L9 18.8a7.5 7.5 0 0 1-2.1-1.2l-2.1.8-2-3.4 1.8-1.4A7.5 7.5 0 0 1 4.5 12z'],
  'live':        ['M12 15a3 3 0 1 0 0-6 3 3 0 0 0 0 6z', 'M6.5 6.5a8 8 0 0 0 0 11', 'M17.5 17.5a8 8 0 0 0 0-11'],
  'refresh':     ['M20 12a8 8 0 1 1-2.6-5.9', 'M20 4v5h-5'],
  'alert':       ['M12 3l9 16H3z', 'M12 10v4', 'M12 17h.01'],
  'info':        ['M12 21a9 9 0 1 0 0-18 9 9 0 0 0 0 18z', 'M12 11v5', 'M12 8h.01'],
  'undo':        ['M4 9h11a5 5 0 0 1 0 10h-6', 'M8 5L4 9l4 4'],
  'download':    ['M12 4v11', 'M7 11l5 5 5-5', 'M4 20h16'],
  'more':        ['M6 12h.01', 'M12 12h.01', 'M18 12h.01'],
  'plus':        ['M12 5v14', 'M5 12h14'],
  'minus':       ['M5 12h14'],
  'command':     ['M9 6a3 3 0 1 0-3 3h12a3 3 0 1 0-3-3v12a3 3 0 1 0 3-3H6a3 3 0 1 0 3 3z'],
  'sun':         ['M12 16a4 4 0 1 0 0-8 4 4 0 0 0 0 8z', 'M12 2v2', 'M12 20v2',
                  'M4.9 4.9l1.4 1.4', 'M17.7 17.7l1.4 1.4', 'M2 12h2', 'M20 12h2',
                  'M4.9 19.1l1.4-1.4', 'M17.7 6.3l1.4-1.4'],
  'moon':        ['M20 14.5A8.5 8.5 0 0 1 9.5 4a8.5 8.5 0 1 0 10.5 10.5z'],
  'auto':        ['M4 5h16v11H4z', 'M8 20h8', 'M12 16v4', 'M12 5v11'],
  'bookmark':    ['M6 4h12v16l-6-4-6 4z'],
  'select':      ['M4 6h4', 'M4 12h4', 'M4 18h4', 'M11 9l2 2 5-5', 'M11 17h9'],
  'check-square':['M4 4h16v16H4z', 'M8 12l3 3 5-6'],
  'square':      ['M4 4h16v16H4z'],
  'image-off':   ['M4 5h16v14H4z', 'M4 16l4.5-4.5 3 3L15 11l5 5', 'M4 4l16 16'],
  'clock':       ['M12 21a9 9 0 1 0 0-18 9 9 0 0 0 0 18z', 'M12 7v5l3 2'],
  'moon-phase':  ['M20 14.5A8.5 8.5 0 0 1 9.5 4a8.5 8.5 0 1 0 10.5 10.5z'],
  'sunrise':     ['M12 4v5', 'M8 8l4-4 4 4', 'M3 17h18', 'M6 13a6 6 0 0 1 12 0'],
  'sunset':      ['M12 9V4', 'M8 5l4 4 4-4', 'M3 17h18', 'M6 13a6 6 0 0 1 12 0'],
  'sparkle':     ['M12 3l1.9 5.1L19 10l-5.1 1.9L12 17l-1.9-5.1L5 10l5.1-1.9z'],
  'layers':      ['M12 3l9 5-9 5-9-5z', 'M3 13l9 5 9-5'],
  'disk':        ['M12 21a9 9 0 1 0 0-18 9 9 0 0 0 0 18z', 'M12 14a2 2 0 1 0 0-4 2 2 0 0 0 0 4z'],
  'external':    ['M14 4h6v6', 'M20 4l-9 9', 'M18 14v6H4V6h6']
};

export var ICON_NAMES = Object.keys(PATHS);

var installed = false;

/** Inject the sprite once. Safe to call repeatedly. */
export function installSprite() {
  if (installed || document.getElementById('at-icon-sprite')) { installed = true; return; }
  var sprite = document.createElementNS(NS, 'svg');
  sprite.setAttribute('id', 'at-icon-sprite');
  sprite.setAttribute('aria-hidden', 'true');
  sprite.setAttribute('focusable', 'false');
  sprite.setAttribute('class', 'icon-sprite');

  for (var i = 0; i < ICON_NAMES.length; i++) {
    var name = ICON_NAMES[i];
    var sym = document.createElementNS(NS, 'symbol');
    sym.setAttribute('id', 'i-' + name);
    sym.setAttribute('viewBox', '0 0 24 24');
    var d = PATHS[name];
    for (var p = 0; p < d.length; p++) {
      var path = document.createElementNS(NS, 'path');
      path.setAttribute('d', d[p]);
      sym.appendChild(path);
    }
    sprite.appendChild(sym);
  }
  document.body.insertBefore(sprite, document.body.firstChild);
  installed = true;
}

/**
 * One icon.
 *   icon('trash')                        decorative, 20px, aria-hidden
 *   icon('alert', { size:'lg' })         24px
 *   icon('live', { label:'Live now' })   meaningful: role=img + <title>
 */
export function icon(name, opts) {
  var o = opts || {};
  var cls = 'i';
  if (o.size === 'sm') cls += ' i-sm';
  else if (o.size === 'lg') cls += ' i-lg';
  if (o.class) cls += ' ' + o.class;

  var el = svg('svg', {
    'class': cls,
    viewBox: '0 0 24 24',
    focusable: 'false'
  });
  if (o.label) {
    el.setAttribute('role', 'img');
    var title = document.createElementNS(NS, 'title');
    title.textContent = o.label;
    el.appendChild(title);
  } else {
    el.setAttribute('aria-hidden', 'true');
  }
  el.appendChild(svg('use', { href: '#i-' + name }));
  return el;
}
