/* ============================================================================
   core/dom.js — element construction and keyed list reconciliation.

   Everything the app puts on screen goes through here. The two rules this
   module exists to enforce:

     1. Model data NEVER reaches innerHTML. h() writes every string through
        textContent or setAttribute, so a species name containing < or " is
        text, not markup.
     2. Lists are reconciled, not replaced. keyedList() patches in place, so a
        refresh keeps decoded images, scroll position, focus and selection.

   iOS 15 floor: no optional chaining on the left of assignment, no ??=, no
   .at(), no structuredClone, no top-level await.

   PUBLIC API
     h(sel, props?, ...children)  -> Element
     svg(sel, props?, ...children)-> SVGElement (namespaced)
     frag(...children)            -> DocumentFragment
     clear(node)                  -> node          (removes all children)
     mount(parent, ...children)   -> parent        (clear + append)
     on(target, type, fn, opts?)  -> off()
     delegate(root, type, sel, fn, opts?) -> off()
     keyedList(container, items, spec) -> { nodes: Map, order: [] }
   ========================================================================= */

/* Tag selector: "div", "li.clip.sp--deer", "a.clip__open", "input#sel-3". */
var SEL_RE = /([.#]?[^\s.#]+)/g;

function parseSel(sel) {
  var tag = 'div';
  var classes = [];
  var id = null;
  var m;
  SEL_RE.lastIndex = 0;
  while ((m = SEL_RE.exec(sel)) !== null) {
    var tok = m[1];
    if (tok.charAt(0) === '.') classes.push(tok.slice(1));
    else if (tok.charAt(0) === '#') id = tok.slice(1);
    else tag = tok;
  }
  return { tag: tag, classes: classes, id: id };
}

/* Attributes that must be set as DOM properties, not attributes, so that the
   live value (not the initial one) is what changes. */
var PROPS = {
  value: 1, checked: 1, indeterminate: 1, selected: 1, disabled: 1,
  muted: 1, playsInline: 1, autoplay: 1, loop: 1, controls: 1, srcObject: 1,
  tabIndex: 1, textContent: 1, scrollTop: 1, scrollLeft: 1
};

function applyClass(el, value) {
  if (!value) return;
  if (typeof value === 'string') {
    var parts = value.split(/\s+/);
    for (var i = 0; i < parts.length; i++) if (parts[i]) el.classList.add(parts[i]);
    return;
  }
  if (Array.isArray(value)) {
    for (var j = 0; j < value.length; j++) applyClass(el, value[j]);
    return;
  }
  for (var k in value) if (value[k]) applyClass(el, k);
}

function applyProps(el, props) {
  if (!props) return;
  for (var key in props) {
    if (!Object.prototype.hasOwnProperty.call(props, key)) continue;
    var v = props[key];
    if (v === null || v === undefined || v === false) {
      /* false removes a boolean attribute; null/undefined is "not set". */
      if (v === false && !PROPS[key]) el.removeAttribute(key);
      else if (PROPS[key]) el[key] = v === null ? '' : v;
      continue;
    }
    if (key === 'class' || key === 'className') { applyClass(el, v); continue; }
    if (key === 'text') { el.textContent = String(v); continue; }
    if (key === 'style') {
      if (typeof v === 'string') { el.setAttribute('style', v); continue; }
      for (var sk in v) {
        if (sk.charAt(0) === '-') el.style.setProperty(sk, String(v[sk]));
        else el.style[sk] = v[sk];
      }
      continue;
    }
    if (key === 'dataset') { for (var dk in v) el.dataset[dk] = String(v[dk]); continue; }
    if (key === 'aria') {
      for (var ak in v) {
        var av = v[ak];
        if (av === null || av === undefined) el.removeAttribute('aria-' + ak);
        else el.setAttribute('aria-' + ak, String(av));
      }
      continue;
    }
    if (key === 'on') { for (var ek in v) el.addEventListener(ek, v[ek]); continue; }
    if (key.length > 2 && key.slice(0, 2) === 'on' && typeof v === 'function') {
      el.addEventListener(key.slice(2).toLowerCase(), v);
      continue;
    }
    if (PROPS[key]) { el[key] = v; continue; }
    if (v === true) { el.setAttribute(key, ''); continue; }
    el.setAttribute(key, String(v));
  }
}

function appendChild(parent, child) {
  if (child === null || child === undefined || child === false || child === true) return;
  if (Array.isArray(child)) {
    for (var i = 0; i < child.length; i++) appendChild(parent, child[i]);
    return;
  }
  if (child.nodeType) { parent.appendChild(child); return; }
  parent.appendChild(document.createTextNode(String(child)));
}

/**
 * Build an element.
 *   h('li.clip', { 'data-path': p }, h('h3.clip__title', { text: species }))
 * Strings passed as children become TEXT NODES — never parsed as HTML.
 */
export function h(sel, props) {
  var parsed = parseSel(sel);
  var el = document.createElement(parsed.tag);
  if (parsed.id) el.id = parsed.id;
  for (var i = 0; i < parsed.classes.length; i++) el.classList.add(parsed.classes[i]);
  var start = 1;
  if (props && typeof props === 'object' && !props.nodeType && !Array.isArray(props)) {
    applyProps(el, props);
    start = 2;
  }
  for (var a = start; a < arguments.length; a++) appendChild(el, arguments[a]);
  return el;
}

var SVG_NS = 'http://www.w3.org/2000/svg';
var XLINK_NS = 'http://www.w3.org/1999/xlink';

/** SVG counterpart of h(). Needed for the icon sprite and the gauge spark. */
export function svg(sel, props) {
  var parsed = parseSel(sel);
  var el = document.createElementNS(SVG_NS, parsed.tag);
  for (var i = 0; i < parsed.classes.length; i++) el.classList.add(parsed.classes[i]);
  if (parsed.id) el.setAttribute('id', parsed.id);
  var start = 1;
  if (props && typeof props === 'object' && !props.nodeType && !Array.isArray(props)) {
    for (var key in props) {
      var v = props[key];
      if (v === null || v === undefined || v === false) continue;
      if (key === 'class' || key === 'className') { applyClass(el, v); continue; }
      if (key === 'text') { el.textContent = String(v); continue; }
      if (key === 'on') { for (var ek in v) el.addEventListener(ek, v[ek]); continue; }
      /* Safari 15 honours href on <use>; xlink:href is the belt-and-braces. */
      if (key === 'href') {
        el.setAttribute('href', String(v));
        el.setAttributeNS(XLINK_NS, 'xlink:href', String(v));
        continue;
      }
      el.setAttribute(key, String(v));
    }
    start = 2;
  }
  for (var a = start; a < arguments.length; a++) appendChild(el, arguments[a]);
  return el;
}

export function frag() {
  var f = document.createDocumentFragment();
  for (var i = 0; i < arguments.length; i++) appendChild(f, arguments[i]);
  return f;
}

export function clear(node) {
  while (node && node.firstChild) node.removeChild(node.firstChild);
  return node;
}

export function mount(parent) {
  clear(parent);
  for (var i = 1; i < arguments.length; i++) appendChild(parent, arguments[i]);
  return parent;
}

/**
 * addEventListener with a disposer. Every listener the app attaches goes
 * through here or delegate(), so a view's unmount() can release all of them.
 */
export function on(target, type, fn, opts) {
  target.addEventListener(type, fn, opts);
  var released = false;
  return function off() {
    if (released) return;
    released = true;
    target.removeEventListener(type, fn, opts);
  };
}

/**
 * One listener for a whole list. The handler is called with
 * (event, matchedElement); `this` is not used.
 */
export function delegate(root, type, selector, fn, opts) {
  function handler(ev) {
    var node = ev.target;
    if (!node || node.nodeType !== 1) node = node && node.parentElement;
    while (node && node !== root) {
      if (node.matches && node.matches(selector)) { fn(ev, node); return; }
      node = node.parentElement;
    }
  }
  root.addEventListener(type, handler, opts);
  var released = false;
  return function off() {
    if (released) return;
    released = true;
    root.removeEventListener(type, handler, opts);
  };
}

/**
 * Keyed list reconciliation.
 *
 *   keyedList(ul, clips, {
 *     key:    function (clip) { return clip.path },
 *     create: function (clip, key) { return buildShell(clip) },
 *     update: function (el, clip, key, index) { patchCard(el, clip) },
 *     remove: function (el, key) { el.remove() }        // optional
 *   })
 *
 * update() runs for EVERY item, including one that was just created — so a
 * caller writes the "put the data into the node" logic exactly once and
 * create() only has to build empty structure.
 *
 * Nodes are matched by their `data-key` attribute, so a node that survives a
 * refresh keeps its decoded <img>, its focus and its checked state. Nodes are
 * moved only when they are actually out of order.
 *
 * Returns { nodes: Map<key, Element>, order: [key] }.
 */
export function keyedList(container, items, spec) {
  var keyOf = spec.key;
  var existing = new Map();
  var child = container.firstElementChild;
  while (child) {
    var k = child.getAttribute('data-key');
    if (k !== null) existing.set(k, child);
    child = child.nextElementSibling;
  }

  var nodes = new Map();
  var order = [];
  var cursor = container.firstElementChild;

  for (var i = 0; i < items.length; i++) {
    var item = items[i];
    var key = String(keyOf(item, i));
    var el = existing.get(key);
    if (el) {
      existing.delete(key);
    } else {
      el = spec.create(item, key, i);
      el.setAttribute('data-key', key);
    }
    if (spec.update) spec.update(el, item, key, i);
    if (cursor === el) {
      cursor = cursor.nextElementSibling;
    } else {
      container.insertBefore(el, cursor);
    }
    nodes.set(key, el);
    order.push(key);
  }

  existing.forEach(function (el, key) {
    if (spec.remove) spec.remove(el, key);
    else if (el.parentNode) el.parentNode.removeChild(el);
  });

  return { nodes: nodes, order: order };
}
