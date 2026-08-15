/* my walk of life, reveals, dark mode, and four canvas visualisations.
   Drawn by hand on canvas in a p5-ish idiom: sketchy strokes, generative jitter,
   animated entrances. Every number and every joint angle below is measured output
   read from /data, nothing here is illustrative. */

const css = v => getComputedStyle(document.body).getPropertyValue(v).trim();
const MINE = () => css('--mine');
const THEM = () => css('--them');

/* ── dark mode ──────────────────────────────────────────────── */
const root = document.documentElement, btn = document.getElementById('mode');
const stored = localStorage.getItem('theme');
if (stored) root.setAttribute('data-theme', stored);
const isDark = () => root.getAttribute('data-theme') === 'dark' ||
  (!root.getAttribute('data-theme') && matchMedia('(prefers-color-scheme:dark)').matches);
const paintBtn = () => {
  btn.classList.toggle('is-dark', isDark());
  btn.setAttribute('aria-pressed', String(isDark()));
};
paintBtn();
btn.addEventListener('click', () => {
  root.setAttribute('data-theme', isDark() ? 'light' : 'dark');
  localStorage.setItem('theme', root.getAttribute('data-theme'));
  paintBtn();
  requestAnimationFrame(() => charts.forEach(c => c.draw(1)));
});

/* ── scroll reveals ─────────────────────────────────────────── */
const io = new IntersectionObserver(es => es.forEach(e => {
  if (e.isIntersecting) { e.target.classList.add('in'); io.unobserve(e.target); }
}), { threshold: 0.12, rootMargin: '0px 0px -8% 0px' });
document.querySelectorAll('.reveal').forEach(el => io.observe(el));
/* the photo, its arrow and its label share one trigger so they play as a sequence */
document.querySelectorAll('.portrait-stage').forEach(el => io.observe(el));

/* ── the rig: not a trigger but a scrub, the scrollbar is the timeline ──
   --p runs 0→1 as the figure rises into the frame (photo, then arrow, then
   label, sequenced in CSS); --q runs 0→1 across its whole traversal of the
   viewport and only drives a few pixels of parallax. */
const rig = document.querySelector('.rig-solo');
if (rig && !matchMedia('(prefers-reduced-motion:reduce)').matches) {
  const clamp01 = v => v < 0 ? 0 : v > 1 ? 1 : v;
  let queued = false;
  const track = () => {
    queued = false;
    const r = rig.getBoundingClientRect(), vh = innerHeight || 1;
    rig.style.setProperty('--p', clamp01((vh - r.top) / (vh * .58)).toFixed(4));
    rig.style.setProperty('--q', clamp01((vh - r.top) / (vh + r.height)).toFixed(4));
  };
  const onScroll = () => { if (!queued) { queued = true; requestAnimationFrame(track); } };
  addEventListener('scroll', onScroll, { passive: true });
  addEventListener('resize', onScroll);
  track();
}

/* ── canvas helper: HiDPI, animate once when scrolled into view ── */
const charts = [];
function makeChart(canvas, render, ratio = 2, ms = 1100) {
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  function size() {
    const dpr = Math.min(devicePixelRatio || 1, 2);
    const r = canvas.getBoundingClientRect();
    const w = r.width, h = r.height || r.width / ratio;
    canvas.width = w * dpr; canvas.height = h * dpr;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    return [w, h];
  }
  const obj = { draw(t) { const [w, h] = size(); ctx.clearRect(0, 0, w, h); render(ctx, w, h, t); } };
  charts.push(obj);
  /* no entrance animation when the viewer has asked for less motion: draw it finished */
  if (matchMedia('(prefers-reduced-motion: reduce)').matches) {
    new IntersectionObserver(es => es.forEach(e => {
      if (e.isIntersecting) obj.draw(1);
    }), { threshold: 0.25 }).observe(canvas);
    addEventListener('resize', () => obj.draw(1));
    obj.draw(1);
    return;
  }
  let started = false;
  new IntersectionObserver(es => es.forEach(e => {
    if (!e.isIntersecting || started) return;
    started = true;
    const t0 = performance.now();
    (function frame(now) {
      const raw = Math.min((now - t0) / ms, 1);
      obj.draw(raw < 1 ? raw * raw * (3 - 2 * raw) : 1);
      if (raw < 1) requestAnimationFrame(frame);
    })(t0);
  }), { threshold: 0.25 }).observe(canvas);
  addEventListener('resize', () => obj.draw(1));
}

/* deterministic jitter, so strokes read as drawn rather than plotted */
const wobble = i => ((Math.sin(i * 12.9898) * 43758.5453) % 1);

/* Every mark in the two comparison charts is a sole, not a dot, the page is
   about a shoe wearing out, so the unit of data may as well be a footprint.
   Round toe at the top, the ball of the foot at its widest, a waist just past
   halfway, then a smaller round heel, the right side drawn out and the left
   drawn back. `right` mirrors it into the other foot, and a few degrees of splay
   per mark keep them stamped rather than plotted. */
function sole(ctx, x, y, s, ang, right, fill) {
  /* a real sole is about 2.4 times longer than it is wide, and its waist only
     pinches in by a quarter, any more and it reads as an hourglass */
  const L = s / 2, W = s * .21;
  ctx.save();
  ctx.translate(x, y); ctx.rotate(ang); if (!right) ctx.scale(-1, 1);
  ctx.beginPath();
  ctx.moveTo(0, -L);
  /* outer edge: near straight from the ball down to a narrower heel */
  ctx.bezierCurveTo(W * .82, -L * .98, W * .99, -L * .85, W * .97, -L * .52);
  ctx.bezierCurveTo(W * .95, -L * .15, W * .88, L * .28, W * .80, L * .66);
  ctx.bezierCurveTo(W * .74, L * .95, W * .42, L, 0, L);
  /* inner edge: the arch bites in, which is what makes it read as a foot */
  ctx.bezierCurveTo(-W * .42, L, -W * .72, L * .95, -W * .76, L * .66);
  ctx.bezierCurveTo(-W * .80, L * .34, -W * .42, L * .22, -W * .45, -L * .05);
  ctx.bezierCurveTo(-W * .48, -L * .32, -W * .90, -L * .42, -W * .95, -L * .60);
  ctx.bezierCurveTo(-W * 1.0, -L * .82, -W * .80, -L * .98, 0, -L);
  ctx.closePath();
  fill ? ctx.fill() : ctx.stroke();
  ctx.restore();
}
function sketch(ctx, pts, amt = 0.8) {
  ctx.beginPath();
  pts.forEach((p, i) => {
    const j = wobble(i) * amt;
    i ? ctx.lineTo(p[0] + j, p[1] - j) : ctx.moveTo(p[0], p[1]);
  });
  ctx.stroke();
}

/* ── data ───────────────────────────────────────────────────── */
Promise.all([
  fetch('/data/comparison.json').then(r => r.json()).catch(() => null),
  fetch('/data/stride.json').then(r => r.json()).catch(() => null),
  fetch('/data/foot_shapes.json').then(r => r.json()).catch(() => null),
  fetch('/data/mesh_frames.json').then(r => r.json()).catch(() => null),
  fetch('/data/width_locomotion.json').then(r => r.json()).catch(() => null),
]).then(([D, ST, FS, MF, WL]) => {
  statsAndCharts(D);
  if (ST) { legs(ST); cyclo(ST); }
  if (FS) { walkPrints(FS); walkPrints(FS, 'walkPrintsB', true); }
  if (MF) meshWalk(MF);
  if (WL) widthDist(WL);
  footCompare();
  swingLanes();
});

/* ── my recovered body, drawn live rather than played back ──────────────
   These are the actual SMPL-X vertices GVHMR fitted to my footage: 40 frames of one
   gait cycle, decimated to 900 points, seen head-on. Facing the camera is the only view
   in which lateral foot placement exists at all -- in profile both feet fall on the same
   image line however far apart they are, which is the whole subject of this section.
   Drawing the geometry instead of shipping an mp4 also means it takes the page's ink
   colour in either theme and renders at the device's own pixel ratio. */
function meshWalk(MF) {
  const cv = document.getElementById('meshWalk');
  if (!cv) return;
  const ctx = cv.getContext('2d');
  const Q = MF.quant || 10000;
  const F = MF.frames.map(f => Float32Array.from(f, v => v / Q));
  const NP = MF.n_points;

  const TRAIL = 2;        // head-on, older poses hide behind the near one; two is enough
  const DEPTH = 0.42;     // how much further away each older pose sits
  const MS = 70;          // ms per frame of the cycle

  let t0 = null, raf = 0, running = false;

  function draw(now) {
    if (t0 === null) t0 = now;
    const dpr = Math.min(devicePixelRatio || 1, 2);
    const r = cv.getBoundingClientRect();
    const w = r.width, h = r.height || 420;
    cv.width = w * dpr; cv.height = h * dpr;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, w, h);

    const scale = h * 0.92;                 // body height in px at the nearest pose
    const base = h * 0.97;                  // where the near feet stand
    const horizon = h * 0.18;               // vanishing point the trail recedes toward
    const cx = w * 0.5;
    const phase = (now - t0) / MS;
    const ink = css('--ink');

    /* Older poses sit further away: smaller, and their feet ride up toward the horizon.
       That is what makes it read as walking *at* you rather than bobbing on the spot --
       there is no root translation in the data, so depth has to come from the camera. */
    for (let k = TRAIL; k >= 0; k--) {
      const fi = (Math.floor(phase) - k * 2) % F.length;
      const f = F[(fi + F.length * 4) % F.length];
      const lead = k === 0;
      const depth = 1 + k * DEPTH;
      const s = scale / depth;
      const foot = horizon + (base - horizon) / depth;
      const a = lead ? 0.95 : 0.20 * (1 - k / (TRAIL + 1));
      ctx.fillStyle = ink;
      ctx.globalAlpha = a;
      const sz = (lead ? 2.4 : 2.0) / Math.sqrt(depth);
      for (let i = 0; i < NP; i++) {
        const x = cx + f[i * 2] * s;
        const y = foot - f[i * 2 + 1] * s;
        ctx.fillRect(x, y, sz, sz);
      }
    }
    ctx.globalAlpha = 1;
    if (running) raf = requestAnimationFrame(draw);
  }

  new IntersectionObserver(es => es.forEach(e => {
    if (e.isIntersecting && !running) { running = true; raf = requestAnimationFrame(draw); }
    else if (!e.isIntersecting && running) { running = false; cancelAnimationFrame(raf); }
  }), { threshold: 0.05 }).observe(cv);
  addEventListener('resize', () => { if (!running) draw(performance.now()); });
  draw(performance.now());
}


/* ---- a real foot, seen from above ------------------------------------
   The stylised sole used for the data marks elsewhere is a symbol; here the shape has to
   carry an angle and a width, so it needs a real outline: narrow heel, arch biting into the
   medial edge, widest across the ball, and five toes stepping down from the big toe. Drawn
   pointing up the screen with the heel at the origin, so callers position it by the heel,
   which is the landmark both measurements are taken from. */
/* A real foot, traced. The hand-authored bezier version read as a diagram rather than a
   foot, and nudging control points was not converging. This contour is lifted from a
   generated silhouette by an actual edge trace, so the arch, the heel and the five toes
   are the shape's own rather than my approximation of it. Normalised: heel at the
   origin, toes at y=-1, so a caller scales by foot length and rotates about the heel. */
const FOOT_PATH = [
  -0.0091, -0.0013, 0.0157, -0.0013, 0.0352, -0.0052, 0.0587, -0.0144, 0.0796, -0.0287, 0.1057, -0.0614,
  0.1175, -0.0862, 0.1266, -0.1201, 0.1292, -0.1762, 0.124, -0.2089, 0.124, -0.2298, 0.1332, -0.2768,
  0.1292, -0.3355, 0.1371, -0.3825, 0.1384, -0.4543, 0.1423, -0.4909, 0.1527, -0.5379, 0.1932, -0.6462,
  0.2037, -0.6984, 0.2037, -0.7324, 0.1971, -0.7807, 0.1958, -0.8238, 0.2063, -0.8708, 0.205, -0.9112,
  0.1919, -0.9648, 0.1854, -0.9804, 0.1749, -0.9909, 0.1488, -1, 0.1214, -0.9974, 0.1044, -0.9883,
  0.0927, -0.9739, 0.0836, -0.9504, 0.077, -0.9086, 0.0809, -0.8877, 0.0979, -0.846, 0.0966, -0.8316,
  0.0914, -0.8211, 0.0718, -0.8211, 0.0653, -0.8251, 0.0614, -0.8355, 0.0587, -0.8603, 0.0587, -0.9164,
  0.0627, -0.9386, 0.0614, -0.9791, 0.0522, -0.9909, 0.0457, -0.9935, 0.017, -0.9935, -0.0013, -0.9869,
  -0.0091, -0.9791, -0.0117, -0.9713, -0.0091, -0.9517, 0.0026, -0.9178, 0.0039, -0.8525, 0, -0.8329,
  -0.0052, -0.8303, -0.0091, -0.8381, -0.0104, -0.8864, -0.0078, -0.9151, -0.0104, -0.9256, -0.0261, -0.9543,
  -0.0339, -0.9582, -0.0496, -0.9569, -0.0692, -0.9426, -0.0809, -0.923, -0.0809, -0.9073, -0.064, -0.8681,
  -0.064, -0.8499, -0.0587, -0.8264, -0.0601, -0.795, -0.064, -0.7911, -0.0692, -0.7911, -0.0744, -0.8055,
  -0.0757, -0.8251, -0.0692, -0.846, -0.0718, -0.8708, -0.0927, -0.9008, -0.107, -0.9034, -0.1214, -0.8956,
  -0.141, -0.8695, -0.1462, -0.8551, -0.1449, -0.8381, -0.1305, -0.8081, -0.1227, -0.7807, -0.1214, -0.7493,
  -0.1266, -0.7298, -0.1358, -0.7206, -0.1384, -0.7219, -0.1371, -0.7376, -0.1266, -0.7676, -0.1279, -0.7846,
  -0.1371, -0.8029, -0.154, -0.8172, -0.1684, -0.8185, -0.1802, -0.8107, -0.201, -0.7572, -0.1958, -0.6932,
  -0.2063, -0.6423, -0.2076, -0.6005, -0.1906, -0.5222, -0.1775, -0.3995, -0.1671, -0.359, -0.1436, -0.2924,
  -0.1345, -0.2533, -0.1305, -0.1606, -0.1227, -0.1175, -0.107, -0.0783, -0.0875, -0.0496, -0.0653, -0.0274,
  -0.047, -0.0144, -0.0274, -0.0052
];

function realFoot(ctx, hx, hy, len, ang, right) {
  ctx.save();
  ctx.translate(hx, hy);
  ctx.rotate(ang);
  if (right) ctx.scale(-1, 1);            // mirror into the other foot
  ctx.beginPath();
  for (let i = 0; i < FOOT_PATH.length; i += 2) {
    const x = FOOT_PATH[i] * len, y = FOOT_PATH[i + 1] * len;
    i ? ctx.lineTo(x, y) : ctx.moveTo(x, y);
  }
  ctx.closePath();
  ctx.fill();
  ctx.restore();
}

/* ---- where I plant them, against where they are expected --------------
   Both pairs drawn to the same scale from the measured medians: heel separation and
   toe-out, nothing else. The grey pair is the actors' figures, the black pair mine. */
function footCompare() {
  makeChart(document.getElementById('footCompare'), (ctx, w, h, t) => {
    const CM = Math.min(w / 52, h / 32);      // px per cm, with room for the callouts
    const cx = w * 0.5, hy = h * 0.78;        // heel line, centred
    const LEN = 24.5 * CM;                    // a real foot, heel to toe
    const D = Math.PI / 180;

    const pairs = [
      { sep: 15.4, toe: 8.0, col: css('--ink-3'), a: 0.30, label: 'expected' },
      { sep: 6.6, toe: 23.9, col: css('--ink'), a: 0.92, label: 'mine' },
    ];

    pairs.forEach((p, pi) => {
      const grow = Math.max(0, Math.min(1, (t - pi * 0.18) / 0.7));
      if (grow <= 0) return;
      ctx.globalAlpha = p.a * grow;
      ctx.fillStyle = p.col;
      [-1, 1].forEach(side => {
        realFoot(ctx, cx + side * (p.sep / 2) * CM, hy, LEN, side * p.toe * D, side > 0);
      });
      ctx.globalAlpha = 1;
    });

    if (t < 0.55) return;
    const fade = (t - 0.55) / 0.45;
    ctx.globalAlpha = fade;
    ctx.font = '500 11.5px Archivo, sans-serif';

    /* heel separation, marked across the heels themselves */
    const mark = (y, sep, col, text, above) => {
      const x0 = cx - (sep / 2) * CM, x1 = cx + (sep / 2) * CM;
      ctx.strokeStyle = col; ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(x0, y); ctx.lineTo(x1, y);
      ctx.moveTo(x0, y - 4); ctx.lineTo(x0, y + 4);
      ctx.moveTo(x1, y - 4); ctx.lineTo(x1, y + 4);
      ctx.stroke();
      ctx.fillStyle = col; ctx.textAlign = 'center';
      ctx.fillText(text, cx, above ? y - 8 : y + 15);
    };
    /* both dimension lines sit clear of the heels, stacked so neither crosses a foot */
    mark(hy + 26, 6.6, css('--ink'), 'me  6.6 cm', false);
    mark(hy + 54, 15.4, css('--ink-3'), 'expected  15.4 cm', false);

    /* the toe-out angle, drawn as the arc it actually is */
    const ax = cx + (6.6 / 2) * CM, ay = hy;
    ctx.strokeStyle = css('--ink'); ctx.lineWidth = 1; ctx.setLineDash([3, 3]);
    ctx.beginPath(); ctx.moveTo(ax, ay); ctx.lineTo(ax, ay - LEN * 0.92); ctx.stroke();
    ctx.setLineDash([]);
    ctx.beginPath();
    ctx.arc(ax, ay, LEN * 0.55, -Math.PI / 2, -Math.PI / 2 + 23.9 * D);
    ctx.stroke();
    ctx.fillStyle = css('--ink'); ctx.textAlign = 'left';
    ctx.fillText('23.9° toes out', cx + LEN * 0.62, ay - LEN * 0.50);
    ctx.fillStyle = css('--ink-3');
    ctx.fillText('expected 8.0°', cx + LEN * 0.62, ay - LEN * 0.50 + 15);

    /* and the forefoot, which is where the two pairs cross over */
    ctx.fillStyle = css('--ink'); ctx.textAlign = 'right';
    ctx.fillText('at the forefoot  18.4 cm', cx - LEN * 0.60, hy - LEN * 0.94);
    ctx.fillStyle = css('--ink-3');
    ctx.fillText('expected 17.7 cm', cx - LEN * 0.60, hy - LEN * 0.94 + 15);
    ctx.globalAlpha = 1; ctx.textAlign = 'left';
  }, 1.5, 1500);
}

/* ── every measured step, mine against the actors ──────────────────────
   684 steps of mine and 1,384 of theirs, both through the same estimator. The overlap
   is the honest part of the picture: these are distributions, not two bars. */
function widthDist(WL) {
  const A = WL.mine_cm || [], B = WL.actors_cm || [];
  if (!A.length || !B.length) return;
  makeChart(document.getElementById('widthDist'), (ctx, w, h, t) => {
    const pad = { l: 46, r: 20, top: 18, b: 44 };
    const LO = 0, HI = 40, BINS = 40;
    const X = v => pad.l + (v - LO) / (HI - LO) * (w - pad.l - pad.r);
    /* The last bin collects everything past 40 cm rather than discarding it. Dropping the
       tail silently removed 67 of the actors' 1,384 steps, up to 74 cm, which made them look
       tidier than they are and quietly understated the difference. */
    const hist = arr => {
      const c = new Array(BINS).fill(0);
      arr.forEach(v => {
        const i = Math.min(BINS - 1, Math.max(0, Math.floor((v - LO) / (HI - LO) * BINS)));
        c[i]++;
      });
      const m = Math.max(...c) || 1;
      return c.map(v => v / m);
    };
    const hm = hist(A), ht = hist(B);
    const bw = (w - pad.l - pad.r) / BINS;
    const floor = h - pad.b;

    ctx.strokeStyle = css('--line'); ctx.lineWidth = 1;
    ctx.font = '500 11px Archivo, sans-serif'; ctx.textAlign = 'center';
    for (let v = 0; v <= 40; v += 10) {
      ctx.beginPath(); ctx.moveTo(X(v), pad.top); ctx.lineTo(X(v), floor); ctx.stroke();
      ctx.fillStyle = css('--ink-3');
      ctx.fillText(v === 40 ? '40 cm+' : v + ' cm', X(v), floor + 17);
    }
    ctx.fillStyle = css('--ink-3'); ctx.textAlign = 'center';
    ctx.fillText('how far apart the feet landed', (pad.l + w - pad.r) / 2, floor + 36);
    /* the vertical axis is each curve scaled to its own tallest bar, so say that rather
       than leave a number line the reader will read as a count */
    ctx.save();
    ctx.translate(14, (pad.top + floor) / 2); ctx.rotate(-Math.PI / 2);
    ctx.textAlign = 'center';
    ctx.fillText('how often', 0, 0);
    ctx.restore();

    /* actors behind as a filled band, mine in front as a solid outline */
    [[ht, css('--ink-3'), true], [hm, css('--ink'), false]].forEach(([hh, col, fill]) => {
      ctx.beginPath();
      ctx.moveTo(pad.l, floor);
      hh.forEach((v, i) => {
        const x = pad.l + i * bw, y = floor - v * (floor - pad.top) * 0.92 * t;
        ctx.lineTo(x, y); ctx.lineTo(x + bw, y);
      });
      ctx.lineTo(w - pad.r, floor);
      if (fill) { ctx.fillStyle = col; ctx.globalAlpha = 0.22; ctx.fill(); ctx.globalAlpha = 1; }
      ctx.strokeStyle = col; ctx.lineWidth = fill ? 1 : 1.8; ctx.stroke();
    });

    /* the two medians, marked where they actually fall */
    [[6.65, css('--ink'), 'me 6.6'], [15.43, css('--ink-3'), 'actors 15.4']]
      .forEach(([v, col, label], i) => {
        ctx.globalAlpha = Math.max(0, (t - 0.5) * 2);
        ctx.strokeStyle = col; ctx.lineWidth = 1.5; ctx.setLineDash([3, 3]);
        ctx.beginPath(); ctx.moveTo(X(v), pad.top); ctx.lineTo(X(v), floor); ctx.stroke();
        ctx.setLineDash([]);
        ctx.fillStyle = col; ctx.textAlign = 'left';
        ctx.fillText(label, X(v) + 6, pad.top + 12 + i * 15);
        ctx.globalAlpha = 1;
      });
    ctx.textAlign = 'left';
  }, 2.4, 1300);
}

/* ── the walk itself: footfalls stamped where they landed, then fading ──
   Seen from behind, travelling up the page, so the thing the section is about -
   heels near the centre line while the toes turn out, reads left-to-right
   instead of hiding along the viewing axis. Positions and foot angles are the
   measured ones, not drawn: each mark is placed from its own heel and toe.
   The lateral axis is magnified, because a 6 cm gap beside a 26 cm foot is
   otherwise invisible; the caption says so. */
function walkPrints(FS, id = 'walkPrints', down = false) {
  const cv = document.getElementById(id);
  if (!cv) return;
  const ctx = cv.getContext('2d');

  /* Laid out from the measured medians rather than replayed from one pass. A single episode
     carries its own residual path curvature (65 cm of lateral drift over 12 m here) and the
     odd double-detected contact, and at this zoom both swamp the 6.6 cm the section is about.
     The numbers below are the measurement -- 684 steps of mine -- and the caption says so. */
  const S = FS.stats || {};
  const HEEL_SEP = (S.ankle_cm ? S.ankle_cm[0] : 6.6) / 100;
  const TOE_OUT = (S.toeout_deg ? S.toeout_deg[0] : 23.9) * Math.PI / 180;
  const FOOT_L = 0.25;          // heel to toe, metres
  const DRAW = 0.72;            // feet drawn at 72% -- a gesture beside the title, not a plot
  const STRIDE = 0.64;          // measured metres between successive footfalls

  const PERIOD = 380;           // ms between footfalls
  const LIFE = 1900;            // ms a print stays -- long enough that all four overlap
  const GAP = 1000;             // ms of empty screen before the walk comes round again
  const N = 4;                  // footfalls per traverse

  let t0 = null, raf = 0, running = false;

  function draw(now) {
    if (t0 === null) t0 = now;
    const dpr = Math.min(devicePixelRatio || 1, 2);
    const r = cv.getBoundingClientRect();
    const w = r.width, h = r.height || 360;
    cv.width = w * dpr; cv.height = h * dpr;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, w, h);

    /* a diagonal traverse right-to-left, away toward the top corner. The span is kept
       short enough that four footfalls at true proportion (stride 0.64 m against a 0.25 m
       foot) still leave the print big enough to read the toe-out. */
    /* near 45 degrees on screen: the toe-out is +/-24 deg about the line of travel, and on
       a shallow diagonal that splay just reads as two unrelated angles rather than a V */
    /* the closing walk runs the other way, top-left down to bottom-right, so the page
       ends walking out of frame rather than back into it */
    const x0 = down ? w * 0.14 : w * 0.94, y0 = down ? h * 0.12 : h * 0.92;
    const x1 = down ? w * 0.86 : w * 0.16, y1 = down ? h * 0.88 : h * 0.16;
    const dx = x1 - x0, dy = y1 - y0;
    const path = Math.hypot(dx, dy);
    const th = Math.atan2(dy, dx);                 // screen heading
    const scale = path / ((N - 1) * STRIDE);       // px per metre
    const ux = Math.cos(th), uy = Math.sin(th);    // along travel
    const nx = -uy, ny = ux;                       // perpendicular, screen-left of travel

    const cycle = N * PERIOD + GAP;
    const t = (now - t0) % cycle;

    for (let i = 0; i < N; i++) {
      const age = t - i * PERIOD;
      if (age < 0 || age > LIFE) continue;
      /* fade in fast, hold, then out -- a stamp, not a dissolve */
      const a = Math.min(1, age / 90) * (1 - Math.max(0, (age - LIFE * 0.35) / (LIFE * 0.65)));
      if (a <= 0.01) continue;

      const right = i % 2 === 1;
      const s = right ? 1 : -1;
      const along = i * STRIDE * scale;
      const lat = s * (HEEL_SEP / 2) * scale;
      const hx = x0 + ux * along + nx * lat;
      const hy = y0 + uy * along + ny * lat;
      /* the toe leaves the heel at the measured toe-out, turned away from the midline */
      const ta = th + s * TOE_OUT;
      const tx = hx + Math.cos(ta) * FOOT_L * scale;
      const ty = hy + Math.sin(ta) * FOOT_L * scale;

      /* --ink-2 rather than --ink: four solid black marks beside a black headline read
         as a lot of weight for what is a quiet aside */
      ctx.globalAlpha = a * 0.85;
      ctx.fillStyle = css('--ink-2');
      sole(ctx, (hx + tx) / 2, (hy + ty) / 2, FOOT_L * scale * DRAW,
           Math.atan2(ty - hy, tx - hx) + Math.PI / 2, right, true);
    }
    ctx.globalAlpha = 1;

    if (running) raf = requestAnimationFrame(draw);
  }

  /* only animate while it is on screen -- an off-screen rAF loop is wasted battery */
  new IntersectionObserver(es => es.forEach(e => {
    if (e.isIntersecting && !running) { running = true; raf = requestAnimationFrame(draw); }
    else if (!e.isIntersecting && running) { running = false; cancelAnimationFrame(raf); }
  }), { threshold: 0.05 }).observe(cv);

  addEventListener('resize', () => { if (!running) draw(performance.now()); });

  if (matchMedia('(prefers-reduced-motion: reduce)').matches) {
    /* no loop: show one full traverse, all prints at once */
    running = false;
    const one = performance.now();
    t0 = one - (N - 1) * PERIOD - 150;   // clear of the fade-in, so all four are drawn
    draw(one);
  } else {
    draw(performance.now());
  }
}

function statsAndCharts(D) {
  const hip = D?.hip_roll_rom_deg ?? { mine: { median: 18.49 }, lafan1: { median: 47.38 } };
  const cv = D?.step_time_cv ?? { mine: { median: .232, values: [] }, lafan1: { median: .383, values: [] } };

  /* Measured on the SMPL-X and BVH skeletons, NOT on the retargeted G1. The G1 has a fixed
     pelvis width and its own ankle frame, so both of these came out wrong when read off the
     robot: width was clamped up to the robot's stance (6.6 -> 18.6 cm) and toe-out came out
     inverted (23.9 -> 6.1 deg). Retargeting preserves joint angles, not foot geometry. */
  const pct = (a, b) => 100 * (a - b) / Math.abs(b);
  const rows = [
    ['how far my toes point out', '23.9', '°', pct(23.9, 8.0), '8.0°'],
    ['how much my hips sway', hip.mine.median.toFixed(1), '°', pct(hip.mine.median, hip.lafan1.median), hip.lafan1.median.toFixed(1) + '°'],
    ['how wide my heels land', '6.6', 'cm', pct(6.6, 15.4), '15.4cm'],
  ];
  const host = document.getElementById('stats');
  if (host) host.innerHTML = rows.map(([label, v, unit, d, ref]) => `
    <dl class="stat">
      <dt>${label}</dt>
      <dd class="num">${v}<small> ${unit}</small></dd>
      <div class="vs">${d > 0 ? '+' : ''}${d.toFixed(0)}% vs <b>${ref}</b>, mocap actors</div>
    </dl>`).join('');

  /* shoe overlay, where a near-parallel, wide-set foot loads a sole */
  makeChart(document.getElementById('shoeCanvas'), (ctx, w, h, t) => {
    const spots = [[.30, .66, 'heel strike'], [.58, .52, 'mid-stance'], [.79, .40, 'toe-off']];
    ctx.lineWidth = 2;
    spots.forEach(([x, y, text], i) => {
      const p = Math.max(0, Math.min(1, (t - i * .18) / .55));
      if (p <= 0) return;
      const cx = x * w, cy = y * h, r = 16 + i * 7;
      ctx.strokeStyle = MINE(); ctx.globalAlpha = .95;
      ctx.beginPath(); ctx.arc(cx, cy, r, -Math.PI / 2, -Math.PI / 2 + p * Math.PI * 2); ctx.stroke();
      ctx.globalAlpha = .16; ctx.fillStyle = MINE();
      ctx.beginPath(); ctx.arc(cx, cy, r * p, 0, 7); ctx.fill();
      ctx.globalAlpha = p; ctx.fillStyle = css('--ink');
      ctx.font = '500 12px Archivo, sans-serif';
      ctx.fillText(text, cx + r + 8, cy + 4);
      ctx.globalAlpha = 1;
    });
  });

  /* rhythm swarm, every measured segment in both datasets */
  makeChart(document.getElementById('swarm'), (ctx, w, h, t) => {
    const pad = { l: 62, r: 22, t: 24, b: 46 };
    const A = cv.mine.values?.length ? cv.mine.values : [cv.mine.median];
    const B = cv.lafan1.values?.length ? cv.lafan1.values : [cv.lafan1.median];
    const all = A.concat(B), lo = Math.min(...all) * .9, hi = Math.max(...all) * 1.04;
    const X = v => pad.l + (v - lo) / (hi - lo) * (w - pad.l - pad.r);

    ctx.strokeStyle = css('--line'); ctx.lineWidth = 1;
    ctx.font = '500 11.5px Archivo, sans-serif';
    for (let i = 0; i <= 4; i++) {
      const v = lo + (hi - lo) * i / 4, x = X(v);
      ctx.beginPath(); ctx.moveTo(x, pad.t); ctx.lineTo(x, h - pad.b); ctx.stroke();
      ctx.fillStyle = css('--ink-3'); ctx.textAlign = 'center';
      ctx.fillText(v.toFixed(2), x, h - pad.b + 18);
    }
    ctx.textAlign = 'left'; ctx.fillStyle = css('--ink-3');
    ctx.fillText('step-to-step variability   (lower = steadier)', pad.l, h - 10);

    /* mine filled, theirs hollow, with 183 dots in one row, rings stay readable
       where discs would blot. Each dot drops the last few pixels into its place,
       staggered, so the swarm assembles rather than appears. */
    [[B, (h - pad.b) * .62, 'actors', false], [A, (h - pad.b) * .34, 'me', true]]
      .forEach(([vals, y, name, solid]) => {
        /* ink, not accent: the prints read as stamps on paper */
        const col = css(solid ? '--ink' : '--ink-3');
        vals.forEach((v, i) => {
          /* they land left to right at a walking pace, one after another */
          const raw = (t - (solid ? .04 : 0) - (i / vals.length) * .62) / .2;
          const p = Math.max(0, Math.min(1, raw));
          if (p <= 0) return;
          const land = 1 - (1 - p) * (1 - p);          // drops fast, settles slow
          const cy = y + wobble(i) * 26 - (1 - land) * 16;
          const s = 11.5 * (.6 + .4 * land), ang = wobble(i * 3) * .34;
          if (solid) { ctx.globalAlpha = .62 * p; ctx.fillStyle = col; }
          else { ctx.globalAlpha = .66 * p; ctx.strokeStyle = col; ctx.lineWidth = 1.2; }
          sole(ctx, X(v), cy, s, ang, i % 2 === 0, solid);
        });
        ctx.globalAlpha = t;
        const med = vals.slice().sort((a, b) => a - b)[Math.floor(vals.length / 2)];
        ctx.strokeStyle = col; ctx.lineWidth = solid ? 2.5 : 1.6;
        ctx.setLineDash(solid ? [] : [5, 4]);
        ctx.beginPath(); ctx.moveTo(X(med), y - 16); ctx.lineTo(X(med), y + 40); ctx.stroke();
        ctx.setLineDash([]);
        ctx.fillStyle = css('--ink'); ctx.font = '600 13px Archivo, sans-serif';
        ctx.fillText(name, 10, y + 16);
        ctx.globalAlpha = 1;
      });
  }, 2, 3400);

  /* how much data there is on each side, one dot per measured episode */
  makeChart(document.getElementById('episodes'), (ctx, w, h, t) => {
    const nA = cv.mine.values?.length || 60, nB = cv.lafan1.values?.length || 183;
    const head = 22, gap = 30, box = h - head * 2 - gap;
    /* pick the column count whose block shape best fits the box, so the grid fills
       it, this chart has to stand exactly as tall as the swarm beside it */
    const rowsAt = c => Math.ceil(nA / c) + Math.ceil(nB / c);
    let COLS = 30, best = Infinity;
    for (let c = 18; c <= 36; c++) {
      const err = Math.abs(c / rowsAt(c) - w / box);
      if (err < best) { best = err; COLS = c; }
    }
    const rowsOf = n => Math.ceil(n / COLS);
    const cell = Math.min(w / COLS, box / rowsAt(COLS));
    const r = Math.min(cell * .28, 6);
    /* any slack left over is split top and bottom rather than dumped at the end */
    const slack = (box - rowsAt(COLS) * cell) / 2;
    const blockTop = [head + slack, head + slack + rowsOf(nA) * cell + gap + head];

    ctx.textAlign = 'left';
    [[nA, 'me', 'segments', true], [nB, 'mocap actors', 'windows', false]]
      .forEach(([n, name, unit, solid], b) => {
        const top = blockTop[b], col = css(solid ? '--ink' : '--ink-3');
        ctx.font = '600 13px Archivo, sans-serif'; ctx.fillStyle = css('--ink');
        ctx.globalAlpha = Math.min(1, t * 3);
        ctx.fillText(name, 0, top - 8);
        ctx.font = '500 11.5px Archivo, sans-serif'; ctx.fillStyle = css('--ink-3');
        ctx.fillText(`${n} ${unit}`, ctx.measureText(name).width + 74, top - 8);

        for (let i = 0; i < n; i++) {
          /* one print at a time, left then right along each row: a trail being
             walked rather than a grid being filled */
          const delay = (b * .06) + (i / Math.max(nA, nB)) * .74;
          const p = Math.max(0, Math.min(1, (t - delay) / .16));
          if (p <= 0) continue;
          const land = 1 - (1 - p) * (1 - p);
          const cx = (i % COLS) * cell + cell / 2;
          const cy = top + Math.floor(i / COLS) * cell + cell / 2 - (1 - land) * 11;
          const s = cell * .76 * (.45 + .55 * land), ang = wobble(i * 7) * .26;
          if (solid) { ctx.globalAlpha = .9 * p; ctx.fillStyle = col; }
          else { ctx.globalAlpha = .8 * p; ctx.strokeStyle = col; ctx.lineWidth = 1.2; }
          sole(ctx, cx, cy, s, ang, i % 2 === 0, solid);
        }
        ctx.globalAlpha = 1;
      });
  }, 2, 3800);
}

/* ── legs: two chains posed straight from measured joint angles ── */
function legs(ST) {
  const cv = document.getElementById('legs');
  if (!cv) return;
  const ctx = cv.getContext('2d');
  const label = document.getElementById('phaseLabel');
  const N = ST.n_samples || 100;
  const at = (arr, u) => {
    const x = (((u % 1) + 1) % 1) * N, i = Math.floor(x), f = x - i;
    return arr[i % N] * (1 - f) + arr[(i + 1) % N] * f;
  };

  function leg(cx, cy, S, u, col, s, lead) {
    const hp = at(lead ? S.hip_pitch : S.r_hip_pitch, u);
    const kn = at(lead ? S.knee : S.r_knee, u);
    const an = at(lead ? S.ankle : S.r_ankle, u);
    const L1 = 64 * s, L2 = 60 * s, FT = 24 * s;
    const kx = cx + Math.sin(hp) * L1, ky = cy + Math.cos(hp) * L1;
    const a2 = hp - kn;
    const ax = kx + Math.sin(a2) * L2, ay = ky + Math.cos(a2) * L2;
    const tx = ax + Math.cos(a2 + an) * FT, ty = ay + Math.abs(Math.sin(a2 + an)) * FT * .3;
    ctx.strokeStyle = col; ctx.lineCap = 'round'; ctx.lineJoin = 'round';
    ctx.lineWidth = lead ? 5 : 3.2; ctx.globalAlpha = lead ? 1 : .38;
    ctx.beginPath(); ctx.moveTo(cx, cy); ctx.lineTo(kx, ky); ctx.lineTo(ax, ay); ctx.lineTo(tx, ty); ctx.stroke();
    ctx.fillStyle = col;
    ctx.beginPath(); ctx.arc(kx, ky, lead ? 4 : 3, 0, 7); ctx.fill();
    ctx.beginPath(); ctx.arc(ax, ay, lead ? 3.4 : 2.6, 0, 7); ctx.fill();
    ctx.globalAlpha = 1;
    return [tx, ty];
  }

  const trails = [[], []];
  let u = 0, last = performance.now(), visible = false;
  new IntersectionObserver(e => { visible = e[0].isIntersecting; }, { threshold: .1 }).observe(cv);

  (function frame(now) {
    const dt = Math.min((now - last) / 1000, .05); last = now;
    if (visible) u += dt / 1.9;                       // ~1.9 s per stride loop
    const dpr = Math.min(devicePixelRatio || 1, 2);
    const r = cv.getBoundingClientRect(), w = r.width, h = r.height;
    if (w && h) {
      cv.width = w * dpr; cv.height = h * dpr; ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.clearRect(0, 0, w, h);
      const ground = h * .78, s = Math.min(w / 780, h / 380) * 1.45;
      ctx.strokeStyle = css('--line'); ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(w * .05, ground); ctx.lineTo(w * .95, ground); ctx.stroke();

      [[w * .30, ST.mine, MINE(), 0], [w * .70, ST.actors, THEM(), 1]].forEach(([cx, S, col, k]) => {
        const hipY = ground - 124 * s;
        leg(cx, hipY, S, u + .5, col, s, false);       // trailing leg, half a cycle behind
        const toe = leg(cx, hipY, S, u, col, s, true);
        ctx.strokeStyle = col; ctx.lineWidth = 4.5;
        ctx.beginPath(); ctx.moveTo(cx, hipY); ctx.lineTo(cx, hipY - 50 * s); ctx.stroke();
        ctx.fillStyle = col; ctx.beginPath(); ctx.arc(cx, hipY, 6.5, 0, 7); ctx.fill();
        const tr = trails[k]; tr.push(toe); if (tr.length > 100) tr.shift();
        ctx.globalAlpha = .3; ctx.lineWidth = 1.6; ctx.strokeStyle = col;
        sketch(ctx, tr, .5); ctx.globalAlpha = 1;
      });

      if (label) {
        const pc = Math.round((((u % 1) + 1) % 1) * 100);
        label.textContent = `gait cycle ${String(pc).padStart(2, '0')}%  ·  toe path traced live from measured joint angles`;
      }
    }
    requestAnimationFrame(frame);
  })(last);
}

/* ── cyclogram: hip pitch against knee flexion, real stride ─── */
function cyclo(ST) {
  makeChart(document.getElementById('cyclo'), (ctx, w, h, t) => {
    const pad = { l: 62, r: 26, t: 26, b: 50 };
    const PW = w - pad.l - pad.r, PH = h - pad.t - pad.b;
    const sets = [['actors', ST.actors, THEM()], ['me', ST.mine, MINE()]];
    const allH = sets.flatMap(([, S]) => S.hip_pitch), allK = sets.flatMap(([, S]) => S.knee);
    const hx = [Math.min(...allH), Math.max(...allH)], ky = [Math.min(...allK), Math.max(...allK)];
    const X = v => pad.l + (v - hx[0]) / ((hx[1] - hx[0]) || 1) * PW;
    const Y = v => pad.t + PH - (v - ky[0]) / ((ky[1] - ky[0]) || 1) * PH;

    ctx.strokeStyle = css('--line'); ctx.lineWidth = 1;
    for (let i = 0; i <= 4; i++) {
      const y = pad.t + PH * i / 4, x = pad.l + PW * i / 4;
      ctx.beginPath(); ctx.moveTo(pad.l, y); ctx.lineTo(w - pad.r, y); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(x, pad.t); ctx.lineTo(x, h - pad.b); ctx.stroke();
    }
    ctx.fillStyle = css('--ink-3'); ctx.font = '500 11.5px Archivo, sans-serif';
    ctx.fillText('hip pitch, radians →', pad.l, h - 14);
    ctx.save(); ctx.translate(16, pad.t + PH / 2); ctx.rotate(-Math.PI / 2);
    ctx.textAlign = 'center'; ctx.fillText('knee flexion →', 0, 0); ctx.restore();
    ctx.textAlign = 'left';

    sets.forEach(([name, S, col]) => {
      const n = Math.floor(S.hip_pitch.length * t);
      const pts = [];
      for (let i = 0; i <= n; i++) {
        pts.push([X(S.hip_pitch[i % S.hip_pitch.length]), Y(S.knee[i % S.knee.length])]);
      }
      ctx.strokeStyle = col; ctx.lineWidth = name === 'me' ? 2.8 : 2;
      ctx.globalAlpha = name === 'me' ? 1 : .72;
      sketch(ctx, pts, .7);
      if (pts.length) {
        const last = pts[pts.length - 1];
        ctx.fillStyle = col; ctx.globalAlpha = 1;
        ctx.beginPath(); ctx.arc(last[0], last[1], 5, 0, 7); ctx.fill();
      }
      ctx.globalAlpha = 1;
    });
  });
}

/* ---- swing time, played rather than plotted -------------------------------
   A bar chart of three numbers would be honest and forgettable. Swing time is a DURATION,
   so the readable form is a foot actually taking that long to travel: three lanes each
   stepping at its own measured rate, in real time, against the same traced foot used
   everywhere else on the page. The rhythm is the data.
   Measured over four minutes of simulation per policy, ~450 steps each. */
const SWING = [
  { label: 'me',                   t: 0.43, col: () => css('--ink') },
  { label: 'trained on my walk',   t: 0.42, col: () => css('--ink-2') },
  { label: 'trained on actors',    t: 0.26, col: () => css('--ink-3') },
];

function swingLanes() {
  makeChart(document.getElementById('swing'), (ctx, w, h, t) => {
    /* Swing time is a rhythm, so it is drawn as one: a continuous chain of arcs, each arc
       exactly one swing long on a shared time axis. Nothing is assumed about stance or
       speed -- only the measured airborne duration sets the wavelength, so a faster robot
       simply packs more arcs into the same seconds. Strokes carry the page's hand-drawn
       jitter rather than being plotted clean. */
    const pad = { l: 172, r: 92, top: 24, b: 44 };
    const SPAN = 2.4;                                   // seconds shown across the row
    const X = v => pad.l + (v / SPAN) * (w - pad.l - pad.r);
    const rowH = (h - pad.top - pad.b) / SWING.length;

    ctx.font = '500 11.5px Archivo, sans-serif';
    ctx.textAlign = 'center'; ctx.fillStyle = css('--ink-3');
    ctx.strokeStyle = css('--line'); ctx.lineWidth = 1;
    for (let v = 0; v <= SPAN; v += 0.4) {
      ctx.beginPath(); ctx.moveTo(X(v), pad.top); ctx.lineTo(X(v), h - pad.b); ctx.stroke();
      ctx.fillText(v.toFixed(1) + 's', X(v), h - pad.b + 17);
    }
    ctx.fillText('seconds', (pad.l + w - pad.r) / 2, h - pad.b + 33);

    SWING.forEach((sw, i) => {
      const cy = pad.top + rowH * (i + 0.68);
      const col = sw.col();
      const amp = rowH * 0.40;
      const arcW = X(sw.t) - X(0);

      ctx.fillStyle = css('--ink'); ctx.textAlign = 'right';
      ctx.font = '500 13px Archivo, sans-serif';
      ctx.fillText(sw.label, pad.l - 18, cy + 4);

      // the ground: everything below the line is contact, every arch is one swing
      ctx.strokeStyle = css('--line'); ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(pad.l, cy); ctx.lineTo(w - pad.r, cy); ctx.stroke();

      const reveal = X(0) + (w - pad.r - X(0)) * t;
      ctx.strokeStyle = col; ctx.lineWidth = 2; ctx.lineJoin = 'round';
      let k = 0;
      for (let x0 = X(0); x0 < w - pad.r; x0 += arcW, k++) {
        const pts = [];
        for (let u = 0; u <= 1.0001; u += 0.05) {
          const x = x0 + u * arcW;
          if (x > reveal) break;
          pts.push([x, cy - Math.sin(u * Math.PI) * amp]);
        }
        if (pts.length > 1) sketch(ctx, pts, 1.1 + (k % 3) * 0.25);
      }

      if (t > 0.9) {
        ctx.globalAlpha = (t - 0.9) / 0.1;
        ctx.fillStyle = col; ctx.textAlign = 'left';
        ctx.font = '600 16px "JetBrains Mono", monospace';
        ctx.fillText(sw.t.toFixed(2) + 's', w - pad.r + 14, cy + 5);
        ctx.globalAlpha = 1;
      }
    });
    ctx.textAlign = 'left';
  }, 2.4, 1600);
}
