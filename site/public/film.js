/* my walk of life — the film.
   The page is a scroll; this is the same material as a timeline. Every section becomes one
   scene, each scene owns a slice of a single clock, and the animations are driven from that
   clock rather than from scroll position or from their own rAF loops. That means a scene can
   be scrubbed, replayed, or paused deterministically, which a pile of independent
   IntersectionObservers cannot do.

   The drawing primitives (the traced foot, the measured numbers) are the same ones the page
   uses, so the two never drift apart. */

const css = v => getComputedStyle(document.body).getPropertyValue(v).trim();
const clamp = (v, a = 0, b = 1) => v < a ? a : v > b ? b : v;
const ease = t => t * t * (3 - 2 * t);
const easeOut = t => 1 - Math.pow(1 - t, 3);

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
/* heel at the origin, toes one length forward, so callers place it by the heel */
function realFoot(ctx, hx, hy, len, ang, right) {
  ctx.save();
  ctx.translate(hx, hy);
  ctx.rotate(ang);
  if (right) ctx.scale(-1, 1);
  ctx.beginPath();
  for (let i = 0; i < FOOT_PATH.length; i += 2) {
    const x = FOOT_PATH[i] * len, y = FOOT_PATH[i + 1] * len;
    i ? ctx.lineTo(x, y) : ctx.moveTo(x, y);
  }
  ctx.closePath();
  ctx.fill();
  ctx.restore();
}

/* deterministic jitter so strokes read as drawn rather than plotted */
const wobble = i => ((Math.sin(i * 12.9898) * 43758.5453) % 1);
function sketch(ctx, pts, amt = 0.9) {
  ctx.beginPath();
  pts.forEach((p, i) => {
    const j = wobble(i) * amt;
    i ? ctx.lineTo(p[0] + j, p[1] - j) : ctx.moveTo(p[0], p[1]);
  });
  ctx.stroke();
}

function hidpi(cv, ratio) {
  const dpr = Math.min(devicePixelRatio || 1, 2);
  const r = cv.getBoundingClientRect();
  const w = r.width, h = r.height || r.width / (ratio || 2);
  cv.width = w * dpr; cv.height = h * dpr;
  const ctx = cv.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, w, h);
  return [ctx, w, h];
}

/* ── the film ──────────────────────────────────────────────────────
   Each scene declares how long it holds, the markup it needs, and a draw(u) where u runs
   0→1 across the scene. Assets are the page's own. */

let DATA = { mesh: null, feet: null, width: null };

const SCENES = [

/* 1. title */
{ id: 'title', dur: 5.5, html: `
    <div class="inner" style="text-align:left">
      <h1><span class="w">my walk</span><br><em class="w">of life</em></h1>
      <p class="lede" style="margin-top:2rem"><span class="w">Everyone tells me that I walk in a particular way.</span>
      <span class="w">Just like my dad did.</span></p>
      <canvas id="c-title" style="height:150px;margin-top:2rem"></canvas>
    </div>`,
  draw(u, el) {
    words(el, u, 0.05, 0.34);
    const [ctx, w, h] = hidpi(el.querySelector('#c-title'));
    const N = 5, stride = w / (N + 1.4), L = Math.min(56, stride * 0.52);
    ctx.fillStyle = css('--ink-2');
    for (let i = 0; i < N; i++) {
      const a = clamp((u - 0.24 - i * 0.09) / 0.16);
      if (a <= 0) continue;
      ctx.globalAlpha = a * 0.9;
      const s = i % 2 ? 1 : -1;
      realFoot(ctx, stride * (i + 0.8), h * 0.62 + s * 13, L,
               Math.PI / 2 + s * 0.42, i % 2 === 1);
    }
    ctx.globalAlpha = 1;
  }},

/* 2. him and me */
{ id: 'dad', dur: 5, html: `
    <div class="inner two-up">
      <div>
        <p class="kicker">where it starts</p>
        <h2><span class="w">It runs</span> <span class="w">in the family.</span></h2>
      </div>
      <figure class="plate"><img src="/img/dad-and-me.jpg" alt="Me as a boy beside my dad">
        <figcaption><span class="marker">him and me</span></figcaption></figure>
    </div>`,
  draw(u, el) { words(el, u, 0.08, 0.4); rise(el.querySelector('figure'), u, 0.1); }},

/* 3. the shoes — the evidence that started it */
{ id: 'shoes', dur: 5, html: `
    <div class="inner two-up">
      <figure class="plate"><img src="/img/shoe2.png" alt="A sneaker seen from above, its heel splayed"></figure>
      <div>
        <h2><span class="w">My shoes get</span> <span class="w">a weird shape</span> <span class="w">after a while.</span></h2>
      </div>
    </div>`,
  draw(u, el) { words(el, u, 0.1, 0.42); rise(el.querySelector('figure'), u, 0.05); }},

/* 4. the rig */
{ id: 'rig', dur: 4.5, html: `
    <div class="inner two-up">
      <div><h2><span class="w">So I checked</span> <span class="w">whether it was true.</span></h2>
        <p class="lede" style="margin-top:1.2rem"><span class="w">A phone under a hat brim. 72 clips of me walking.</span></p></div>
      <figure class="plate"><img src="/img/hat-rig.jpg" alt="A phone mounted under a hat brim">
        <figcaption><span class="marker">the rig</span></figcaption></figure>
    </div>`,
  draw(u, el) { words(el, u, 0.08, 0.36); rise(el.querySelector('figure'), u, 0.12); }},

/* 5. the verdict */
{ id: 'verdict', dur: 3.6, html: `
    <div class="inner" style="text-align:center">
      <h1 class="wide" style="max-width:none"><span class="w">Turns out</span><br><span class="w">it's</span> <em class="w">true</em></h1>
    </div>`,
  draw(u, el) { words(el, u, 0.12, 0.5); }},

/* 6. the walk itself, stamped */
{ id: 'prints', dur: 7, html: `
    <div class="inner">
      <h2 class="wide"><span class="w">My heels walk a tightrope.</span> <span class="w">My toes don't.</span></h2>
      <canvas id="c-prints" style="height:min(52vh,420px);margin-top:1.4rem"></canvas>
    </div>`,
  draw(u, el) {
    words(el, u, 0.06, 0.36);
    const [ctx, w, h] = hidpi(el.querySelector('#c-prints'));
    /* laid out from the measured medians: heels 6.6 cm apart, toes turned out 23.9 deg */
    const HEEL = 0.066, TOE = 23.9 * Math.PI / 180, FOOT = 0.25, STRIDE = 0.64, N = 7;
    const x0 = w * 0.10, y0 = h * 0.88, x1 = w * 0.92, y1 = h * 0.18;
    const dx = x1 - x0, dy = y1 - y0, th = Math.atan2(dy, dx);
    const scale = Math.hypot(dx, dy) / ((N - 1) * STRIDE);
    const ux = Math.cos(th), uy = Math.sin(th), nx = -uy, ny = ux;
    for (let i = 0; i < N; i++) {
      const a = clamp((u - 0.18 - i * 0.075) / 0.12);
      if (a <= 0) continue;
      const s = i % 2 ? 1 : -1;
      const along = i * STRIDE * scale, lat = s * (HEEL / 2) * scale;
      const hx = x0 + ux * along + nx * lat, hy = y0 + uy * along + ny * lat;
      ctx.globalAlpha = a * 0.9;
      ctx.fillStyle = css('--ink-2');
      realFoot(ctx, hx, hy, FOOT * scale * 0.8, th + s * TOE + Math.PI / 2, i % 2 === 1);
    }
    ctx.globalAlpha = 1;
  }},

/* 7. the two foot positions, to scale */
{ id: 'feet', dur: 6.5, html: `
    <div class="inner two-up">
      <div>
        <p class="kicker">measured</p>
        <h2><span class="w">Heels in.</span> <span class="w">Toes out.</span></h2>
        <div class="stat-row" style="flex-direction:column;gap:1.5rem">
          <div class="stat"><div class="n" data-n="6.6" data-d="1">0<small>cm</small></div>
            <div class="k">between my heels</div><div class="v">actors: 15.4 cm</div></div>
          <div class="stat"><div class="n" data-n="23.9" data-d="1">0<small>°</small></div>
            <div class="k">my toes turn out</div><div class="v">actors: 8.0°</div></div>
        </div>
      </div>
      <figure><canvas id="c-feet" style="height:min(56vh,460px)"></canvas></figure>
    </div>`,
  draw(u, el) {
    words(el, u, 0.06, 0.4);
    counters(el, clamp((u - 0.2) / 0.45));
    const [ctx, w, h] = hidpi(el.querySelector('#c-feet'));
    const CM = Math.min(w / 44, h / 30), cx = w * 0.5, hy = h * 0.80, LEN = 24.5 * CM;
    const D = Math.PI / 180;
    [{ sep: 15.4, toe: 8.0, col: css('--ink-3'), a: .30, t0: 0.10 },
     { sep: 6.6, toe: 23.9, col: css('--ink'), a: .92, t0: 0.26 }].forEach(p => {
      const g = clamp((u - p.t0) / 0.3);
      if (g <= 0) return;
      ctx.globalAlpha = p.a * g; ctx.fillStyle = p.col;
      [-1, 1].forEach(s => realFoot(ctx, cx + s * (p.sep / 2) * CM, hy, LEN, s * p.toe * D, s > 0));
    });
    ctx.globalAlpha = 1;
  }},

/* 8. the recovered body */
{ id: 'body', dur: 6.5, html: `
    <div class="inner two-up">
      <figure><canvas id="c-body" style="height:min(60vh,520px)"></canvas>
        <figcaption>the SMPL-X body fitted to my footage</figcaption></figure>
      <div>
        <p class="kicker">from video to body</p>
        <h2><span class="w">A phone,</span> <span class="w">then a body,</span> <span class="w">then a robot.</span></h2>
      </div>
    </div>`,
  draw(u, el) {
    words(el, u, 0.1, 0.4);
    const F = DATA.mesh; if (!F) return;
    const [ctx, w, h] = hidpi(el.querySelector('#c-body'));
    const scale = h * 0.84, base = h * 0.96, cx = w * 0.5;
    const fi = Math.floor(u * F.frames.length * 3) % F.frames.length;
    const f = F.frames[fi];
    ctx.fillStyle = css('--ink');
    for (let i = 0; i < F.n_points; i++) {
      ctx.globalAlpha = 0.9;
      ctx.fillRect(cx + f[i * 2] * scale, base - f[i * 2 + 1] * scale, 2.4, 2.4);
    }
    ctx.globalAlpha = 1;
  }},

/* 9. every step, both datasets */
{ id: 'dist', dur: 6, html: `
    <div class="inner">
      <h2 class="wide"><span class="w">Every step I took,</span> <span class="w">against theirs.</span></h2>
      <figure><canvas id="c-dist" style="height:min(48vh,380px)"></canvas>
        <figcaption>how far apart the feet landed. 684 of my steps against 1,384 of the actors'</figcaption></figure>
    </div>`,
  draw(u, el) {
    words(el, u, 0.06, 0.34);
    const W = DATA.width; if (!W) return;
    const [ctx, w, h] = hidpi(el.querySelector('#c-dist'));
    const pad = { l: 40, r: 30, t: 16, b: 40 }, LO = 0, HI = 40, BINS = 40;
    const X = v => pad.l + (v - LO) / (HI - LO) * (w - pad.l - pad.r), floor = h - pad.b;
    const hist = arr => { const c = new Array(BINS).fill(0);
      arr.forEach(v => { const i = Math.min(BINS - 1, Math.max(0, Math.floor(v / HI * BINS))); c[i]++; });
      const m = Math.max(...c) || 1; return c.map(v => v / m); };
    const hm = hist(W.mine_cm), ht = hist(W.actors_cm), bw = (w - pad.l - pad.r) / BINS;
    ctx.strokeStyle = css('--line'); ctx.lineWidth = 1;
    ctx.font = '500 11px ' + css('--sans'); ctx.textAlign = 'center'; ctx.fillStyle = css('--ink-3');
    for (let v = 0; v <= 40; v += 10) {
      ctx.beginPath(); ctx.moveTo(X(v), pad.t); ctx.lineTo(X(v), floor); ctx.stroke();
      ctx.fillText(v + (v === 40 ? ' cm+' : ' cm'), X(v), floor + 17);
    }
    const g = clamp((u - 0.12) / 0.5);
    [[ht, css('--ink-3'), true], [hm, css('--ink'), false]].forEach(([hh, col, fill]) => {
      ctx.beginPath(); ctx.moveTo(pad.l, floor);
      hh.forEach((v, i) => { const x = pad.l + i * bw, y = floor - v * (floor - pad.t) * 0.9 * g;
        ctx.lineTo(x, y); ctx.lineTo(x + bw, y); });
      ctx.lineTo(w - pad.r, floor);
      if (fill) { ctx.fillStyle = col; ctx.globalAlpha = .2; ctx.fill(); ctx.globalAlpha = 1; }
      ctx.strokeStyle = col; ctx.lineWidth = fill ? 1 : 1.8; ctx.stroke();
    });
  }},

/* 10. the robot */
{ id: 'robot', dur: 5.5, html: `
    <div class="inner">
      <h2 class="wide"><span class="w">I only saw it</span> <span class="w">once it was a robot.</span></h2>
      <figure class="plate" style="margin-top:1.2rem">
        <video src="/video/robot-sidebyside.mp4" autoplay loop muted playsinline></video>
        <figcaption>my footage, then the same instant on a Unitree G1</figcaption></figure>
    </div>`,
  draw(u, el) { words(el, u, 0.06, 0.34); rise(el.querySelector('figure'), u, 0.1); }},

/* 11. two robots */
{ id: 'two', dur: 6.5, html: `
    <div class="inner">
      <h2 class="wide"><span class="w">Then I trained</span> <span class="w">two robots.</span></h2>
      <p class="lede" style="margin-bottom:1.4rem"><span class="w">One learned from my 93 walks. The other learned from actors.</span></p>
      <figure class="plate"><video src="/video/two-policies.mp4" autoplay loop muted playsinline></video></figure>
    </div>`,
  draw(u, el) { words(el, u, 0.05, 0.32); rise(el.querySelector('figure'), u, 0.16); }},

/* 12. the rhythm — the one thing that carried */
{ id: 'swing', dur: 7, html: `
    <div class="inner">
      <h2 class="wide"><span class="w">Neither plants its feet like me.</span>
        <span class="w">But one keeps my time.</span></h2>
      <figure><canvas id="c-swing" style="height:min(46vh,340px)"></canvas>
        <figcaption>how long each foot spends off the ground, per step</figcaption></figure>
    </div>`,
  draw(u, el) {
    words(el, u, 0.05, 0.34);
    const [ctx, w, h] = hidpi(el.querySelector('#c-swing'));
    const rows = [['me', 0.43, css('--ink')],
                  ['trained on my walk', 0.42, css('--ink-2')],
                  ['trained on actors', 0.26, css('--ink-3')]];
    const pad = { l: 168, r: 84, t: 16, b: 34 }, SPAN = 2.4;
    const X = v => pad.l + (v / SPAN) * (w - pad.l - pad.r);
    const rowH = (h - pad.t - pad.b) / rows.length;
    const g = clamp((u - 0.16) / 0.6);
    rows.forEach(([label, t, col], i) => {
      const cy = pad.t + rowH * (i + 0.66), amp = rowH * 0.38, arcW = X(t) - X(0);
      ctx.fillStyle = css('--ink'); ctx.textAlign = 'right';
      ctx.font = '500 13px ' + css('--sans');
      ctx.fillText(label, pad.l - 18, cy + 4);
      ctx.strokeStyle = css('--line'); ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(pad.l, cy); ctx.lineTo(w - pad.r, cy); ctx.stroke();
      const reveal = pad.l + (w - pad.r - pad.l) * g;
      ctx.strokeStyle = col; ctx.lineWidth = 2; ctx.lineJoin = 'round';
      let k = 0;
      for (let x0 = pad.l; x0 < w - pad.r; x0 += arcW, k++) {
        const pts = [];
        for (let q = 0; q <= 1.0001; q += 0.06) {
          const x = x0 + q * arcW; if (x > reveal) break;
          pts.push([x, cy - Math.sin(q * Math.PI) * amp]);
        }
        if (pts.length > 1) sketch(ctx, pts, 1 + (k % 3) * 0.2);
      }
      if (u > 0.62) {
        ctx.globalAlpha = clamp((u - 0.62) / 0.14);
        ctx.fillStyle = col; ctx.textAlign = 'left';
        ctx.font = '600 15px ' + css('--mono');
        ctx.fillText(t.toFixed(2) + 's', w - pad.r + 12, cy + 5);
        ctx.globalAlpha = 1;
      }
    });
  }},

/* 13. close */
{ id: 'end', dur: 5.5, html: `
    <div class="inner" style="text-align:center">
      <h2 class="wide" style="margin:0 auto"><span class="w">Of everything that makes my walk mine,</span>
        <span class="w">one thing was inherited.</span></h2>
      <p class="lede" style="margin:1.6rem auto 0"><span class="w">The rhythm.</span></p>
      <canvas id="c-end" style="height:140px;margin-top:1.8rem"></canvas>
    </div>`,
  draw(u, el) {
    words(el, u, 0.06, 0.4);
    const [ctx, w, h] = hidpi(el.querySelector('#c-end'));
    const N = 6, stride = w / (N + 1.6), L = Math.min(52, stride * 0.5);
    ctx.fillStyle = css('--ink-3');
    for (let i = 0; i < N; i++) {
      const a = clamp((u - 0.4 - i * 0.06) / 0.12);
      if (a <= 0) continue;
      const s = i % 2 ? 1 : -1;
      ctx.globalAlpha = a * 0.75;
      realFoot(ctx, stride * (i + 0.9), h * 0.6 + s * 12, L, Math.PI / 2 + s * 0.42, i % 2 === 1);
    }
    ctx.globalAlpha = 1;
  }},
];

/* ── shared entrance helpers ───────────────────────────────────────
   Words arrive one at a time and elements rise into place. Both are driven from the scene's
   own u rather than from CSS animations, so scrubbing backwards puts them back. */
function words(el, u, start, span) {
  const ws = el.querySelectorAll('.w');
  ws.forEach((n, i) => {
    const t = start + (span / Math.max(ws.length, 1)) * i;
    n.classList.toggle('in', u > t);
  });
}
function rise(node, u, start) {
  if (!node) return;
  const p = easeOut(clamp((u - start) / 0.4));
  node.style.opacity = p;
  node.style.transform = `translateY(${(1 - p) * 22}px)`;
}
function counters(el, p) {
  el.querySelectorAll('[data-n]').forEach(n => {
    const target = parseFloat(n.dataset.n), d = parseInt(n.dataset.d || '0', 10);
    const unit = n.querySelector('small');
    const v = (target * ease(p)).toFixed(d);
    n.firstChild.nodeValue = v;
    if (unit) n.appendChild(unit);
  });
}

/* ── build ─────────────────────────────────────────────────────── */
const stage = document.getElementById('stage');
const fill = document.getElementById('fill');
const chapters = document.getElementById('chapters');
const hud = document.getElementById('hud');
const clockEl = document.getElementById('clock');

SCENES.forEach((s, i) => {
  const d = document.createElement('section');
  d.className = 'scene'; d.id = 's-' + s.id; d.innerHTML = s.html;
  stage.appendChild(d); s.el = d;
  s.start = SCENES.slice(0, i).reduce((a, x) => a + x.dur, 0);
  const c = document.createElement('i');
  c.title = s.id;
  c.onclick = () => { clock = s.start + 0.001; render(); };
  chapters.appendChild(c);
});
const TOTAL = SCENES.reduce((a, s) => a + s.dur, 0);

/* Deep-linking: ?scene=prints jumps to a scene, ?t=12.5 to a time, and ?still=1 freezes
   there. Useful for sharing a moment, and it is the only way to inspect a given scene in a
   headless browser, where requestAnimationFrame does not advance. */
const params = new URLSearchParams(location.search);
let clock = 0, playing = true, last = performance.now(), current = -1;
{
  const sc = params.get('scene');
  if (sc) { const s = SCENES.find(x => x.id === sc); if (s) clock = s.start + s.dur * 0.55; }
  const tt = parseFloat(params.get('t'));
  if (!isNaN(tt)) clock = Math.max(0, tt);
  if (params.get('still') === '1') playing = false;
}

function render() {
  let idx = SCENES.findIndex(s => clock >= s.start && clock < s.start + s.dur);
  if (idx < 0) idx = SCENES.length - 1;
  const s = SCENES[idx];
  if (idx !== current) {
    SCENES.forEach((x, i) => x.el.classList.toggle('on', i === idx));
    /* restart any video so each scene begins at its first frame */
    s.el.querySelectorAll('video').forEach(v => { try { v.currentTime = 0; v.play(); } catch (e) {} });
    current = idx;
  }
  const u = clamp((clock - s.start) / s.dur);
  try { s.draw && s.draw(u, s.el); } catch (e) { /* a scene must never kill the film */ }
  fill.style.width = (clock / TOTAL * 100) + '%';
  const sec = Math.floor(clock);
  clockEl.textContent = `${Math.floor(sec / 60)}:${String(sec % 60).padStart(2, '0')} / ${Math.floor(TOTAL / 60)}:${String(Math.floor(TOTAL % 60)).padStart(2, '0')}`;
}

function frame(now) {
  const dt = Math.min((now - last) / 1000, 0.05);
  last = now;
  if (playing) {
    clock += dt;
    if (clock >= TOTAL) clock = 0;          // loop
  }
  render();
  requestAnimationFrame(frame);
}

/* ── transport ─────────────────────────────────────────────────── */
const playBtn = document.getElementById('play');
const setPlay = v => { playing = v; playBtn.textContent = v ? 'pause' : 'play'; };
playBtn.onclick = () => setPlay(!playing);
document.getElementById('fwd').onclick = () => {
  const n = SCENES.find(s => s.start > clock + 0.01);
  clock = n ? n.start + 0.001 : 0;
};
document.getElementById('back').onclick = () => {
  const cur = SCENES[current];
  clock = (clock - cur.start > 0.6) ? cur.start + 0.001
        : Math.max(0, (SCENES[current - 1] || SCENES[0]).start + 0.001);
};
addEventListener('keydown', e => {
  if (e.code === 'Space') { e.preventDefault(); setPlay(!playing); }
  if (e.code === 'ArrowRight') document.getElementById('fwd').click();
  if (e.code === 'ArrowLeft') document.getElementById('back').click();
});
let hideT;
addEventListener('pointermove', () => {
  hud.classList.add('show');
  clearTimeout(hideT);
  hideT = setTimeout(() => hud.classList.remove('show'), 2200);
});

if (matchMedia('(prefers-reduced-motion: reduce)').matches) setPlay(false);
if (params.get('still') === '1') setPlay(false);

/* data the scenes need; the film starts either way rather than blocking on it */
Promise.all([
  fetch('/data/mesh_frames.json').then(r => r.json()).catch(() => null),
  fetch('/data/width_locomotion.json').then(r => r.json()).catch(() => null),
]).then(([mesh, width]) => {
  if (mesh) {
    const Q = mesh.quant || 10000;
    DATA.mesh = { n_points: mesh.n_points,
                  frames: mesh.frames.map(f => Float32Array.from(f, v => v / Q)) };
  }
  DATA.width = width;
});

requestAnimationFrame(frame);
