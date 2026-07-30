/* 四轨混音台 —— P7 的最小可用版本。
 *
 * 音频图（每轨）：
 *   AudioBufferSourceNode → lowShelf → peaking → highShelf
 *     → trackGain → StereoPanner → trackAnalyser → masterGain → masterAnalyser → destination
 *
 * 三条必须守住的规矩：
 *  1. 所有增益变化一律走 setTargetAtTime 做几毫秒平滑，绝不直接赋值 —— 否则必然爆音。
 *  2. 四轨必须用同一个 when 参数 start()，否则会有采样级错位（听起来像相位问题）。
 *  3. AudioContext 必须在用户手势里 resume()，浏览器自动播放策略挡着。
 */

const STEMS = [
  { id: 'vocals', zh: '人声', color: '#f4a261' },
  { id: 'drums',  zh: '鼓',   color: '#e76f51' },
  { id: 'bass',   zh: '贝斯', color: '#2a9d8f' },
  { id: 'other',  zh: '其他', color: '#9b8ade' },
];

const SEG_COLOR = { intro: '#3b4a63', verse: '#35526b', chorus: '#4a3f6b', bridge: '#5b3f52', outro: '#3f4a52' };
const GAIN_SMOOTH = 0.015;   // 秒。太短会爆音，太长会觉得滑块迟钝

const S = {
  ctx: null,
  master: null,
  masterAnalyser: null,
  tracks: {},          // id -> {buffer, gain, panner, eq, analyser, source, volume, pan, mute, solo, canvas, off, meter}
  duration: 0,
  playing: false,
  startedAt: 0,        // ctx.currentTime，本次 start 的时刻
  offset: 0,           // 暂停时保留的播放位置
  loop: false,
  gen: 0,              // 播放代次，用来忽略过期的 onended
  analysis: null,
  srcLabel: '合成演示曲',
};

const $ = (id) => document.getElementById(id);

/* ============================ 工具 ============================ */

function fmtTime(sec) {
  if (!isFinite(sec) || sec < 0) sec = 0;
  const m = Math.floor(sec / 60);
  const s = sec - m * 60;
  return `${m}:${s.toFixed(1).padStart(4, '0')}`;
}

let toastTimer = null;
function toast(msg, ms = 2600) {
  const el = $('toast');
  el.textContent = msg;
  el.hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => { el.hidden = true; }, ms);
}

/** 按设备像素比设置画布尺寸，避免高分屏下发虚。返回 **CSS 像素**宽高。
 *
 * 坑：canvas 的 width/height 属性存的是**位图像素数**，这里会被写成 CSS 尺寸 × DPR。
 * 所以绝不能反过来拿 getAttribute('height') 当 CSS 高度用 —— 每调用一次就翻一倍
 * （56 → 112 → 224），resize 几次画布就撑爆了。CSS 高度只在第一次读取并记进 dataset。
 */
function fitCanvas(canvas) {
  const dpr = window.devicePixelRatio || 1;
  if (!canvas.dataset.cssH) {
    canvas.dataset.cssH = canvas.getAttribute('height') || canvas.clientHeight || 100;
  }
  const h = parseFloat(canvas.dataset.cssH);
  const w = canvas.clientWidth || canvas.parentElement.clientWidth;
  canvas.style.height = h + 'px';
  canvas.width = Math.max(1, Math.round(w * dpr));
  canvas.height = Math.max(1, Math.round(h * dpr));
  const ctx = canvas.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  return { w, h, ctx };
}

/* ============================ 音频图 ============================ */

function ensureContext() {
  if (S.ctx) return S.ctx;
  S.ctx = new (window.AudioContext || window.webkitAudioContext)();

  S.master = S.ctx.createGain();
  S.master.gain.value = parseFloat($('masterVol').value);

  S.masterAnalyser = S.ctx.createAnalyser();
  S.masterAnalyser.fftSize = 2048;
  S.masterAnalyser.smoothingTimeConstant = 0.75;

  S.master.connect(S.masterAnalyser);
  S.masterAnalyser.connect(S.ctx.destination);
  return S.ctx;
}

function buildTrackNodes(id) {
  const ctx = S.ctx;
  const low  = ctx.createBiquadFilter(); low.type  = 'lowshelf';  low.frequency.value = 200;
  const mid  = ctx.createBiquadFilter(); mid.type  = 'peaking';   mid.frequency.value = 1000; mid.Q.value = 1;
  const high = ctx.createBiquadFilter(); high.type = 'highshelf'; high.frequency.value = 4000;

  const gain = ctx.createGain();
  const panner = ctx.createStereoPanner();
  const analyser = ctx.createAnalyser();
  analyser.fftSize = 1024;

  low.connect(mid); mid.connect(high); high.connect(gain);
  gain.connect(panner); panner.connect(analyser); analyser.connect(S.master);

  return { low, mid, high, gain, panner, analyser, input: low };
}

/** 独奏优先于静音：一旦有任意轨 solo，只有 solo 的轨出声。 */
function applyGains() {
  if (!S.ctx) return;
  const anySolo = STEMS.some((s) => S.tracks[s.id]?.solo);
  const now = S.ctx.currentTime;

  for (const { id } of STEMS) {
    const t = S.tracks[id];
    if (!t) continue;
    const audible = anySolo ? t.solo : !t.mute;
    t.gain.gain.setTargetAtTime(audible ? t.volume : 0, now, GAIN_SMOOTH);
    t.row.classList.toggle('silent', !audible);
  }
}

/* ============================ 播放控制 ============================ */

function currentPosition() {
  if (!S.playing) return S.offset;
  const p = S.offset + (S.ctx.currentTime - S.startedAt);
  return S.loop && S.duration > 0 ? p % S.duration : Math.min(p, S.duration);
}

async function play() {
  const ctx = ensureContext();
  if (ctx.state === 'suspended') await ctx.resume();   // 自动播放策略：必须在手势里
  if (S.playing) return;

  const gen = ++S.gen;
  // 统一的启动时刻：略微延后，给四条轨的调度留出余量，保证采样级同步
  const when = ctx.currentTime + 0.06;
  const startOffset = S.offset >= S.duration - 0.01 ? 0 : S.offset;

  for (const { id } of STEMS) {
    const t = S.tracks[id];
    if (!t?.buffer) continue;
    const src = ctx.createBufferSource();
    src.buffer = t.buffer;
    src.loop = S.loop;
    if (S.loop) { src.loopStart = 0; src.loopEnd = t.buffer.duration; }
    src.connect(t.input);
    src.start(when, startOffset);
    src.onended = () => { if (gen === S.gen && !S.loop) onReachedEnd(); };
    t.source = src;
  }

  S.offset = startOffset;
  S.startedAt = when;
  S.playing = true;
  $('playIcon').textContent = '⏸';
  applyGains();
}

function stopSources() {
  for (const { id } of STEMS) {
    const t = S.tracks[id];
    if (t?.source) {
      try { t.source.onended = null; t.source.stop(); } catch (_) {}
      t.source.disconnect();
      t.source = null;
    }
  }
}

function pause() {
  if (!S.playing) return;
  S.offset = currentPosition();
  S.gen++;
  stopSources();
  S.playing = false;
  $('playIcon').textContent = '▶';
}

function stop() {
  S.gen++;
  stopSources();
  S.playing = false;
  S.offset = 0;
  $('playIcon').textContent = '▶';
}

function onReachedEnd() {
  stop();
}

async function seek(t) {
  const pos = Math.max(0, Math.min(t, S.duration));
  if (S.playing) {
    S.gen++;
    stopSources();
    S.playing = false;
    S.offset = pos;
    await play();
  } else {
    S.offset = pos;
  }
}

/* ============================ 波形与时间轴 ============================ */

/** 把 AudioBuffer 画成静态波形，缓存到离屏画布，播放时只需 blit + 画播放头。 */
function renderWaveform(track) {
  const { w, h } = fitCanvas(track.canvas);
  const off = document.createElement('canvas');
  const dpr = window.devicePixelRatio || 1;
  off.width = Math.round(w * dpr);
  off.height = Math.round(h * dpr);
  const c = off.getContext('2d');
  c.setTransform(dpr, 0, 0, dpr, 0, 0);

  c.fillStyle = '#141922';
  c.fillRect(0, 0, w, h);

  const buf = track.buffer;
  if (!buf) { track.off = off; return; }

  const ch0 = buf.getChannelData(0);
  const ch1 = buf.numberOfChannels > 1 ? buf.getChannelData(1) : ch0;
  const step = ch0.length / w;
  const mid = h / 2;

  c.fillStyle = track.color;
  for (let x = 0; x < w; x++) {
    const a = Math.floor(x * step);
    const b = Math.min(Math.floor((x + 1) * step), ch0.length);
    let lo = 0, hi = 0;
    // 每像素取一段的最大最小值，比抽样更能反映真实包络
    for (let i = a; i < b; i++) {
      const v = (ch0[i] + ch1[i]) * 0.5;
      if (v < lo) lo = v;
      if (v > hi) hi = v;
    }
    const y1 = mid - hi * mid * 0.95;
    const y2 = mid - lo * mid * 0.95;
    c.fillRect(x, y1, 1, Math.max(1, y2 - y1));
  }

  // 中线
  c.fillStyle = '#ffffff14';
  c.fillRect(0, mid, w, 1);

  track.off = off;
}

let tlOff = null;
function renderTimeline() {
  const canvas = $('timeline');
  const { w, h } = fitCanvas(canvas);
  const off = document.createElement('canvas');
  const dpr = window.devicePixelRatio || 1;
  off.width = Math.round(w * dpr);
  off.height = Math.round(h * dpr);
  const c = off.getContext('2d');
  c.setTransform(dpr, 0, 0, dpr, 0, 0);

  c.fillStyle = '#1b212c';
  c.fillRect(0, 0, w, h);

  const D = S.duration || 1;
  const px = (t) => (t / D) * w;
  const A = S.analysis;

  if (!A) {
    c.fillStyle = '#8b96a8';
    c.font = '12px -apple-system, sans-serif';
    c.textAlign = 'center';
    c.fillText('无分析数据 —— P3 接入 beat_this / all-in-one 后自动填充', w / 2, h / 2 + 4);
    tlOff = off;
    return;
  }

  // 段落色块
  for (const seg of A.segments || []) {
    const x0 = px(seg.start), x1 = px(seg.end);
    c.fillStyle = SEG_COLOR[seg.label] || '#3b4a63';
    c.fillRect(x0, 0, Math.max(1, x1 - x0), 22);
    c.fillStyle = '#e6ebf2';
    c.font = '600 11px -apple-system, sans-serif';
    c.textAlign = 'left';
    if (x1 - x0 > 42) c.fillText(seg.label, x0 + 6, 15);
  }

  // 拍点（细）
  c.strokeStyle = '#ffffff18';
  c.lineWidth = 1;
  c.beginPath();
  for (const b of A.beats || []) {
    const x = Math.round(px(b)) + 0.5;
    c.moveTo(x, 26); c.lineTo(x, h - 20);
  }
  c.stroke();

  // 小节线（粗）
  c.strokeStyle = '#ffffff45';
  c.lineWidth = 1.5;
  c.beginPath();
  for (const b of A.downbeats || []) {
    const x = Math.round(px(b)) + 0.5;
    c.moveTo(x, 24); c.lineTo(x, h);
  }
  c.stroke();

  // 和弦标签
  c.font = '600 11.5px ui-monospace, SFMono-Regular, Menlo, monospace';
  c.textAlign = 'left';
  for (const ch of A.chords || []) {
    const x0 = px(ch.start), x1 = px(ch.end);
    if (x1 - x0 < 26) continue;
    c.fillStyle = '#4cc9f0';
    c.fillText(ch.label, x0 + 5, h - 22);
  }

  // 小节号
  c.font = '10px ui-monospace, monospace';
  c.fillStyle = '#8b96a8';
  (A.downbeats || []).forEach((b, i) => {
    const x = px(b);
    if (i % 1 === 0 && x < w - 12) c.fillText(String(i + 1), x + 4, h - 6);
  });

  tlOff = off;
}

/* ============================ 每帧绘制 ============================ */

const meterVals = {};

function drawPlayhead(c, w, h, pos) {
  const x = (pos / (S.duration || 1)) * w;
  c.fillStyle = '#ffffff';
  c.fillRect(x - 0.5, 0, 1.5, h);
  c.fillStyle = '#4cc9f0';
  c.fillRect(x - 0.5, 0, 1.5, 4);
}

/** 画一帧。
 *
 * 刻意和 rAF 循环解耦：页面在后台时 requestAnimationFrame 完全不触发
 * （切标签页、或本页被嵌在不可见的面板里都会这样），
 * 如果只在 rAF 里画，画布就是一片空白。所以加载完、resize 完、
 * 以及页面重新可见时都直接调用它一次。
 */
function drawFrame() {
  const pos = currentPosition();

  $('curTime').textContent = fmtTime(pos);

  // 当前小节
  if (S.analysis?.downbeats?.length) {
    let bar = 0;
    for (let i = 0; i < S.analysis.downbeats.length; i++) {
      if (S.analysis.downbeats[i] <= pos + 1e-6) bar = i + 1;
    }
    $('barVal').textContent = bar || '–';
  }

  // 时间轴
  const tl = $('timeline');
  if (tlOff) {
    const c = tl.getContext('2d');
    const dpr = window.devicePixelRatio || 1;
    const w = tl.width / dpr, h = tl.height / dpr;
    c.clearRect(0, 0, w, h);
    c.drawImage(tlOff, 0, 0, w, h);
    drawPlayhead(c, w, h, pos);
  }

  // 各轨波形 + 电平表
  for (const { id } of STEMS) {
    const t = S.tracks[id];
    if (!t?.off) continue;
    const c = t.canvas.getContext('2d');
    const dpr = window.devicePixelRatio || 1;
    const w = t.canvas.width / dpr, h = t.canvas.height / dpr;
    c.clearRect(0, 0, w, h);
    c.drawImage(t.off, 0, 0, w, h);
    drawPlayhead(c, w, h, pos);

    if (S.playing && t.analyser) {
      const buf = new Uint8Array(t.analyser.fftSize);
      t.analyser.getByteTimeDomainData(buf);
      let sum = 0;
      for (let i = 0; i < buf.length; i++) { const v = (buf[i] - 128) / 128; sum += v * v; }
      const rms = Math.sqrt(sum / buf.length);
      const db = 20 * Math.log10(rms + 1e-6);
      const pct = Math.max(0, Math.min(100, ((db + 60) / 60) * 100));
      meterVals[id] = Math.max(pct, (meterVals[id] || 0) * 0.86);   // 峰值保持，衰减更自然
    } else {
      meterVals[id] = (meterVals[id] || 0) * 0.8;
    }
    t.meter.style.height = meterVals[id].toFixed(1) + '%';
  }

  drawSpectrum();
}

function tick() {
  drawFrame();
  requestAnimationFrame(tick);
}

function drawSpectrum() {
  const canvas = $('spectrum');
  const dpr = window.devicePixelRatio || 1;
  const c = canvas.getContext('2d');
  const w = canvas.width / dpr, h = canvas.height / dpr;

  c.fillStyle = '#1b212c';
  c.fillRect(0, 0, w, h);

  if (!S.masterAnalyser) return;
  const n = S.masterAnalyser.frequencyBinCount;
  const data = new Uint8Array(n);
  S.masterAnalyser.getByteFrequencyData(data);

  // 对数频率轴，更符合听感（也更像 DAW 的频谱）
  const bars = Math.min(128, Math.floor(w / 5));
  const nyquist = (S.ctx?.sampleRate || 44100) / 2;
  const bw = w / bars;

  for (let i = 0; i < bars; i++) {
    const f0 = 20 * Math.pow(nyquist / 20, i / bars);
    const f1 = 20 * Math.pow(nyquist / 20, (i + 1) / bars);
    const b0 = Math.floor((f0 / nyquist) * n);
    const b1 = Math.max(b0 + 1, Math.floor((f1 / nyquist) * n));
    let peak = 0;
    for (let b = b0; b < b1 && b < n; b++) peak = Math.max(peak, data[b]);

    const bh = (peak / 255) * (h - 6);
    const g = c.createLinearGradient(0, h, 0, h - bh);
    g.addColorStop(0, '#2a9d8f');
    g.addColorStop(0.6, '#4cc9f0');
    g.addColorStop(1, '#9b8ade');
    c.fillStyle = g;
    c.fillRect(i * bw + 1, h - bh, bw - 2, bh);
  }
}

/* ============================ 界面构建 ============================ */

function buildTrackRows() {
  const wrap = $('tracks');
  wrap.innerHTML = '';

  for (const stem of STEMS) {
    const row = document.createElement('div');
    row.className = 'track';
    row.style.setProperty('--c', stem.color);
    row.innerHTML = `
      <div class="track-id">
        <div class="meter"><i></i></div>
        <div>
          <div class="track-name">${stem.id}</div>
          <div class="track-zh">${stem.zh}</div>
        </div>
      </div>
      <div class="wave-wrap"><canvas class="wave" height="56"></canvas></div>
      <div class="track-ctl">
        <div class="ctl-row">
          <button class="btn-sm mute-btn" aria-pressed="false">M</button>
          <button class="btn-sm solo-btn" aria-pressed="false">S</button>
          <label>音量</label>
          <input type="range" class="vol" min="0" max="1.5" step="0.01" value="1">
          <span class="val vol-val">100%</span>
        </div>
        <div class="ctl-row">
          <button class="btn-sm eq-toggle" aria-pressed="false">EQ</button>
          <label>声像</label>
          <input type="range" class="pan" min="-1" max="1" step="0.02" value="0">
          <span class="val pan-val">C</span>
        </div>
        <div class="eq">
          <div class="ctl-row"><label>低</label><input type="range" class="eq-low"  min="-12" max="12" step="0.5" value="0"><span class="val eq-low-val">0dB</span></div>
          <div class="ctl-row"><label>中</label><input type="range" class="eq-mid"  min="-12" max="12" step="0.5" value="0"><span class="val eq-mid-val">0dB</span></div>
          <div class="ctl-row"><label>高</label><input type="range" class="eq-high" min="-12" max="12" step="0.5" value="0"><span class="val eq-high-val">0dB</span></div>
        </div>`;
    wrap.appendChild(row);

    const t = S.tracks[stem.id] || (S.tracks[stem.id] = {});
    Object.assign(t, {
      id: stem.id, color: stem.color, row,
      canvas: row.querySelector('.wave'),
      meter: row.querySelector('.meter i'),
      volume: t.volume ?? 1, pan: t.pan ?? 0,
      mute: t.mute ?? false, solo: t.solo ?? false,
    });

    // --- 交互 ---
    const q = (s) => row.querySelector(s);

    q('.vol').addEventListener('input', (e) => {
      t.volume = parseFloat(e.target.value);
      q('.vol-val').textContent = Math.round(t.volume * 100) + '%';
      applyGains();
    });

    q('.pan').addEventListener('input', (e) => {
      t.pan = parseFloat(e.target.value);
      const p = t.pan;
      q('.pan-val').textContent = Math.abs(p) < 0.02 ? 'C'
        : (p < 0 ? 'L' : 'R') + Math.round(Math.abs(p) * 100);
      if (t.panner) t.panner.pan.setTargetAtTime(p, S.ctx.currentTime, GAIN_SMOOTH);
    });

    q('.mute-btn').addEventListener('click', (e) => {
      t.mute = !t.mute;
      e.target.setAttribute('aria-pressed', String(t.mute));
      applyGains();
    });

    q('.solo-btn').addEventListener('click', (e) => {
      t.solo = !t.solo;
      e.target.setAttribute('aria-pressed', String(t.solo));
      applyGains();
    });

    q('.eq-toggle').addEventListener('click', (e) => {
      const open = row.querySelector('.eq').classList.toggle('open');
      e.target.setAttribute('aria-pressed', String(open));
    });

    for (const [cls, node] of [['low', 'low'], ['mid', 'mid'], ['high', 'high']]) {
      q(`.eq-${cls}`).addEventListener('input', (e) => {
        const v = parseFloat(e.target.value);
        q(`.eq-${cls}-val`).textContent = (v > 0 ? '+' : '') + v + 'dB';
        if (t[node]) t[node].gain.setTargetAtTime(v, S.ctx.currentTime, GAIN_SMOOTH);
      });
    }

    t.canvas.addEventListener('click', (e) => {
      const r = t.canvas.getBoundingClientRect();
      seek(((e.clientX - r.left) / r.width) * S.duration);
    });
  }
}

/* ============================ 加载音频 ============================ */

async function decodeInto(id, arrayBuffer) {
  const ctx = ensureContext();
  const buffer = await ctx.decodeAudioData(arrayBuffer);
  const t = S.tracks[id];
  Object.assign(t, buildTrackNodes(id));
  t.buffer = buffer;
  t.panner.pan.value = t.pan;
  return buffer;
}

async function loadDemo() {
  $('loadState').hidden = false;
  $('loadState').textContent = '正在加载演示曲…';
  $('mixer').hidden = true;
  stop();
  ensureContext();

  try {
    const [analysis, ...bufs] = await Promise.all([
      fetch('demo/analysis.json').then((r) => (r.ok ? r.json() : null)).catch(() => null),
      ...STEMS.map((s) => fetch(`demo/${s.id}.mp3`).then((r) => {
        if (!r.ok) throw new Error(`缺少 demo/${s.id}.mp3`);
        return r.arrayBuffer();
      })),
    ]);

    buildTrackRows();
    for (let i = 0; i < STEMS.length; i++) await decodeInto(STEMS[i].id, bufs[i]);

    S.analysis = analysis;
    S.srcLabel = '合成演示曲（scripts/make_demo_stems.py）';
    finishLoad();
  } catch (err) {
    $('loadState').textContent =
      `加载失败：${err.message}\n先跑 python -m scripts.make_demo_stems 生成演示音频。`;
  }
}

async function loadUserFiles(fileList) {
  const files = Array.from(fileList);
  const alias = {
    vocals: ['vocal', '人声', 'voice'],
    drums: ['drum', '鼓', 'percussion'],
    bass: ['bass', '贝斯', '低音'],
    other: ['other', '其他', 'inst', 'accompaniment'],
  };

  const matched = {};
  for (const f of files) {
    const name = f.name.toLowerCase();
    for (const stem of STEMS) {
      if (matched[stem.id]) continue;
      if (name.includes(stem.id) || alias[stem.id].some((a) => name.includes(a))) {
        matched[stem.id] = f;
        break;
      }
    }
  }

  const hit = Object.keys(matched);
  if (hit.length === 0) {
    toast('文件名里要含 vocals / drums / bass / other —— 这正是 demucs 的默认输出命名', 5200);
    return;
  }

  $('loadState').hidden = false;
  $('loadState').textContent = '正在解码…';
  $('mixer').hidden = true;
  stop();
  ensureContext();
  buildTrackRows();

  for (const stem of STEMS) {
    const f = matched[stem.id];
    if (!f) { Object.assign(S.tracks[stem.id], buildTrackNodes(stem.id), { buffer: null }); continue; }
    await decodeInto(stem.id, await f.arrayBuffer());
  }

  S.analysis = null;                        // 用户文件没有分析真值，等 P3
  S.srcLabel = `本地文件（${hit.join(' / ')}）`;
  finishLoad();
  if (hit.length < 4) toast(`只匹配到 ${hit.length} 轨：${hit.join('、')}`, 4000);
}

function finishLoad() {
  S.duration = Math.max(...STEMS.map((s) => S.tracks[s.id]?.buffer?.duration || 0));
  S.offset = 0;

  $('srcName').textContent = S.srcLabel;
  $('durTime').textContent = fmtTime(S.duration);
  $('bpmVal').textContent = S.analysis?.bpm ?? '–';
  $('keyVal').textContent = S.analysis?.key ?? '–';
  $('sigVal').textContent = S.analysis?.time_signature ?? '–';
  $('barVal').textContent = '–';

  $('loadState').hidden = true;
  $('mixer').hidden = false;

  // 用 setTimeout 而不是 rAF：页面在后台时 rAF 永远不会触发，首屏就会是空白
  setTimeout(redrawAll, 0);

  renderAnalysisPanel();
  applyGains();
}

/** 重建所有离屏画布并立刻画一帧。加载完成、窗口尺寸变化、页面重新可见时调用。 */
function redrawAll() {
  for (const { id } of STEMS) if (S.tracks[id]?.buffer) renderWaveform(S.tracks[id]);
  renderTimeline();
  fitCanvas($('spectrum'));
  drawFrame();
}

/* ============================ 分析面板 ============================ */

function renderAnalysisPanel() {
  const grid = $('analysisGrid');
  const A = S.analysis;

  if (!A) {
    $('analysisSrc').textContent = '无数据';
    grid.innerHTML = '<div class="stat"><b>–</b><span>本地文件尚无分析结果</span></div>';
    $('analysisNote').textContent =
      'P3 接入 beat_this（拍点/小节线）、all-in-one（曲式结构）、madmom（和弦）、Essentia（调性）之后，这里会自动填充。';
    return;
  }

  $('analysisSrc').textContent = A.ground_truth ? '真值（合成曲，非估计）' : '模型估计';

  const chords = [...new Set((A.chords || []).map((c) => c.label))].join(' – ');
  const stats = [
    [A.bpm, 'BPM（速度）'],
    [A.key, '调性'],
    [A.time_signature, '拍号'],
    [(A.beats || []).length, '拍点数'],
    [(A.downbeats || []).length, '小节数（downbeat）'],
    [(A.segments || []).length, '曲式段落'],
    [chords || '–', '和弦进行'],
    [`${(A.duration || 0).toFixed(1)}s`, '时长'],
  ];
  grid.innerHTML = stats.map(([v, k]) => `<div class="stat"><b>${v}</b><span>${k}</span></div>`).join('');

  $('analysisNote').textContent = A.ground_truth
    ? '这段演示曲由脚本合成，所以上面每一项都是已知真值而非模型估计。P3 接入拍点/和弦/结构模型后，它将作为第一个「必须答对」的测试样例 —— 连自己生成的规整曲子都跟不准，就别谈真实音乐了。'
    : '由模型估计，存在误差。';
}

/* ============================ 项目状态面板 ============================ */

const ANCHOR_META = {
  silence: ['输出全零', '解析已知值，验证实现正确'],
  trivial: ['混音当每一轨', '下界'],
  htdemucs: ['Demucs v4', '复现官方 9.00 dB'],
  oracle: ['IRM 理想掩码', '掩码类方法的上界'],
};

async function loadDashboard() {
  const tbody = $('anchorTable').querySelector('tbody');
  const rows = [];

  for (const name of ['silence', 'trivial', 'htdemucs', 'oracle']) {
    try {
      // M1 之后优先读 MUSDB18-HQ 真实数据的结果，缺失时回落到 M0 的合成数据自检
      let r = await fetch(`../results/p1_${name}.json`);
      if (!r.ok) r = await fetch(`../results/m0_selfcheck_${name}.json`);
      if (!r.ok) continue;
      const d = await r.json();
      // M1 的真实数据结果优先看 cSDR 中位数聚合（museval 官方口径），
      // 合成数据的 M0 自检只有 uSDR
      const v = d.median?.cSDR?.mean ?? d.mean?.uSDR?.mean;
      if (v === undefined) continue;
      const [desc, role] = ANCHOR_META[name];
      rows.push(`<tr><td><code>${name}</code> ${desc}</td>
                     <td><b>${v.toFixed(2)} dB</b></td>
                     <td class="dim">${role}</td></tr>`);
    } catch (_) { /* 直接开文件时 fetch 会失败，忽略 */ }
  }

  tbody.innerHTML = rows.length
    ? rows.join('')
    : `<tr><td colspan="3" class="dim">需通过 HTTP 访问才能读取 results/（用 python -m http.server）</td></tr>`;
}

const MILESTONES = [
  ['M0', '工程地基 · 评测框架', 'done'],
  ['M1', '分离 baseline · htdemucs 8.80 dB', 'done'],
  ['M2', '前端最小可用 · 四轨混音台', 'done'],
  ['M3', '标签 L0→L4', ''],
  ['M4', '分离改进 A+B · 消融表', 'active'],
  ['M5', 'Stem-aware Tagging ★', ''],
  ['M6', '音乐分析 + 自动 Mashup', ''],
  ['M7', '检索 + 服务化', ''],
  ['M8', '蒸馏 + 作品集包装', ''],
];

function renderRoadmap() {
  $('roadmap').innerHTML = MILESTONES
    .map(([m, t, cls]) => `<li class="${cls}"><span class="dot"></span><span class="m">${m}</span>${t}</li>`)
    .join('');
}

/* ============================ 导出 WAV ============================ */

function encodeWav(buffer) {
  const nCh = buffer.numberOfChannels;
  const n = buffer.length;
  const sr = buffer.sampleRate;
  const bytes = 44 + n * nCh * 2;
  const view = new DataView(new ArrayBuffer(bytes));

  const str = (o, s) => { for (let i = 0; i < s.length; i++) view.setUint8(o + i, s.charCodeAt(i)); };
  str(0, 'RIFF'); view.setUint32(4, bytes - 8, true); str(8, 'WAVE');
  str(12, 'fmt '); view.setUint32(16, 16, true);
  view.setUint16(20, 1, true); view.setUint16(22, nCh, true);
  view.setUint32(24, sr, true); view.setUint32(28, sr * nCh * 2, true);
  view.setUint16(32, nCh * 2, true); view.setUint16(34, 16, true);
  str(36, 'data'); view.setUint32(40, n * nCh * 2, true);

  const chans = Array.from({ length: nCh }, (_, c) => buffer.getChannelData(c));
  let o = 44;
  for (let i = 0; i < n; i++) {
    for (let c = 0; c < nCh; c++) {
      const s = Math.max(-1, Math.min(1, chans[c][i]));
      view.setInt16(o, s < 0 ? s * 0x8000 : s * 0x7fff, true);
      o += 2;
    }
  }
  return new Blob([view], { type: 'audio/wav' });
}

async function exportWav() {
  if (!S.duration) return;
  const btn = $('exportBtn');
  btn.disabled = true;
  btn.textContent = '渲染中…';

  try {
    const sr = S.ctx.sampleRate;
    // OfflineAudioContext 以远超实时的速度离线渲染，20 秒的曲子通常 1 秒内出结果
    const off = new OfflineAudioContext(2, Math.ceil(S.duration * sr), sr);
    const master = off.createGain();
    master.gain.value = parseFloat($('masterVol').value);
    master.connect(off.destination);

    const anySolo = STEMS.some((s) => S.tracks[s.id]?.solo);

    for (const stem of STEMS) {
      const t = S.tracks[stem.id];
      if (!t?.buffer) continue;

      const src = off.createBufferSource(); src.buffer = t.buffer;
      const low  = off.createBiquadFilter(); low.type  = 'lowshelf';  low.frequency.value = 200;  low.gain.value  = t.low.gain.value;
      const mid  = off.createBiquadFilter(); mid.type  = 'peaking';   mid.frequency.value = 1000; mid.Q.value = 1; mid.gain.value = t.mid.gain.value;
      const high = off.createBiquadFilter(); high.type = 'highshelf'; high.frequency.value = 4000; high.gain.value = t.high.gain.value;
      const gain = off.createGain();
      const pan  = off.createStereoPanner(); pan.pan.value = t.pan;

      const audible = anySolo ? t.solo : !t.mute;
      gain.gain.value = audible ? t.volume : 0;

      src.connect(low); low.connect(mid); mid.connect(high);
      high.connect(gain); gain.connect(pan); pan.connect(master);
      src.start(0);
    }

    const rendered = await off.startRendering();
    const url = URL.createObjectURL(encodeWav(rendered));
    const a = document.createElement('a');
    a.href = url;
    a.download = 'mix.wav';
    a.click();
    setTimeout(() => URL.revokeObjectURL(url), 4000);
    toast('已导出 mix.wav（当前的音量 / 声像 / EQ / 独奏静音设置）');
  } catch (err) {
    toast('导出失败：' + err.message, 4000);
  } finally {
    btn.disabled = false;
    btn.textContent = '导出 WAV';
  }
}

/* ============================ 事件绑定 ============================ */

function bindGlobal() {
  $('playBtn').addEventListener('click', () => (S.playing ? pause() : play()));
  $('stopBtn').addEventListener('click', stop);

  $('loopBtn').addEventListener('click', async (e) => {
    S.loop = !S.loop;
    e.target.setAttribute('aria-pressed', String(S.loop));
    if (S.playing) { const p = currentPosition(); await seek(p); }   // 重建 source 才能改 loop
  });

  $('masterVol').addEventListener('input', (e) => {
    const v = parseFloat(e.target.value);
    $('masterVolVal').textContent = Math.round(v * 100) + '%';
    if (S.master) S.master.gain.setTargetAtTime(v, S.ctx.currentTime, GAIN_SMOOTH);
  });

  $('timeline').addEventListener('click', (e) => {
    const r = e.currentTarget.getBoundingClientRect();
    seek(((e.clientX - r.left) / r.width) * S.duration);
  });

  $('exportBtn').addEventListener('click', exportWav);
  $('fileInput').addEventListener('change', (e) => { if (e.target.files.length) loadUserFiles(e.target.files); });
  $('resetDemo').addEventListener('click', loadDemo);

  document.addEventListener('keydown', (e) => {
    if (e.target.matches('input, textarea')) return;
    if (e.code === 'Space') { e.preventDefault(); S.playing ? pause() : play(); }
    if (e.code === 'ArrowLeft')  seek(currentPosition() - 2);
    if (e.code === 'ArrowRight') seek(currentPosition() + 2);
    if (e.code === 'Home') seek(0);
  });

  let resizeTimer = null;
  window.addEventListener('resize', () => {
    clearTimeout(resizeTimer);
    resizeTimer = setTimeout(redrawAll, 140);
  });

  // 页面从后台回到前台时，rAF 刚恢复之前先补画一帧，避免看到空白画布
  document.addEventListener('visibilitychange', () => {
    if (!document.hidden) redrawAll();
  });
}

/* ============================ 启动 ============================ */

bindGlobal();
renderRoadmap();
loadDashboard();
loadDemo();
requestAnimationFrame(tick);
