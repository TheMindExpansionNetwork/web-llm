import * as webllm from "@mlc-ai/web-llm";
import directorSpecData from "../artifacts/sonic_forage_director_v0/director_feature_spec.json";

type MusicPlan = {
  bpm: number;
  mood: string;
  layers: string[];
  next_variation: string;
  brightness?: number;
  pulse?: number;
  bass?: number;
  noise?: number;
  tension?: number;
  action?: string;
};

type DirectorSpec = {
  vocab: string[];
  actions: string[];
  moods: string[];
  layers: string[];
  thresholds: { layer_sigmoid: number };
};

type BciVibeSample = {
  focus?: number;
  excitement?: number;
  relaxation?: number;
  stress?: number;
  engagement?: number;
  alpha?: number;
  beta?: number;
  theta?: number;
  gamma?: number;
  quality?: number;
  source?: string;
};

const $ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;
const logEl = $("log");
const planEl = $("plan");
const audioStatus = $("audioStatus");
const llmStatus = $("llmStatus");

let audioCtx: AudioContext | null = null;
let master: GainNode | null = null;
let currentStop: (() => void) | null = null;
let loopTimer: number | null = null;
let engine: webllm.MLCEngineInterface | null = null;
let directorOrt: typeof import("onnxruntime-web") | null = null;
let directorSession: import("onnxruntime-web").InferenceSession | null = null;
let directorSpec: DirectorSpec | null = null;
let bciSocket: WebSocket | null = null;
let bciSimTimer: number | null = null;
let deck = 0;

const directorModelUrl = new URL("../artifacts/sonic_forage_director_v0/director_model.onnx", import.meta.url).toString();
const stopWords = new Set(["the", "and", "with", "now", "please", "make", "keep", "more", "less", "next", "bar", "loop", "signal", "voice", "command", "forage", "sonic"]);

function log(message: string) {
  const stamp = new Date().toLocaleTimeString();
  logEl.textContent = `[${stamp}] ${message}\n` + logEl.textContent;
}

function fallbackPlan(prompt: string): MusicPlan {
  const lower = prompt.toLowerCase();
  const bright = lower.includes("glassy") || lower.includes("shimmer") ? 0.82 : 0.52;
  const pulse = lower.includes("minimal") ? 0.28 : 0.55;
  return {
    bpm: lower.includes("fast") ? 126 : 92,
    mood: prompt.slice(0, 96),
    layers: ["warm drone", "glassy pluck", "soft echo pulse", "tape hiss"],
    next_variation: "raise shimmer slightly, keep bass controlled, no vocals",
    brightness: bright,
    pulse,
  };
}

function showPlan(plan: MusicPlan, source: string) {
  planEl.textContent = JSON.stringify({ source, ...plan }, null, 2);
}

const sigmoid = (x: number) => 1 / (1 + Math.exp(-x));

function vectorizePrompt(prompt: string, spec: DirectorSpec) {
  const features = new Float32Array(spec.vocab.length);
  const vocabIndex = new Map(spec.vocab.map((token, index) => [token, index]));
  const tokens = prompt.toLowerCase().match(/[a-z0-9]+/g) || [];
  for (const token of tokens) {
    if (stopWords.has(token)) continue;
    const index = vocabIndex.get(token);
    if (index !== undefined) features[index] = Math.min(1, features[index] + 0.5);
  }
  return features;
}

async function ensureDirector() {
  if (!directorOrt) directorOrt = await import("onnxruntime-web");
  if (!directorSpec) directorSpec = directorSpecData as DirectorSpec;
  if (!directorSession) {
    llmStatus.textContent = "loading tiny ONNX director";
    directorSession = await directorOrt.InferenceSession.create(directorModelUrl, {
      executionProviders: ["wasm"],
    });
    llmStatus.textContent = "tiny ONNX director ready";
  }
  return { session: directorSession, spec: directorSpec as DirectorSpec };
}

async function runDirectorModel() {
  await ensureAudio();
  const prompt = ($("prompt") as HTMLTextAreaElement).value;
  const { session, spec } = await ensureDirector();
  const features = vectorizePrompt(prompt, spec);
  if (!directorOrt) throw new Error("ONNX Runtime Web did not load");
  const feeds = { features: new directorOrt.Tensor("float32", features, [1, spec.vocab.length]) };
  const outputs = await session.run(feeds);
  const controls = Array.from(outputs.controls.data as Float32Array).map(sigmoid);
  const actionLogits = Array.from(outputs.action_logits.data as Float32Array);
  const moodLogits = Array.from(outputs.mood_logits.data as Float32Array);
  const layerScores = Array.from(outputs.layer_logits.data as Float32Array).map(sigmoid);
  const argmax = (values: number[]) => values.reduce((best, value, index) => value > values[best] ? index : best, 0);
  const layers = layerScores
    .map((score, index) => ({ score, layer: spec.layers[index] }))
    .filter(({ score }) => score > spec.thresholds.layer_sigmoid)
    .sort((a, b) => b.score - a.score)
    .slice(0, 5)
    .map(({ layer }) => layer);
  const plan: MusicPlan = {
    bpm: Math.round(64 + controls[0] * 86),
    mood: spec.moods[argmax(moodLogits)],
    action: spec.actions[argmax(actionLogits)],
    layers,
    next_variation: `${spec.actions[argmax(actionLogits)]} next bar, keep ${spec.moods[argmax(moodLogits)]} identity`,
    brightness: Number(controls[1].toFixed(3)),
    pulse: Number(controls[2].toFixed(3)),
    bass: Number(controls[3].toFixed(3)),
    noise: Number(controls[4].toFixed(3)),
    tension: Number(controls[5].toFixed(3)),
  };
  showPlan(plan, "tiny ONNX realtime director");
  schedulePlan(plan);
  log(`Tiny director predicted ${plan.action}/${plan.mood} locally via ONNX Runtime Web.`);
}

async function checkCompat() {
  const gpu = "gpu" in navigator;
  $("gpuStatus").innerHTML = gpu
    ? '<span class="ok">available</span>'
    : '<span class="bad">unavailable in this browser</span>';
  try {
    const estimate = await navigator.storage?.estimate?.();
    if (estimate?.quota) {
      const used = ((estimate.usage || 0) / 1024 / 1024 / 1024).toFixed(2);
      const quota = (estimate.quota / 1024 / 1024 / 1024).toFixed(2);
      $("storageStatus").textContent = `${used} GB used / ${quota} GB quota`;
    } else {
      $("storageStatus").textContent = "storage estimate unavailable";
    }
  } catch (err) {
    $("storageStatus").textContent = `storage check failed: ${String(err)}`;
  }
}

async function ensureAudio() {
  if (!audioCtx) {
    audioCtx = new AudioContext();
    master = audioCtx.createGain();
    master.gain.value = 0.72;
    master.connect(audioCtx.destination);
  }
  if (audioCtx.state !== "running") {
    await Promise.race([
      audioCtx.resume(),
      new Promise<void>((resolve) => window.setTimeout(resolve, 700)),
    ]);
  }
}

function createNoiseBuffer(ctx: AudioContext, seconds: number) {
  const buffer = ctx.createBuffer(1, Math.max(1, Math.floor(ctx.sampleRate * seconds)), ctx.sampleRate);
  const data = buffer.getChannelData(0);
  let v = 0;
  for (let i = 0; i < data.length; i++) {
    v = v * 0.985 + (Math.random() * 2 - 1) * 0.015;
    data[i] = v;
  }
  return buffer;
}

function stopLoop() {
  if (loopTimer) window.clearTimeout(loopTimer);
  loopTimer = null;
  if (currentStop) currentStop();
  currentStop = null;
  audioStatus.textContent = "stopped";
}

function schedulePlan(plan: MusicPlan) {
  if (!audioCtx || !master) return;
  stopLoop();
  const ctx = audioCtx;
  const duration = 12;
  const crossfade = 3.5;

  const runDeck = () => {
    if (!master) return;
    const now = ctx.currentTime + 0.05;
    const out = ctx.createGain();
    out.gain.setValueAtTime(0, now);
    out.gain.linearRampToValueAtTime(1, now + crossfade);
    out.gain.setValueAtTime(1, now + duration - crossfade);
    out.gain.linearRampToValueAtTime(0, now + duration);
    out.connect(master);

    const baseFreq = 48 + (plan.bpm - 80) * 0.3;
    const drone = ctx.createOscillator();
    drone.type = "sine";
    drone.frequency.setValueAtTime(baseFreq, now);
    const droneGain = ctx.createGain();
    droneGain.gain.value = 0.12 + 0.24 * (plan.bass ?? 0.5);
    drone.connect(droneGain).connect(out);
    drone.start(now); drone.stop(now + duration + 0.1);

    const shimmer = ctx.createOscillator();
    shimmer.type = "triangle";
    shimmer.frequency.setValueAtTime(baseFreq * (plan.brightness && plan.brightness > 0.7 ? 7.5 : 5.0), now);
    const shimmerGain = ctx.createGain();
    shimmerGain.gain.setValueAtTime(0.0, now);
    for (let t = 0; t < duration; t += 1.5) {
      const tt = now + t;
      shimmerGain.gain.linearRampToValueAtTime(0.12, tt + 0.08);
      shimmerGain.gain.exponentialRampToValueAtTime(0.002, tt + 1.1);
    }
    shimmer.connect(shimmerGain).connect(out);
    shimmer.start(now); shimmer.stop(now + duration + 0.1);

    const pulse = ctx.createOscillator();
    pulse.type = "square";
    pulse.frequency.setValueAtTime(baseFreq * 2, now);
    const pulseGain = ctx.createGain();
    pulseGain.gain.value = Math.max(0.02, Math.min(0.16, plan.pulse || 0.08));
    pulse.connect(pulseGain).connect(out);
    pulse.start(now); pulse.stop(now + duration + 0.1);

    const noise = ctx.createBufferSource();
    noise.buffer = createNoiseBuffer(ctx, duration + 0.2);
    const filter = ctx.createBiquadFilter();
    filter.type = "lowpass";
    filter.frequency.value = 900 + 3600 * (plan.brightness || 0.5);
    const noiseGain = ctx.createGain();
    noiseGain.gain.value = 0.012 + 0.07 * (plan.noise ?? 0.3);
    noise.connect(filter).connect(noiseGain).connect(out);
    noise.start(now); noise.stop(now + duration + 0.1);

    const localDeck = ++deck;
    audioStatus.textContent = `playing local deck ${localDeck}, crossfade ${crossfade}s`;
    currentStop = () => {
      try { drone.stop(); shimmer.stop(); pulse.stop(); noise.stop(); } catch (_) { /* already stopped */ }
      out.disconnect();
    };
    loopTimer = window.setTimeout(runDeck, Math.max(1, duration - crossfade) * 1000);
  };
  runDeck();
}

async function renderFallback() {
  await ensureAudio();
  const prompt = $("prompt") as HTMLTextAreaElement;
  const plan = fallbackPlan(prompt.value);
  showPlan(plan, "local procedural fallback");
  schedulePlan(plan);
  log("Rendered local Web Audio plan without endpoint/model download.");
}

async function loadWebLLMAndPlan() {
  const model = ($("model") as HTMLSelectElement).value;
  const prompt = ($("prompt") as HTMLTextAreaElement).value;
  if (!("gpu" in navigator)) {
    log("WebGPU unavailable here; rendering local fallback instead. On iPhone, test Safari/WebGPU over HTTPS.");
    await renderFallback();
    return;
  }
  await ensureAudio();
  llmStatus.textContent = `loading ${model}`;
  engine = await webllm.CreateMLCEngine(
    model,
    {
      initProgressCallback: (report: webllm.InitProgressReport) => {
        llmStatus.textContent = report.text;
        log(report.text);
      },
      logLevel: "INFO",
    },
    { context_window_size: 2048 },
  );
  llmStatus.textContent = `${model} ready`;
  const completion = await engine.chat.completions.create({
    messages: [
      { role: "system", content: "Return only compact JSON for a Web Audio music plan. Keys: bpm number, mood string, layers string[], next_variation string, brightness number 0-1, pulse number 0-1." },
      { role: "user", content: prompt },
    ],
    temperature: 0.4,
    max_tokens: 180,
  });
  const text = completion.choices[0]?.message?.content || "{}";
  let plan: MusicPlan;
  try {
    const match = text.match(/\{[\s\S]*\}/);
    plan = { ...fallbackPlan(prompt), ...(JSON.parse(match ? match[0] : text) as Partial<MusicPlan>) };
  } catch (err) {
    log(`JSON parse failed, using fallback plan: ${String(err)}`);
    plan = fallbackPlan(prompt);
  }
  showPlan(plan, "WebLLM");
  schedulePlan(plan);
  log("WebLLM generated a plan and Web Audio rendered it locally.");
}

function sfx(kind: string) {
  if (!audioCtx || !master) return;
  const ctx = audioCtx;
  const now = ctx.currentTime;
  const osc = ctx.createOscillator();
  const gain = ctx.createGain();
  osc.type = kind === "impact" ? "sine" : "triangle";
  osc.frequency.setValueAtTime(kind === "chirp" ? 880 : kind === "rise" ? 180 : 72, now);
  osc.frequency.exponentialRampToValueAtTime(kind === "rise" ? 1300 : kind === "chirp" ? 2200 : 45, now + (kind === "rise" ? 1.4 : 0.45));
  gain.gain.setValueAtTime(0.0001, now);
  gain.gain.exponentialRampToValueAtTime(kind === "impact" ? 0.55 : 0.24, now + 0.04);
  gain.gain.exponentialRampToValueAtTime(0.0001, now + (kind === "rise" ? 1.6 : 0.55));
  osc.connect(gain).connect(master);
  osc.start(now); osc.stop(now + 1.8);
  log(`Played local SFX: ${kind}`);
}

function clamp01(value: number | undefined, fallback = 0.5) {
  if (typeof value !== "number" || Number.isNaN(value)) return fallback;
  return Math.max(0, Math.min(1, value));
}

function bciSampleToPlan(sample: BciVibeSample): MusicPlan {
  const focus = clamp01(sample.focus ?? sample.engagement, 0.5);
  const excitement = clamp01(sample.excitement ?? sample.beta, 0.45);
  const relaxation = clamp01(sample.relaxation ?? sample.alpha, 0.5);
  const stress = clamp01(sample.stress ?? sample.gamma, 0.25);
  const theta = clamp01(sample.theta, 0.35);
  const bpm = Math.round(72 + excitement * 58 - relaxation * 18 + stress * 20);
  const mood = stress > 0.66
    ? "urgent wolf edge"
    : relaxation > 0.64
      ? "calm wolf drift"
      : focus > 0.62
        ? "locked-in wolf hunt"
        : "curious wolf scan";
  const layers = [
    relaxation > 0.55 ? "warm drone" : "sub bass",
    focus > 0.6 ? "glassy pluck" : "soft echo pulse",
    excitement > 0.58 ? "portal riser" : "tape hiss",
    stress > 0.55 ? "radio static" : "shimmer pad",
  ];
  return {
    bpm,
    mood,
    action: stress > 0.65 ? "calm" : excitement > 0.65 ? "drop" : focus > 0.6 ? "deepen" : "wander",
    layers,
    next_variation: `BCI vibe source=${sample.source || "unknown"}; focus=${focus.toFixed(2)} excitement=${excitement.toFixed(2)} relaxation=${relaxation.toFixed(2)} stress=${stress.toFixed(2)}`,
    brightness: Number((0.25 + focus * 0.45 + excitement * 0.25).toFixed(3)),
    pulse: Number((0.1 + excitement * 0.75 + stress * 0.15).toFixed(3)),
    bass: Number((0.25 + relaxation * 0.45 + theta * 0.2).toFixed(3)),
    noise: Number((0.05 + stress * 0.65).toFixed(3)),
    tension: Number((stress * 0.75 + excitement * 0.25).toFixed(3)),
  };
}

async function applyBciSample(sample: BciVibeSample) {
  await ensureAudio();
  const plan = bciSampleToPlan(sample);
  showPlan(plan, "EPOC X / BCI vibe bridge");
  schedulePlan(plan);
  $("bciStatus").textContent = `${sample.source || "bridge"}: ${plan.mood}`;
  log(`BCI vibe mapped to ${plan.action}/${plan.mood}.`);
}

function connectBciBridge() {
  if (bciSimTimer) window.clearInterval(bciSimTimer);
  bciSimTimer = null;
  if (bciSocket) bciSocket.close();
  const url = ($("bciBridgeUrl") as HTMLInputElement).value.trim() || "ws://127.0.0.1:8765";
  $("bciStatus").textContent = `connecting ${url}`;
  bciSocket = new WebSocket(url);
  bciSocket.addEventListener("open", () => {
    $("bciStatus").textContent = `connected ${url}`;
    log(`Connected BCI bridge at ${url}.`);
  });
  bciSocket.addEventListener("message", (event) => {
    try {
      void applyBciSample(JSON.parse(String(event.data)) as BciVibeSample);
    } catch (err) {
      log(`BCI bridge message ignored: ${String(err)}`);
    }
  });
  bciSocket.addEventListener("close", () => { $("bciStatus").textContent = "bridge closed"; });
  bciSocket.addEventListener("error", () => { $("bciStatus").textContent = "bridge error"; });
}

function simulateBciVibe() {
  if (bciSocket) bciSocket.close();
  if (bciSimTimer) window.clearInterval(bciSimTimer);
  let t = 0;
  const tick = () => {
    t += 0.18;
    void applyBciSample({
      source: "simulated EPOC X vibe",
      focus: 0.5 + 0.35 * Math.sin(t * 0.7),
      excitement: 0.48 + 0.32 * Math.sin(t * 1.1 + 0.6),
      relaxation: 0.5 + 0.3 * Math.sin(t * 0.45 + 2.2),
      stress: 0.2 + 0.18 * Math.max(0, Math.sin(t * 0.9 + 1.4)),
      theta: 0.42 + 0.18 * Math.sin(t * 0.38),
      quality: 1,
    });
  };
  tick();
  bciSimTimer = window.setInterval(tick, 9000);
  $("bciStatus").textContent = "simulating BCI vibe";
}

function boot() {
  try {
    $("renderLocal").addEventListener("click", renderFallback);
    $("runDirector").addEventListener("click", runDirectorModel);
    $("loadLLM").addEventListener("click", loadWebLLMAndPlan);
    $("stop").addEventListener("click", stopLoop);
    $("connectBci").addEventListener("click", connectBciBridge);
    $("simulateBci").addEventListener("click", simulateBciVibe);
    document.querySelectorAll<HTMLButtonElement>(".sfx").forEach((button) => {
      button.addEventListener("click", async () => { await ensureAudio(); sfx(button.dataset.kind || "chirp"); });
    });
    $("speak").addEventListener("click", () => {
      const text = "Sonic Forage local browser lab is running. Web Audio is rendering the loop locally.";
      speechSynthesis.cancel();
      speechSynthesis.speak(new SpeechSynthesisUtterance(text));
      log("Spoke local SpeechSynthesis fallback caption.");
    });
    Object.assign(window, { __sonicForageSmoke: { renderFallback, runDirectorModel, simulateBciVibe, stopLoop } });
    $("bootStatus").textContent = "handlers attached";
    log("Sonic Forage app booted; handlers attached.");
    void checkCompat();
  } catch (err) {
    $("bootStatus").textContent = `boot failed: ${String(err)}`;
    console.error(err);
  }
}

if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", boot, { once: true });
} else {
  boot();
}
