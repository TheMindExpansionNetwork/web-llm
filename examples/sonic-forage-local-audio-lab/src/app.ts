import * as webllm from "@mlc-ai/web-llm";

type MusicPlan = {
  bpm: number;
  mood: string;
  layers: string[];
  next_variation: string;
  brightness?: number;
  pulse?: number;
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
let deck = 0;

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
  if (audioCtx.state !== "running") await audioCtx.resume();
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
    droneGain.gain.value = 0.24;
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
    noiseGain.gain.value = 0.035;
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

$("renderLocal").addEventListener("click", renderFallback);
$("loadLLM").addEventListener("click", loadWebLLMAndPlan);
$("stop").addEventListener("click", stopLoop);
document.querySelectorAll<HTMLButtonElement>(".sfx").forEach((button) => {
  button.addEventListener("click", async () => { await ensureAudio(); sfx(button.dataset.kind || "chirp"); });
});
$("speak").addEventListener("click", () => {
  const text = "Sonic Forage local browser lab is running. Web Audio is rendering the loop locally.";
  speechSynthesis.cancel();
  speechSynthesis.speak(new SpeechSynthesisUtterance(text));
  log("Spoke local SpeechSynthesis fallback caption.");
});

checkCompat();
