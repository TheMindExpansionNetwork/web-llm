# iPhone WebGPU WebLLM Lane

This fork is intended to keep a quick path for running WebLLM directly in an iPhone browser.

## Core finding

WebLLM runs LLM inference in the browser using WebGPU and WebAssembly. Since Safari 26 / iOS 26, Apple ships WebGPU support on iOS/iPadOS, so modern iPhones can run WebLLM locally in Safari or Home Screen web apps without a server-side inference API.

Authoritative references checked:

- WebLLM repo/docs: https://github.com/mlc-ai/web-llm and https://webllm.mlc.ai/docs/
- WebLLM demo: https://chat.webllm.ai/
- Apple Safari 26 release notes: WebGPU support added for iOS 26/iPadOS 26/macOS/visionOS.
- WebKit Safari 26 feature post: WebGPU now ships across Apple platforms.

## iPhone requirements

- iOS 26+ / Safari 26+ for built-in WebGPU.
- Open over HTTPS, not `file://`. Local LAN testing should use an HTTPS tunnel or deploy to Pages/Vercel/Netlify/Cloudflare Pages.
- Test WebGPU first at https://webgpureport.org/ from the target iPhone.
- Expect first load to download hundreds of MB to multiple GB depending on model. Keep Wi-Fi on.
- Keep the page foregrounded while the model downloads/initializes; iOS may suspend background tabs.
- Add to Home Screen after validation; Safari 26 allows any site to become a web app on iOS/iPadOS.

## Recommended first models for iPhone

Prefer smallest prebuilt WebLLM models first:

- `Llama-3.2-1B-Instruct-q4f16_1-MLC` — roughly 879 MB VRAM in WebLLM config; best first smoke target.
- `Llama-3.2-1B-Instruct-q4f32_1-MLC` — roughly 1129 MB VRAM; slightly heavier.
- `Qwen3-4B-q4f16_1-MLC` — useful Qwen thinking/non-thinking demo, but likely too heavy for the first iPhone smoke.
- Avoid 7B/8B models for first iPhone tests; they may work on desktops but are not the first mobile target.

## Fast local smoke test from this fork

```bash
cd /opt/data/workspace/github-forks/web-llm/examples/get-started
npm install
npm start
# Parcel serves on http://localhost:8888 by default.
```

For iPhone testing, expose the dev server through HTTPS:

```bash
# Option A: deploy built static app
npm run build
# upload examples/get-started/lib/ to Pages/Vercel/Netlify/Cloudflare Pages

# Option B: use an HTTPS tunnel for temporary local testing
# cloudflared tunnel --url http://localhost:8888
```

Then open the HTTPS URL on the iPhone.

## Minimal app pattern

```ts
import * as webllm from "@mlc-ai/web-llm";

const selectedModel = "Llama-3.2-1B-Instruct-q4f16_1-MLC";

const engine = await webllm.CreateMLCEngine(selectedModel, {
  initProgressCallback: (report) => {
    console.log(report.text);
  },
  logLevel: "INFO",
}, {
  context_window_size: 2048,
});

const chunks = await engine.chat.completions.create({
  stream: true,
  stream_options: { include_usage: true },
  messages: [{ role: "user", content: "Say hello from local iPhone WebGPU." }],
  max_tokens: 128,
});

let text = "";
for await (const chunk of chunks) {
  text += chunk.choices[0]?.delta?.content || "";
  console.log(text);
}
```

## iPhone UX notes

- Show a compatibility panel before trying to load: `navigator.gpu`, user agent, device memory if available, and storage estimate from `navigator.storage.estimate()`.
- Show model download/progress text from `initProgressCallback`.
- Provide a “small / medium / desktop” model selector with the 1B model as default.
- Provide a cache-reset button that calls WebLLM cache-management helpers or guides the user to Safari site-data cleanup.
- Use a web worker for production UI responsiveness.

## Remote GPU browser fallback

If the target iPhone is older than iOS 26 or WebGPU fails:

1. Run the app in a remote GPU desktop/browser environment.
2. Stream the browser to the iPhone with noVNC/WebRTC/remote desktop.
3. Or serve a normal API-backed chat endpoint and keep WebLLM as the local/offline path.

This fallback is not the same as local iPhone inference; label it clearly.

## Verification checklist

- [ ] `webgpureport.org` shows WebGPU available on the iPhone.
- [ ] Demo loads over HTTPS.
- [ ] `navigator.gpu` compatibility check passes.
- [ ] 1B model downloads and initializes.
- [ ] A short streaming response appears.
- [ ] Refresh reuses browser cache/OPFS rather than fully redownloading.
- [ ] Home Screen web app launch still works.
