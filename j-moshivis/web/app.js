const TARGET_SAMPLE_RATE = 24000;

const imageInput = document.getElementById("image-input");
const imageDrop = document.querySelector(".image-drop");
const preview = document.getElementById("preview");
const statusEl = document.getElementById("status");
const transcriptEl = document.getElementById("transcript");
const gateFill = document.getElementById("gate-fill");
const connectionDot = document.getElementById("connection-dot");
const connectButton = document.getElementById("connect-button");
const talkButton = document.getElementById("talk-button");
const resetButton = document.getElementById("reset-button");
const textTemp = document.getElementById("text-temp");
const audioTemp = document.getElementById("audio-temp");
const textTopk = document.getElementById("text-topk");
const audioTopk = document.getElementById("audio-topk");

const decoder = new TextDecoder("utf-8");

let imageFile = null;
let ws = null;
let ready = false;
let recording = false;
let audioContext = null;
let mediaStream = null;
let mediaSource = null;
let processor = null;
let silentGain = null;
let pendingPcm = new Float32Array(0);
let playbackTime = 0;
let frameSize = 1920;

function setStatus(text) {
  statusEl.textContent = text;
}

function setConnected(value) {
  ready = value;
  connectionDot.classList.toggle("ready", value);
  connectionDot.classList.toggle("error", false);
  talkButton.disabled = !value;
  resetButton.disabled = !value;
  connectButton.textContent = value ? "切断" : "接続";
}

function setError(text) {
  setStatus(text);
  connectionDot.classList.remove("ready");
  connectionDot.classList.add("error");
}

function appendText(piece) {
  transcriptEl.textContent += piece;
  transcriptEl.scrollTop = transcriptEl.scrollHeight;
}

function clearTranscript() {
  transcriptEl.textContent = "";
  gateFill.style.width = "0%";
}

function sendBinary(kind, payload) {
  if (!ws || ws.readyState !== WebSocket.OPEN) {
    return;
  }
  const body = payload
    ? payload instanceof ArrayBuffer
      ? new Uint8Array(payload)
      : new Uint8Array(payload.buffer, payload.byteOffset, payload.byteLength)
    : new Uint8Array(0);
  const message = new Uint8Array(1 + body.byteLength);
  message[0] = kind;
  message.set(body, 1);
  ws.send(message.buffer);
}

function websocketUrl() {
  const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
  const params = new URLSearchParams({
    text_temperature: textTemp.value,
    audio_temperature: audioTemp.value,
    text_topk: textTopk.value,
    audio_topk: audioTopk.value,
  });
  return `${protocol}//${window.location.host}/api/chat?${params.toString()}`;
}

async function ensureAudioContext() {
  if (!audioContext) {
    audioContext = new AudioContext();
  }
  if (audioContext.state !== "running") {
    await audioContext.resume();
  }
}

async function connect() {
  if (ws && ws.readyState === WebSocket.OPEN) {
    disconnect();
    return;
  }
  if (!imageFile) {
    setError("先に画像を選択してください。");
    return;
  }

  await ensureAudioContext();
  clearTranscript();
  setStatus("モデルへ接続中です。");
  setConnected(false);

  ws = new WebSocket(websocketUrl());
  ws.binaryType = "arraybuffer";

  ws.onopen = async () => {
    const imageBuffer = await imageFile.arrayBuffer();
    sendBinary(8, imageBuffer);
    setStatus("画像を送信しました。モデルの準備を待っています。");
  };

  ws.onmessage = (event) => {
    const buffer = event.data;
    const bytes = new Uint8Array(buffer);
    const kind = bytes[0];

    if (kind === 0) {
      const info = JSON.parse(decoder.decode(buffer.slice(1)));
      frameSize = info.frame_size || frameSize;
      playbackTime = audioContext.currentTime + 0.08;
      setConnected(true);
      setStatus("準備完了。話すボタンを押して画像について質問できます。");
    } else if (kind === 1) {
      playPcm(new Float32Array(buffer.slice(1)));
    } else if (kind === 3) {
      setStatus(decoder.decode(buffer.slice(1)));
    } else if (kind === 4) {
      setError(decoder.decode(buffer.slice(1)));
    } else if (kind === 7) {
      const gate = bytes[1] / 255;
      gateFill.style.width = `${Math.round(gate * 100)}%`;
      appendText(decoder.decode(buffer.slice(2)));
    }
  };

  ws.onerror = () => {
    setError("WebSocketでエラーが発生しました。");
  };

  ws.onclose = () => {
    stopRecording(false);
    setConnected(false);
    setStatus("接続を閉じました。");
  };
}

function disconnect() {
  stopRecording(false);
  if (ws) {
    ws.close();
  }
  ws = null;
  setConnected(false);
}

async function startRecording() {
  if (!ready) {
    return;
  }
  await ensureAudioContext();
  mediaStream = await navigator.mediaDevices.getUserMedia({
    audio: {
      channelCount: 1,
      echoCancellation: true,
      noiseSuppression: true,
      autoGainControl: true,
    },
  });
  pendingPcm = new Float32Array(0);

  mediaSource = audioContext.createMediaStreamSource(mediaStream);
  processor = audioContext.createScriptProcessor(4096, 1, 1);
  silentGain = audioContext.createGain();
  silentGain.gain.value = 0;

  processor.onaudioprocess = (event) => {
    if (!recording || !ready) {
      return;
    }
    const input = event.inputBuffer.getChannelData(0);
    const resampled = resample(input, audioContext.sampleRate, TARGET_SAMPLE_RATE);
    queuePcm(resampled);
  };

  mediaSource.connect(processor);
  processor.connect(silentGain);
  silentGain.connect(audioContext.destination);

  recording = true;
  talkButton.classList.add("recording");
  talkButton.textContent = "停止";
  setStatus("聞き取り中です。話し終えたら停止してください。");
}

function stopRecording(sendStop = true) {
  if (!recording && !processor && !mediaStream) {
    return;
  }
  recording = false;
  talkButton.classList.remove("recording");
  talkButton.textContent = "話す";

  if (processor) {
    processor.disconnect();
    processor.onaudioprocess = null;
  }
  if (mediaSource) {
    mediaSource.disconnect();
  }
  if (silentGain) {
    silentGain.disconnect();
  }
  if (mediaStream) {
    mediaStream.getTracks().forEach((track) => track.stop());
  }

  processor = null;
  mediaSource = null;
  silentGain = null;
  mediaStream = null;

  if (pendingPcm.length > 0) {
    const padded = new Float32Array(frameSize);
    padded.set(pendingPcm.subarray(0, Math.min(frameSize, pendingPcm.length)));
    sendBinary(1, padded);
    pendingPcm = new Float32Array(0);
  }

  if (sendStop && ready) {
    sendBinary(2);
    setStatus("応答を生成中です。");
  }
}

function queuePcm(samples) {
  pendingPcm = concatFloat32(pendingPcm, samples);
  while (pendingPcm.length >= frameSize) {
    const frame = pendingPcm.slice(0, frameSize);
    pendingPcm = pendingPcm.slice(frameSize);
    sendBinary(1, frame);
  }
}

function concatFloat32(a, b) {
  if (a.length === 0) {
    return new Float32Array(b);
  }
  const out = new Float32Array(a.length + b.length);
  out.set(a, 0);
  out.set(b, a.length);
  return out;
}

function resample(input, fromRate, toRate) {
  if (fromRate === toRate) {
    return new Float32Array(input);
  }
  const ratio = fromRate / toRate;
  const outLength = Math.floor(input.length / ratio);
  const out = new Float32Array(outLength);
  for (let i = 0; i < outLength; i += 1) {
    const pos = i * ratio;
    const left = Math.floor(pos);
    const right = Math.min(left + 1, input.length - 1);
    const frac = pos - left;
    out[i] = input[left] * (1 - frac) + input[right] * frac;
  }
  return out;
}

function playPcm(samples) {
  if (!audioContext || samples.length === 0) {
    return;
  }
  const buffer = audioContext.createBuffer(1, samples.length, TARGET_SAMPLE_RATE);
  buffer.copyToChannel(samples, 0);
  const source = audioContext.createBufferSource();
  source.buffer = buffer;
  source.connect(audioContext.destination);

  const now = audioContext.currentTime;
  const startAt = Math.max(now + 0.02, playbackTime);
  source.start(startAt);
  playbackTime = startAt + buffer.duration;
}

imageInput.addEventListener("change", async () => {
  imageFile = imageInput.files[0] || null;
  if (!imageFile) {
    return;
  }
  preview.src = URL.createObjectURL(imageFile);
  imageDrop.classList.add("has-image");
  setStatus("画像を選択しました。接続できます。");
  if (ready) {
    sendBinary(8, await imageFile.arrayBuffer());
  }
});

connectButton.addEventListener("click", connect);

talkButton.addEventListener("click", async () => {
  if (recording) {
    stopRecording(true);
  } else {
    await startRecording();
  }
});

resetButton.addEventListener("click", () => {
  clearTranscript();
  playbackTime = audioContext ? audioContext.currentTime + 0.05 : 0;
  sendBinary(9);
});

window.addEventListener("beforeunload", () => {
  disconnect();
});
