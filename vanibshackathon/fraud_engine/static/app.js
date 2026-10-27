"use strict";

const $ = (selector) => document.querySelector(selector);
const elements = {
  activityCount: $("#activity-count"),
  activityEmpty: $("#activity-empty"),
  activityRows: $("#activity-rows"),
  activityTable: $("#activity-table-wrap"),
  activityTotal: $("#activity-total"),
  analyzeButton: $("#analyze-button"),
  connectionLabel: $("#connection-label"),
  cueCount: $("#cue-count"),
  cueList: $("#cue-list"),
  engineVersion: $("#engine-version"),
  form: $("#analysis-form"),
  formError: $("#form-error"),
  headerDot: $("#header-dot"),
  language: $("#language"),
  metricConnections: $("#metric-connections"),
  metricFrames: $("#metric-frames"),
  metricLatency: $("#metric-latency"),
  metricP95: $("#metric-p95"),
  metricSessions: $("#metric-sessions"),
  mitigationDetail: $("#mitigation-detail"),
  mitigationIcon: $("#mitigation-icon"),
  mitigationRow: $("#mitigation-row"),
  mitigationState: $("#mitigation-state"),
  mitigationTitle: $("#mitigation-title"),
  normalizedTranscript: $("#normalized-transcript"),
  rawJson: $("#raw-json"),
  refreshButton: $("#refresh-health"),
  riskSummary: $("#risk-summary"),
  scoreFill: $("#score-fill"),
  sidebarDot: $("#sidebar-dot"),
  sidebarStatus: $("#sidebar-status"),
  speakButton: $("#speak-button"),
  threatLevel: $("#threat-level"),
  threatScore: $("#threat-score"),
  transcript: $("#transcript"),
  transcriptCount: $("#transcript-count"),
  vectorList: $("#vector-list"),
  verdictContent: $("#verdict-content"),
  verdictEmpty: $("#verdict-empty"),
  verdictTime: $("#verdict-time"),
  voiceActions: $("#voice-actions"),
  voiceScript: $("#voice-script"),
  toast: $("#toast"),
};

const scenarios = {
  utility: {
    transcript: "Bijli cut ho jayegi 10 minute mein, QuickSupport apk download karo",
    language: "hi-IN",
    caller: "+919988776655",
  },
  authority: {
    transcript: "Main CBI officer bol raha hoon, digital arrest warrant hai, turant AnyDesk install karo",
    language: "hinglish",
    caller: "+919876543210",
  },
  safe: {
    transcript: "Bhaiya bijli ka bill bhar diya maine PhonePe se",
    language: "hi-IN",
    caller: "+919123456789",
  },
};

const vectors = [
  ["authority_impersonation", "Authority impersonation"],
  ["urgency_coercion", "Urgency & coercion"],
  ["remote_access_apk", "Remote access / APK"],
  ["credential_harvesting", "Credential harvesting"],
  ["synergy_bonus", "Synergy bonus"],
];

const state = {
  socket: null,
  sessionId: `sentinel-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`,
  connecting: false,
  retryTimer: null,
  retryDelay: 1000,
  healthTimer: null,
  results: [],
  latestVoiceScript: "",
  toastTimer: null,
};

function setConnectionStatus(status, label) {
  elements.connectionLabel.textContent = label;
  elements.sidebarStatus.textContent = label;
  for (const dot of [elements.headerDot, elements.sidebarDot]) {
    dot.classList.toggle("is-connected", status === "connected");
    dot.classList.toggle("is-error", status === "error");
  }
}

function connectSocket() {
  if (state.connecting || state.socket?.readyState === WebSocket.OPEN) return;
  state.connecting = true;
  setConnectionStatus("connecting", "Connecting");

  const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
  let socket;
  try {
    socket = new WebSocket(`${protocol}//${window.location.host}/ws/stream`);
  } catch (error) {
    state.connecting = false;
    setConnectionStatus("error", "Connection failed");
    showToast(`Could not open the threat engine connection: ${error.message}`, true);
    scheduleReconnect();
    return;
  }
  state.socket = socket;

  socket.addEventListener("open", () => {
    state.connecting = false;
    state.retryDelay = 1000;
    window.clearTimeout(state.retryTimer);
    setConnectionStatus("connected", "Engine connected");
  });
  socket.addEventListener("message", (event) => {
    let response;
    try {
      response = JSON.parse(event.data);
    } catch {
      if (event.data !== "PONG") showToast("The engine returned an unreadable response.", true);
      return;
    }

    if (response.status === "error") {
      elements.analyzeButton.disabled = false;
      elements.analyzeButton.querySelector("span").textContent = "Analyze signal";
      setAudioButtonLoading(false);
      const detail = typeof response.detail === "string"
        ? response.detail
        : JSON.stringify(response.detail);
      setFormError(`${response.error_code || "ENGINE_ERROR"}: ${detail}`);
      return;
    }
    if (response.threat_level) renderVerdict(response);
  });
  socket.addEventListener("error", () => {
    setConnectionStatus("error", "Connection issue");
  });
  socket.addEventListener("close", () => {
    state.connecting = false;
    setAudioButtonLoading(false);
    setConnectionStatus("error", "Reconnecting");
    scheduleReconnect();
  });
}

function scheduleReconnect() {
  window.clearTimeout(state.retryTimer);
  state.retryTimer = window.setTimeout(connectSocket, state.retryDelay);
  state.retryDelay = Math.min(state.retryDelay * 2, 15000);
}

async function refreshHealth(notifyOnError = false) {
  elements.refreshButton.classList.add("is-loading");
  try {
    const response = await fetch("/health", { headers: { Accept: "application/json" } });
    if (!response.ok) throw new Error(`Health endpoint returned ${response.status}`);
    const health = await response.json();
    elements.metricFrames.textContent = formatNumber(health.total_frames_processed);
    elements.metricLatency.textContent = formatDecimal(health.avg_latency_ms);
    elements.metricP95.textContent = formatDecimal(health.p95_latency_ms);
    elements.metricSessions.textContent = formatNumber(health.tracked_sessions);
    elements.metricConnections.textContent = formatNumber(health.active_ws_connections);
    elements.engineVersion.textContent = health.engine_version || "—";
    if (health.status !== "ok") {
      setConnectionStatus("error", `Engine ${health.status}`);
    } else if (state.socket?.readyState === WebSocket.OPEN) {
      setConnectionStatus("connected", "Engine connected");
    }
  } catch (error) {
    elements.metricFrames.textContent = "—";
    elements.metricLatency.textContent = "—";
    elements.metricP95.textContent = "—";
    elements.metricSessions.textContent = "—";
    elements.metricConnections.textContent = "—";
    if (state.socket?.readyState !== WebSocket.OPEN) {
      setConnectionStatus("error", "Engine unavailable · retrying");
    }
    if (notifyOnError) showToast(`Could not load engine health: ${error.message}`, true);
  } finally {
    elements.refreshButton.classList.remove("is-loading");
  }
}

function formatNumber(value) {
  return Number.isFinite(Number(value)) ? Number(value).toLocaleString() : "—";
}

function formatDecimal(value) {
  return Number.isFinite(Number(value)) ? Number(value).toFixed(3) : "—";
}

function setFormError(message) {
  elements.formError.textContent = message;
  elements.formError.hidden = !message;
}

function showToast(message, isError = false) {
  elements.toast.textContent = message;
  elements.toast.classList.toggle("error", isError);
  elements.toast.hidden = false;
  window.clearTimeout(state.toastTimer);
  state.toastTimer = window.setTimeout(() => { elements.toast.hidden = true; }, 4000);
}

function renderVerdict(verdict) {
  const level = String(verdict.threat_level || "SAFE").toLowerCase();
  const score = Math.max(0, Math.min(1, Number(verdict.composite_threat_score) || 0));
  const mitigation = verdict.mitigation || {};
  const labels = {
    authority_impersonation: "Authority impersonation",
    urgency_coercion: "Urgency & coercion",
    remote_access_apk: "Remote access / APK",
    credential_harvesting: "Credential harvesting",
    synergy_bonus: "Synergy bonus",
  };

  elements.verdictEmpty.hidden = true;
  elements.verdictContent.hidden = false;
  elements.verdictTime.textContent = new Date(Number(verdict.server_timestamp) * 1000).toLocaleTimeString([], {
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
  });
  elements.riskSummary.className = `risk-summary ${level}`;
  elements.threatLevel.textContent = verdict.threat_level || "—";
  elements.threatScore.textContent = score.toFixed(2);
  elements.scoreFill.style.width = `${Math.round(score * 100)}%`;
  elements.riskSummary.querySelector("#risk-detail").textContent =
    `${formatDecimal(verdict.latency_ms)} ms analysis latency`;

  const locked = Boolean(mitigation.lock_upi);
  elements.mitigationRow.classList.toggle("is-locked", locked);
  elements.mitigationIcon.textContent = locked ? "!" : mitigation.action === "WARN" ? "△" : "✓";
  elements.mitigationTitle.textContent = mitigation.action === "HALT_AND_FREEZE"
    ? "Halt and freeze"
    : mitigation.action === "WARN"
      ? "Warn user"
      : "Monitor only";
  elements.mitigationDetail.textContent = locked
    ? `UPI freeze directive · ${mitigation.cooling_off_minutes || 0} minute cooling-off`
    : "UPI transfers remain active";
  elements.mitigationState.textContent = locked ? "DIRECTIVE" : "ACTIVE";

  const cueTokens = Array.isArray(verdict.detected_cue_tokens) ? verdict.detected_cue_tokens : [];
  elements.cueCount.textContent = `${cueTokens.length} ${cueTokens.length === 1 ? "cue" : "cues"}`;
  elements.cueList.replaceChildren();
  if (cueTokens.length) {
    for (const token of cueTokens) {
      const chip = document.createElement("span");
      chip.className = "cue-chip";
      chip.textContent = token;
      elements.cueList.append(chip);
    }
  } else {
    const empty = document.createElement("span");
    empty.className = "muted-inline";
    empty.textContent = "No threat cues detected.";
    elements.cueList.append(empty);
  }

  elements.vectorList.replaceChildren();
  const scores = verdict.vector_scores || {};
  for (const [key, fallbackLabel] of vectors) {
    const value = Math.max(0, Math.min(1, Number(scores[key]) || 0));
    const row = document.createElement("div");
    row.className = "vector-row";
    row.innerHTML = `<span class="vector-name"></span><span class="vector-track"><span></span></span><span class="vector-value"></span>`;
    row.querySelector(".vector-name").textContent = labels[key] || fallbackLabel;
    row.querySelector(".vector-track span").style.width = `${Math.round(value * 100)}%`;
    row.querySelector(".vector-value").textContent = value.toFixed(2);
    elements.vectorList.append(row);
  }

  elements.normalizedTranscript.textContent = verdict.normalized_transcript || "No transcript returned.";
  elements.rawJson.textContent = JSON.stringify(verdict, null, 2);
  state.latestVoiceScript = mitigation.vernacular_voice_script || "";
  elements.voiceActions.hidden = !state.latestVoiceScript;
  elements.voiceScript.textContent = state.latestVoiceScript;
  elements.analyzeButton.disabled = false;
  elements.analyzeButton.querySelector("span").textContent = "Analyze signal";
  setAudioButtonLoading(false);
  setFormError("");
  refreshHealth();
  addActivity(verdict);
}

function addActivity(verdict) {
  const result = {
    time: new Date(Number(verdict.server_timestamp) * 1000).toLocaleTimeString([], {
      hour: "2-digit",
      minute: "2-digit",
      second: "2-digit",
    }),
    sessionId: verdict.session_id || "—",
    level: verdict.threat_level || "SAFE",
    score: Number(verdict.composite_threat_score) || 0,
    action: verdict.mitigation?.action || "MONITOR",
  };
  state.results.unshift(result);
  state.results = state.results.slice(0, 20);
  elements.activityRows.replaceChildren();
  for (const item of state.results) {
    const row = document.createElement("tr");
    row.innerHTML = "<td></td><td class=\"table-session\"></td><td><span class=\"table-threat\"></span></td><td></td><td></td>";
    row.cells[0].textContent = item.time;
    row.cells[1].textContent = item.sessionId;
    const badge = row.querySelector(".table-threat");
    badge.textContent = item.level;
    badge.classList.add(item.level.toLowerCase());
    row.cells[3].textContent = item.score.toFixed(2);
    row.cells[4].textContent = item.action;
    elements.activityRows.append(row);
  }
  elements.activityCount.textContent = String(state.results.length);
  elements.activityTotal.textContent = `${state.results.length} ${state.results.length === 1 ? "analysis" : "analyses"}`;
  elements.activityEmpty.hidden = true;
  elements.activityTable.hidden = false;
}

elements.form.addEventListener("submit", (event) => {
  event.preventDefault();
  const payload = elements.transcript.value.trim();
  if (!payload) {
    setFormError("Enter a transcript or message before analyzing.");
    elements.transcript.focus();
    return;
  }
  if (payload.length > 4000) {
    setFormError("Transcript must be 4,000 characters or fewer.");
    return;
  }
  if (!state.socket || state.socket.readyState !== WebSocket.OPEN) {
    setFormError("The threat engine is not connected yet. Reconnecting now—please try again shortly.");
    connectSocket();
    return;
  }

  setFormError("");
  elements.analyzeButton.disabled = true;
  elements.analyzeButton.querySelector("span").textContent = "Analyzing…";
  try {
    state.socket.send(JSON.stringify({
      session_id: state.sessionId,
      client_timestamp: Date.now() / 1000,
      stream_type: "TEXT_TRANSCRIPT",
      payload,
      metadata: {
        caller_number: $("#caller-number").value.trim() || "UNKNOWN",
        duration_seconds: 0,
        language: elements.language.value,
        is_in_contacts: false,
      },
    }));
  } catch (error) {
    elements.analyzeButton.disabled = false;
    elements.analyzeButton.querySelector("span").textContent = "Analyze signal";
    setFormError(`Could not send this transcript: ${error.message}`);
  }
});

function setAudioButtonLoading(isLoading) {
  const button = $("#audio-test-button");
  button.disabled = isLoading;
  button.querySelector("strong").textContent = isLoading
    ? "Sending audio frame…"
    : "Synthetic PCM audio frame";
}

$("#audio-test-button").addEventListener("click", () => {
  if (!state.socket || state.socket.readyState !== WebSocket.OPEN) {
    setFormError("The threat engine is not connected yet. Reconnecting now—please try again shortly.");
    connectSocket();
    return;
  }

  const pcmBytes = new Uint8Array(640);
  for (let index = 0; index < pcmBytes.length; index += 1) {
    pcmBytes[index] = index % 2 === 0 ? 0x10 : 0x27;
  }
  let binary = "";
  for (const byte of pcmBytes) binary += String.fromCharCode(byte);
  setFormError("");
  setAudioButtonLoading(true);
  try {
    state.socket.send(JSON.stringify({
      session_id: state.sessionId,
      client_timestamp: Date.now() / 1000,
      stream_type: "RAW_PCM_AUDIO",
      payload: btoa(binary),
      metadata: {
        caller_number: $("#caller-number").value.trim() || "UNKNOWN",
        duration_seconds: 0,
        language: elements.language.value,
        is_in_contacts: false,
      },
    }));
  } catch (error) {
    setAudioButtonLoading(false);
    setFormError(`Could not send this audio frame: ${error.message}`);
  }
});

$("#new-session-button").addEventListener("click", () => {
  state.sessionId = `sentinel-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
  showToast("Started a new analysis session. Previous page results are unchanged.");
});

for (const button of document.querySelectorAll("[data-scenario]")) {
  button.addEventListener("click", () => {
    const scenario = scenarios[button.dataset.scenario];
    if (!scenario) return;
    elements.transcript.value = scenario.transcript;
    elements.language.value = scenario.language;
    $("#caller-number").value = scenario.caller;
    updateTranscriptCount();
    setFormError("");
    elements.transcript.focus();
  });
}

function updateTranscriptCount() {
  elements.transcriptCount.textContent = `${elements.transcript.value.length} / 4000`;
}
elements.transcript.addEventListener("input", updateTranscriptCount);
elements.refreshButton.addEventListener("click", () => refreshHealth(true));
elements.speakButton.addEventListener("click", () => {
  if (!state.latestVoiceScript) return;
  if (!("speechSynthesis" in window)) {
    showToast("Speech playback is not supported by this browser.", true);
    return;
  }
  window.speechSynthesis.cancel();
  const utterance = new SpeechSynthesisUtterance(state.latestVoiceScript);
  utterance.lang = elements.language.value === "bho-IN" ? "hi-IN" : elements.language.value;
  window.speechSynthesis.speak(utterance);
});

for (const link of document.querySelectorAll(".nav-link")) {
  link.addEventListener("click", () => {
    document.querySelectorAll(".nav-link").forEach((item) => {
      item.classList.remove("active");
      item.removeAttribute("aria-current");
    });
    link.classList.add("active");
    link.setAttribute("aria-current", "page");
  });
}

window.addEventListener("beforeunload", () => {
  window.clearTimeout(state.retryTimer);
  window.clearInterval(state.healthTimer);
  if ("speechSynthesis" in window) window.speechSynthesis.cancel();
  state.socket?.close();
});

updateTranscriptCount();
connectSocket();
refreshHealth();
state.healthTimer = window.setInterval(refreshHealth, 10000);
