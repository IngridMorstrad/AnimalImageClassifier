"use strict";
// Vanilla JS, no build step (§6). Thumbnail grid grouped by label, confidence and
// search filters, a detail view drawing detection boxes on a <canvas>, and a label
// picker that POSTs a re-tag and refreshes in place.

const state = { label: null, minConf: 0, includeUnscored: false, q: "" };

async function json(url, opts) {
  const r = await fetch(url, opts);
  if (!r.ok) throw new Error((await r.json().catch(() => ({}))).error || r.statusText);
  return r.json();
}

async function loadRun() {
  const { run } = await json("/api/run");
  const el = document.getElementById("run-info");
  if (run) el.textContent = `${run.source_root || ""} — ${run.state} (${run.n_done || 0} done)`;
}

async function loadLabels() {
  const { labels } = await json("/api/labels");
  const ul = document.getElementById("label-list");
  ul.innerHTML = "";
  for (const l of labels) {
    const li = document.createElement("li");
    if (l.label === state.label) li.classList.add("active");
    const warn = l.files_on_disk !== l.count ? ` <span class="warn" title="run verify">⚠</span>` : "";
    li.innerHTML = `<span>${l.label}</span><span class="count">${l.count}${warn}</span>`;
    li.onclick = () => { state.label = state.label === l.label ? null : l.label; loadLabels(); loadImages(); };
    ul.appendChild(li);
  }
}

async function loadImages() {
  const p = new URLSearchParams();
  if (state.label) p.set("label", state.label);
  if (state.minConf > 0) p.set("min_conf", state.minConf);
  if (state.includeUnscored) p.set("include_unscored", "true");
  if (state.q) p.set("q", state.q);
  const data = await json("/api/images?" + p.toString());
  const grid = document.getElementById("grid");
  grid.innerHTML = "";
  for (const item of data.items) grid.appendChild(tile(item));
  const note = document.getElementById("unscored-note");
  if (data.unscored_excluded > 0) {
    note.textContent = `${data.unscored_excluded} images have no confidence score — show them`;
    note.onclick = () => { state.includeUnscored = true; document.getElementById("include-unscored").checked = true; loadImages(); };
  } else note.textContent = "";
}

function tile(item) {
  const div = document.createElement("div");
  div.className = "tile";
  const img = document.createElement("img");
  img.src = `/api/images/${item.sha256}/thumb`;
  img.onerror = () => {
    const ph = document.createElement("div");
    ph.className = "placeholder";
    ph.textContent = "no image";
    img.replaceWith(ph);
  };
  const cap = document.createElement("div");
  cap.className = "cap";
  const sp = item.species_common ? `<span class="sp">${item.species_common}</span>` : "";
  const conf = item.confidence != null ? ` ${(item.confidence * 100).toFixed(0)}%` : "";
  cap.innerHTML = `${item.label}${conf}<br>${sp}`;
  div.append(img, cap);
  div.onclick = () => openDetail(item);
  return div;
}

function openDetail(item) {
  const detail = document.getElementById("detail");
  detail.classList.remove("hidden");
  detail.innerHTML = `<span class="close">✕</span>`;
  detail.querySelector(".close").onclick = () => detail.classList.add("hidden");

  const canvas = document.createElement("canvas");
  detail.appendChild(canvas);
  const img = new Image();
  img.style.imageOrientation = "from-image";
  img.onload = () => {
    const scale = Math.min(900 / img.width, 600 / img.height, 1);
    canvas.width = img.width * scale;
    canvas.height = img.height * scale;
    const ctx = canvas.getContext("2d");
    ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
    // Boxes are in the stored (EXIF-transposed) pixel frame; scale to canvas.
    const sx = canvas.width / item.width, sy = canvas.height / item.height;
    for (const b of item.boxes || []) {
      ctx.strokeStyle = b.is_dominant ? "#5b9dd9" : "#e0a53f";
      ctx.lineWidth = 2;
      ctx.strokeRect(b.x0 * sx, b.y0 * sy, (b.x1 - b.x0) * sx, (b.y1 - b.y0) * sy);
      const cap = `${b.cls} ${(b.area_frac * 100).toFixed(1)}%` +
        (b.species_common ? ` ${b.species_common} (${b.species_rank || "?"})` : "");
      ctx.fillStyle = "rgba(0,0,0,.7)";
      ctx.fillRect(b.x0 * sx, b.y0 * sy - 16, ctx.measureText(cap).width + 8, 16);
      ctx.fillStyle = "#fff";
      ctx.fillText(cap, b.x0 * sx + 4, b.y0 * sy - 4);
    }
  };
  img.src = `/api/images/${item.sha256}/full`;

  const meta = document.createElement("div");
  meta.className = "meta";
  meta.innerHTML = `<strong>${item.label}</strong> — ${item.species_scientific || ""} ` +
    `blur ${item.blur_score != null ? item.blur_score.toFixed(0) : "?"}`;
  detail.appendChild(meta);

  const picker = document.createElement("div");
  picker.className = "picker";
  picker.innerHTML = `<input placeholder="new label" /><button>Re-tag</button>`;
  picker.querySelector("button").onclick = async () => {
    const label = picker.querySelector("input").value.trim();
    try {
      await json(`/api/images/${item.sha256}/label`, {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ label }),
      });
      detail.classList.add("hidden");
      loadLabels(); loadImages();
    } catch (e) { alert(e.message); }
  };
  detail.appendChild(picker);
}

document.getElementById("min-conf").oninput = (e) => {
  state.minConf = parseFloat(e.target.value);
  document.getElementById("min-conf-value").textContent = state.minConf.toFixed(2);
  loadImages();
};
document.getElementById("include-unscored").onchange = (e) => { state.includeUnscored = e.target.checked; loadImages(); };
document.getElementById("search").oninput = (e) => { state.q = e.target.value; loadImages(); };

loadRun(); loadLabels(); loadImages();



// --------------------------------------------------------------------------- //
// Review tab: the active-learning queue.
//
// Every animal the model could not confidently name, least confident first, with
// its own top guesses offered as one-click buttons. The fastest correct label is
// one the user only has to confirm; free text is there for the (common) case where
// the model has never seen the species at all.
// --------------------------------------------------------------------------- //

let reviewLabels = [];
const REVIEW_PAGE = 40;

// A card of hundreds is normal, so the queue is paged: 40 at a time, "load more"
// for the rest, and the badge always shows the true remaining total.
async function loadReview(append = false) {
  const list = document.getElementById("review-list");
  const offset = append ? list.querySelectorAll(".review-card").length : 0;
  const data = await json(`/api/review?limit=${REVIEW_PAGE}&offset=${offset}`);
  reviewLabels = data.known_labels || [];

  const badge = document.getElementById("review-count");
  badge.textContent = data.total ? String(data.total) : "";

  if (!append) list.innerHTML = "";
  list.querySelector(".more-wrap")?.remove();

  if (!data.items.length && !append) {
    list.innerHTML = `<p class="empty">Nothing to review — every animal is either
      confidently named or already labelled by you.</p>`;
    return;
  }
  for (const item of data.items) list.appendChild(reviewCard(item, data.review_below));

  const shown = list.querySelectorAll(".review-card").length;
  if (shown < data.total) {
    const wrap = document.createElement("div");
    wrap.className = "more-wrap";
    wrap.innerHTML = `<button class="more">Load more (${shown} of ${data.total} shown)</button>`;
    wrap.querySelector(".more").onclick = () => loadReview(true);
    list.appendChild(wrap);
  }
}

function reviewCard(item, threshold) {
  const card = document.createElement("div");
  card.className = "review-card";

  const img = document.createElement("img");
  img.src = `/api/images/${item.sha256}/thumb`;
  img.alt = "";
  img.onclick = () => openDetail(item);

  const body = document.createElement("div");
  body.className = "review-body";

  const conf = item.confidence == null
    ? `<span class="never">never scored</span> — no species model has seen this yet`
    : `model's best: <strong>${item.species_common || item.label}</strong>
       at ${(item.confidence * 100).toFixed(0)}% (below ${(threshold * 100).toFixed(0)}%)`;
  const head = document.createElement("p");
  head.className = "review-head";
  head.innerHTML = conf;
  body.appendChild(head);

  // One-click buttons for the model's own top guesses, best first.
  const dominant = (item.boxes || []).find((b) => b.is_dominant) || (item.boxes || [])[0];
  const suggestions = (dominant && dominant.candidates) ? dominant.candidates.slice(0, 3) : [];
  if (suggestions.length) {
    const row = document.createElement("div");
    row.className = "suggestions";
    for (const c of suggestions) {
      const b = document.createElement("button");
      b.className = "suggest";
      b.innerHTML = `${c.common} <span class="score">${(c.score * 100).toFixed(0)}%</span>`;
      b.onclick = () => applyLabel(item.sha256, slugify(c.common), card);
      row.appendChild(b);
    }
    body.appendChild(row);
  }

  // Free text + datalist of labels already known, for a species the model lacks.
  const form = document.createElement("div");
  form.className = "review-form";
  const listId = `labels-${item.sha256.slice(0, 8)}`;
  form.innerHTML = `
    <input list="${listId}" placeholder="type a species (e.g. lion)" />
    <datalist id="${listId}">${reviewLabels.map((l) => `<option value="${l}">`).join("")}</datalist>
    <button class="apply">Label</button>
    <button class="skip">Skip</button>`;
  const input = form.querySelector("input");
  const apply = () => {
    const value = slugify(input.value);
    if (value) applyLabel(item.sha256, value, card);
  };
  form.querySelector(".apply").onclick = apply;
  input.onkeydown = (e) => { if (e.key === "Enter") apply(); };
  form.querySelector(".skip").onclick = () => card.remove();
  body.appendChild(form);

  card.append(img, body);
  return card;
}

/** Mirror taxonomy.slug() closely enough for the input box: a legal label directory. */
function slugify(text) {
  return (text || "")
    .normalize("NFKD")
    .replace(/[\u0300-\u036f]/g, "")
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "_")
    .replace(/^_+|_+$/g, "")
    .slice(0, 64);
}

async function applyLabel(sha256, label, card) {
  try {
    await json(`/api/images/${sha256}/label`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ label, note: "labelled in review" }),
    });
    // Remove the card in place and decrement the badge, rather than reloading the
    // whole page — on a 500-photo queue a full refresh after every label would make
    // the UI unusable and lose your scroll position.
    card.remove();
    const badge = document.getElementById("review-count");
    const remaining = Math.max(0, (parseInt(badge.textContent, 10) || 1) - 1);
    badge.textContent = remaining ? String(remaining) : "";
    if (!document.querySelectorAll("#review-list .review-card").length) loadReview();
    loadLabels();
  } catch (e) {
    alert(`${e.message}\n\nIf this is a species the model has never seen, restart the GUI with --allow-new-labels.`);
  }
}

function showTab(which) {
  const browsing = which === "browse";
  document.getElementById("browse-pane").classList.toggle("hidden", !browsing);
  document.getElementById("review-pane").classList.toggle("hidden", browsing);
  document.getElementById("tab-browse").classList.toggle("active", browsing);
  document.getElementById("tab-review").classList.toggle("active", !browsing);
  if (!browsing) loadReview();
}

document.getElementById("tab-browse").onclick = () => showTab("browse");
document.getElementById("tab-review").onclick = () => showTab("review");

loadReview();
