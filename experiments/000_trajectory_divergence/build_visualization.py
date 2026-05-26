from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd


HERE = Path(__file__).parent
RESULTS_PATH = HERE / "results_1b" / "generations.csv"
OUTPUT_PATH = HERE / "visualization.html"


def normalize_words(text: str) -> list[str]:
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return [w for w in text.split() if w]


def prefix_tokens(text: str, n: int = 18) -> str:
    return " ".join(normalize_words(text)[:n])


def jaccard(a: str, b: str) -> float:
    aw = set(normalize_words(a))
    bw = set(normalize_words(b))
    if not aw and not bw:
        return 1.0
    if not aw or not bw:
        return 0.0
    return len(aw & bw) / len(aw | bw)


def phrase_overlap(a: str, b: str) -> dict:
    aw = normalize_words(a)
    bw = normalize_words(b)
    aset = set(aw)
    bset = set(bw)

    shared = sorted(aset & bset)
    only_a = sorted(aset - bset)
    only_b = sorted(bset - aset)

    return {
        "shared": shared[:28],
        "only_a": only_a[:28],
        "only_b": only_b[:28],
    }


def classify_continuation(text: str) -> str:
    lower = str(text).lower()
    if "?" in text or "can you" in lower or "would you" in lower:
        return "asks_followup"
    if "for example" in lower or "instance" in lower:
        return "adds_example"
    if any(x in lower for x in ["key characteristics", "include:", "includes:"]):
        return "structured_expansion"
    if len(normalize_words(text)) <= 25:
        return "compact_answer"
    if len(normalize_words(text)) >= 65:
        return "long_expansion"
    return "standard_answer"


def prepare_data(df: pd.DataFrame) -> dict:
    required = [
        "prompt_id",
        "perturbation_type",
        "prompt",
        "temperature",
        "run_id",
        "generated_text",
        "tokens_per_second",
        "word_count",
        "lexical_diversity",
    ]

    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    df = df.copy()
    df["temperature"] = df["temperature"].astype(float)
    df["generated_text"] = df["generated_text"].fillna("").astype(str)
    df["continuation_label"] = df["generated_text"].apply(classify_continuation)
    df["prefix"] = df["generated_text"].apply(lambda x: prefix_tokens(x, 18))

    records = []
    for idx, row in df.iterrows():
        words = normalize_words(row["generated_text"])
        records.append(
            {
                "id": int(idx),
                "prompt_id": str(row["prompt_id"]),
                "perturbation_type": str(row["perturbation_type"]),
                "prompt": str(row["prompt"]),
                "temperature": float(row["temperature"]),
                "run_id": int(row["run_id"]),
                "generated_text": str(row["generated_text"]).strip(),
                "tokens_per_second": round(float(row["tokens_per_second"]), 2),
                "word_count": int(row["word_count"]),
                "lexical_diversity": round(float(row["lexical_diversity"]), 3),
                "continuation_label": row["continuation_label"],
                "prefix": row["prefix"],
                "first_words": " ".join(words[:12]),
            }
        )

    temps = sorted(df["temperature"].unique().tolist())
    prompt_ids = sorted(df["prompt_id"].unique().tolist())
    labels = sorted(df["continuation_label"].unique().tolist())

    # Build similarity edges within same prompt variant.
    edges = []
    for prompt_id in prompt_ids:
        sub = [r for r in records if r["prompt_id"] == prompt_id]
        for i in range(len(sub)):
            for j in range(i + 1, len(sub)):
                sim = jaccard(sub[i]["generated_text"], sub[j]["generated_text"])
                if sim >= 0.18:
                    edges.append(
                        {
                            "source": sub[i]["id"],
                            "target": sub[j]["id"],
                            "similarity": round(sim, 3),
                        }
                    )

    # Pairwise comparison candidates: same prompt, different temp.
    comparisons = []
    for prompt_id in prompt_ids:
        sub = [r for r in records if r["prompt_id"] == prompt_id]
        by_temp = defaultdict(list)
        for r in sub:
            by_temp[r["temperature"]].append(r)

        if len(temps) >= 2:
            low_t = min(temps)
            high_t = max(temps)
            if by_temp[low_t] and by_temp[high_t]:
                a = by_temp[low_t][0]
                b = by_temp[high_t][0]
                comparisons.append(
                    {
                        "prompt_id": prompt_id,
                        "low": a,
                        "high": b,
                        "similarity": round(jaccard(a["generated_text"], b["generated_text"]), 3),
                        "phrases": phrase_overlap(a["generated_text"], b["generated_text"]),
                    }
                )

    summary = []
    for temp in temps:
        sub = df[df["temperature"] == temp]
        label_counts = Counter(sub["continuation_label"])
        summary.append(
            {
                "temperature": float(temp),
                "n": int(len(sub)),
                "avg_tokens_per_second": round(float(sub["tokens_per_second"].mean()), 2),
                "avg_word_count": round(float(sub["word_count"].mean()), 2),
                "avg_lexical_diversity": round(float(sub["lexical_diversity"].mean()), 3),
                "labels": dict(label_counts),
            }
        )

    return {
        "records": records,
        "edges": edges,
        "temps": temps,
        "prompt_ids": prompt_ids,
        "labels": labels,
        "comparisons": comparisons,
        "summary": summary,
    }


def build_html(data: dict) -> str:
    data_json = json.dumps(data, ensure_ascii=False)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8" />
<title>Trajectory Divergence</title>
<style>
:root {{
  --bg: #f7f5f0;
  --panel: #ffffff;
  --ink: #171717;
  --muted: #686868;
  --line: #ddd8cd;
  --blue: #2d6cdf;
  --blue-soft: #e8f0ff;
  --green: #1f7a4d;
  --green-soft: #e8f7ef;
  --orange: #c45a21;
  --orange-soft: #fff0e7;
  --purple: #6546b8;
  --purple-soft: #efe9ff;
  --red: #b94242;
  --red-soft: #ffecec;
}}

* {{
  box-sizing: border-box;
}}

body {{
  margin: 0;
  background: var(--bg);
  color: var(--ink);
  font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}}

header {{
  padding: 28px 34px 18px;
  border-bottom: 1px solid var(--line);
  background: rgba(255,255,255,0.68);
  backdrop-filter: blur(10px);
  position: sticky;
  top: 0;
  z-index: 20;
}}

h1 {{
  font-size: 30px;
  line-height: 1;
  letter-spacing: -0.04em;
  margin: 0 0 8px;
}}

.subtitle {{
  color: var(--muted);
  font-size: 14px;
  max-width: 920px;
}}

.shell {{
  display: grid;
  grid-template-columns: 300px 1fr 380px;
  gap: 14px;
  padding: 14px;
  height: calc(100vh - 98px);
}}

.panel {{
  background: var(--panel);
  border: 1px solid var(--line);
  border-radius: 16px;
  overflow: hidden;
  min-height: 0;
}}

.panel-title {{
  padding: 12px 14px;
  border-bottom: 1px solid var(--line);
  font-size: 12px;
  color: var(--muted);
  font-weight: 800;
  text-transform: uppercase;
  letter-spacing: 0.09em;
}}

.panel-body {{
  padding: 14px;
  overflow: auto;
  height: calc(100% - 42px);
}}

.control {{
  margin-bottom: 14px;
}}

label {{
  display: block;
  font-size: 11px;
  font-weight: 800;
  letter-spacing: 0.08em;
  color: var(--muted);
  text-transform: uppercase;
  margin-bottom: 6px;
}}

select, button {{
  width: 100%;
  padding: 9px 10px;
  border-radius: 10px;
  border: 1px solid var(--line);
  background: white;
  font-size: 14px;
}}

button {{
  cursor: pointer;
  font-weight: 700;
}}

.metric-grid {{
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 8px;
}}

.metric {{
  border: 1px solid var(--line);
  border-radius: 12px;
  padding: 10px;
  background: #fcfbf8;
}}

.metric .k {{
  font-size: 10px;
  color: var(--muted);
  text-transform: uppercase;
  letter-spacing: 0.08em;
  font-weight: 800;
}}

.metric .v {{
  font-size: 21px;
  font-weight: 850;
  margin-top: 3px;
}}

#mapWrap {{
  position: relative;
  height: 100%;
  overflow: hidden;
  background:
    radial-gradient(circle at 30% 20%, rgba(45,108,223,0.08), transparent 28%),
    radial-gradient(circle at 70% 70%, rgba(196,90,33,0.08), transparent 32%),
    #fbfaf7;
}}

#map {{
  width: 100%;
  height: 100%;
  cursor: grab;
}}

.node {{
  cursor: pointer;
  transition: opacity 120ms ease, stroke-width 120ms ease;
}}

.node:hover {{
  stroke-width: 4;
}}

.edge {{
  stroke: #b6b1a6;
  stroke-opacity: 0.33;
}}

.edge.strong {{
  stroke-opacity: 0.7;
}}

.axis-label {{
  font-size: 12px;
  font-weight: 800;
  fill: #777;
  text-transform: uppercase;
  letter-spacing: 0.08em;
}}

.temp-band {{
  opacity: 0.08;
}}

.card {{
  border: 1px solid var(--line);
  border-radius: 12px;
  padding: 11px;
  margin-bottom: 10px;
  background: #fcfbf8;
}}

.card h3 {{
  margin: 0 0 6px;
  font-size: 15px;
}}

.small {{
  color: var(--muted);
  font-size: 12px;
}}

.output {{
  font-size: 13px;
  line-height: 1.45;
  margin-top: 8px;
}}

.badge {{
  display: inline-block;
  padding: 3px 7px;
  border-radius: 999px;
  font-size: 11px;
  font-weight: 800;
  margin-right: 4px;
  margin-top: 4px;
}}

.t-low {{ background: var(--green-soft); color: var(--green); }}
.t-mid {{ background: var(--blue-soft); color: var(--blue); }}
.t-high {{ background: var(--orange-soft); color: var(--orange); }}
.label-badge {{ background: var(--purple-soft); color: var(--purple); }}

.phrase-row {{
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 8px;
  margin-top: 9px;
}}

.phrase-box {{
  border-radius: 10px;
  padding: 8px;
  background: white;
  border: 1px solid var(--line);
}}

.phrase-box h4 {{
  margin: 0 0 6px;
  font-size: 11px;
  color: var(--muted);
  text-transform: uppercase;
  letter-spacing: 0.08em;
}}

.token {{
  display: inline-block;
  padding: 2px 6px;
  border-radius: 999px;
  font-size: 11px;
  margin: 2px;
  background: var(--blue-soft);
  color: var(--blue);
}}

.token.red {{
  background: var(--red-soft);
  color: var(--red);
}}

.compare-grid {{
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 9px;
}}

.compare-output {{
  border: 1px solid var(--line);
  border-radius: 12px;
  padding: 10px;
  background: white;
  max-height: 230px;
  overflow: auto;
}}

details {{
  border: 1px solid var(--line);
  border-radius: 10px;
  padding: 8px 10px;
  background: white;
  margin-top: 8px;
}}

summary {{
  cursor: pointer;
  font-weight: 800;
  font-size: 12px;
}}

.legend {{
  position: absolute;
  left: 14px;
  bottom: 14px;
  background: rgba(255,255,255,0.92);
  border: 1px solid var(--line);
  border-radius: 12px;
  padding: 10px;
  font-size: 12px;
}}

.legend-row {{
  display: flex;
  gap: 8px;
  align-items: center;
  margin: 4px 0;
}}

.dot {{
  width: 11px;
  height: 11px;
  border-radius: 50%;
}}

.help {{
  position: absolute;
  right: 14px;
  bottom: 14px;
  background: rgba(255,255,255,0.92);
  border: 1px solid var(--line);
  border-radius: 12px;
  padding: 10px;
  font-size: 12px;
  color: var(--muted);
  max-width: 300px;
}}

@media (max-width: 1100px) {{
  .shell {{
    grid-template-columns: 1fr;
    height: auto;
  }}

  #mapWrap {{
    height: 620px;
  }}
}}
</style>
</head>
<body>
<header>
  <h1>Trajectory Divergence Map</h1>
  <div class="subtitle">
    Data-driven visualization of generation trajectories across temperature and prompt perturbations.
  </div>
</header>

<div class="shell">
  <section class="panel">
    <div class="panel-title">Controls</div>
    <div class="panel-body">
      <div class="control">
        <label>Prompt Variant</label>
        <select id="promptSelect"></select>
      </div>
      <div class="control">
        <label>Behavior Label</label>
        <select id="labelSelect"></select>
      </div>
      <div class="control">
        <button id="resetView">Reset Map View</button>
      </div>

      <div class="metric-grid" id="metrics"></div>

      <div class="card" style="margin-top:14px;">
        <h3>Reading the map</h3>
        <div class="small">
          Each node is one generation. Horizontal position is temperature.
          Vertical position groups similar continuation behavior. Edges connect lexically similar outputs from the same prompt variant.
        </div>
      </div>

      <div class="card">
        <h3>Key quantities</h3>
        <div class="small">
          Lexical diversity = unique words / total words.
          Similarity = Jaccard overlap between output word sets.
        </div>
      </div>
    </div>
  </section>

  <section class="panel">
    <div class="panel-title">Interactive Trajectory Map</div>
    <div id="mapWrap">
      <svg id="map"></svg>
      <div class="legend">
        <div class="legend-row"><span class="dot" style="background:var(--green)"></span> Low temperature</div>
        <div class="legend-row"><span class="dot" style="background:var(--blue)"></span> Mid temperature</div>
        <div class="legend-row"><span class="dot" style="background:var(--orange)"></span> High temperature</div>
      </div>
      <div class="help">
        Drag to pan. Scroll to zoom. Click a node for full details.
      </div>
    </div>
  </section>

  <section class="panel">
    <div class="panel-title">Details</div>
    <div class="panel-body" id="details"></div>
  </section>
</div>

<script>
const DATA = {data_json};

const state = {{
  prompt: "__all__",
  label: "__all__",
  selectedId: null,
  scale: 1,
  tx: 0,
  ty: 0,
}};

const svg = document.getElementById("map");
const details = document.getElementById("details");

function escapeHtml(str) {{
  return String(str)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}}

function unique(arr) {{
  return [...new Set(arr)];
}}

function avg(arr) {{
  if (!arr.length) return 0;
  return arr.reduce((a,b) => a+b, 0) / arr.length;
}}

function tempColor(t) {{
  if (t <= 0.3) return "var(--green)";
  if (t < 1.0) return "var(--blue)";
  return "var(--orange)";
}}

function tempBadgeClass(t) {{
  if (t <= 0.3) return "t-low";
  if (t < 1.0) return "t-mid";
  return "t-high";
}}

function filteredRecords() {{
  return DATA.records.filter(r => {{
    const p = state.prompt === "__all__" || r.prompt_id === state.prompt;
    const l = state.label === "__all__" || r.continuation_label === state.label;
    return p && l;
  }});
}}

function filteredEdges(records) {{
  const ids = new Set(records.map(r => r.id));
  return DATA.edges.filter(e => ids.has(e.source) && ids.has(e.target));
}}

function initControls() {{
  const promptSelect = document.getElementById("promptSelect");
  const labelSelect = document.getElementById("labelSelect");

  promptSelect.innerHTML = [
    `<option value="__all__">All prompt variants</option>`,
    ...DATA.prompt_ids.map(p => `<option value="${{p}}">${{p}}</option>`)
  ].join("");

  labelSelect.innerHTML = [
    `<option value="__all__">All behavior labels</option>`,
    ...DATA.labels.map(l => `<option value="${{l}}">${{l}}</option>`)
  ].join("");

  promptSelect.addEventListener("change", e => {{
    state.prompt = e.target.value;
    state.selectedId = null;
    render();
  }});

  labelSelect.addEventListener("change", e => {{
    state.label = e.target.value;
    state.selectedId = null;
    render();
  }});

  document.getElementById("resetView").addEventListener("click", () => {{
    state.scale = 1;
    state.tx = 0;
    state.ty = 0;
    render();
  }});
}}

function computeLayout(records) {{
  const width = svg.clientWidth || 800;
  const height = svg.clientHeight || 600;

  const temps = [...DATA.temps].sort((a,b) => a-b);
  const labels = [...DATA.labels].sort();

  const xByTemp = new Map();
  temps.forEach((t, i) => {{
    xByTemp.set(Number(t), 130 + i * ((width - 260) / Math.max(temps.length - 1, 1)));
  }});

  const yByLabel = new Map();
  labels.forEach((label, i) => {{
    yByLabel.set(label, 95 + i * ((height - 190) / Math.max(labels.length - 1, 1)));
  }});

  const byBucket = {{}};

  records.forEach((r, idx) => {{
    const key = `${{r.temperature}}-${{r.continuation_label}}`;
    if (!byBucket[key]) byBucket[key] = 0;
    const offset = byBucket[key]++;

    const angle = offset * 2.399;
    const radius = 12 + 6 * Math.sqrt(offset);

    r.x = xByTemp.get(Number(r.temperature)) + Math.cos(angle) * radius;
    r.y = yByLabel.get(r.continuation_label) + Math.sin(angle) * radius;
  }});

  return {{ width, height }};
}}

function renderMetrics(records) {{
  const el = document.getElementById("metrics");

  el.innerHTML = `
    <div class="metric">
      <div class="k">Runs</div>
      <div class="v">${{records.length}}</div>
    </div>
    <div class="metric">
      <div class="k">Avg tok/s</div>
      <div class="v">${{avg(records.map(r => r.tokens_per_second)).toFixed(1)}}</div>
    </div>
    <div class="metric">
      <div class="k">Avg words</div>
      <div class="v">${{avg(records.map(r => r.word_count)).toFixed(1)}}</div>
    </div>
    <div class="metric">
      <div class="k">Avg diversity</div>
      <div class="v">${{avg(records.map(r => r.lexical_diversity)).toFixed(2)}}</div>
    </div>
  `;
}}

function renderMap(records, edges) {{
  const {{ width, height }} = computeLayout(records);
  const recById = new Map(records.map(r => [r.id, r]));

  svg.setAttribute("viewBox", `${{-state.tx}} ${{-state.ty}} ${{width / state.scale}} ${{height / state.scale}}`);

  const temps = [...DATA.temps].sort((a,b) => a-b);
  const labels = [...DATA.labels].sort();

  const tempBands = temps.map((t, i) => {{
    const x = 80 + i * ((width - 160) / Math.max(temps.length - 1, 1));
    return `<rect class="temp-band" x="${{x - 95}}" y="0" width="190" height="${{height}}" fill="${{tempColor(t)}}"></rect>
            <text class="axis-label" x="${{x}}" y="30" text-anchor="middle">T=${{t}}</text>`;
  }}).join("");

  const labelText = labels.map((label, i) => {{
    const y = 95 + i * ((height - 190) / Math.max(labels.length - 1, 1));
    return `<text class="axis-label" x="18" y="${{y + 4}}">${{escapeHtml(label)}}</text>`;
  }}).join("");

  const edgeSvg = edges.map(e => {{
    const a = recById.get(e.source);
    const b = recById.get(e.target);
    if (!a || !b) return "";
    const strong = e.similarity >= 0.35 ? "strong" : "";
    return `<line class="edge ${{strong}}" x1="${{a.x}}" y1="${{a.y}}" x2="${{b.x}}" y2="${{b.y}}" stroke-width="${{1 + e.similarity * 3}}"></line>`;
  }}).join("");

  const nodeSvg = records.map(r => {{
    const selected = state.selectedId === r.id;
    const radius = 7 + Math.min(7, r.lexical_diversity * 8);
    return `<circle
      class="node"
      data-id="${{r.id}}"
      cx="${{r.x}}"
      cy="${{r.y}}"
      r="${{radius}}"
      fill="${{tempColor(r.temperature)}}"
      stroke="${{selected ? "#111" : "white"}}"
      stroke-width="${{selected ? 4 : 2}}"
      opacity="${{selected || state.selectedId === null ? 0.95 : 0.45}}"
    >
      <title>${{escapeHtml(r.prompt_id)}} · T=${{r.temperature}} · ${{escapeHtml(r.continuation_label)}}</title>
    </circle>`;
  }}).join("");

  svg.innerHTML = `
    ${{tempBands}}
    ${{labelText}}
    ${{edgeSvg}}
    ${{nodeSvg}}
  `;

  svg.querySelectorAll(".node").forEach(node => {{
    node.addEventListener("click", () => {{
      state.selectedId = Number(node.dataset.id);
      render();
    }});
  }});
}}

function renderDetails(records) {{
  const selected = records.find(r => r.id === state.selectedId) || records[0];

  if (!selected) {{
    details.innerHTML = `<div class="small">No records match current filters.</div>`;
    return;
  }}

  const samePrompt = DATA.records.filter(r => r.prompt_id === selected.prompt_id);
  const low = samePrompt.filter(r => r.temperature === Math.min(...DATA.temps))[0];
  const high = samePrompt.filter(r => r.temperature === Math.max(...DATA.temps))[0];

  const phrases = low && high ? computePhrases(low.generated_text, high.generated_text) : null;
  const sim = low && high ? jaccard(low.generated_text, high.generated_text).toFixed(3) : "—";

  details.innerHTML = `
    <div class="card">
      <h3>Selected generation</h3>
      <span class="badge ${{tempBadgeClass(selected.temperature)}}">T=${{selected.temperature}}</span>
      <span class="badge label-badge">${{escapeHtml(selected.continuation_label)}}</span>
      <div class="small" style="margin-top:8px;">
        Prompt: <strong>${{escapeHtml(selected.prompt_id)}}</strong> · Run ${{selected.run_id}} ·
        ${{selected.tokens_per_second}} tok/s · ${{selected.word_count}} words · diversity ${{selected.lexical_diversity}}
      </div>
      <div class="output">${{escapeHtml(selected.generated_text)}}</div>

      <details>
        <summary>Full prompt</summary>
        <div class="output">${{escapeHtml(selected.prompt)}}</div>
      </details>
    </div>

    ${{low && high ? `
    <div class="card">
      <h3>Low vs high temperature comparison</h3>
      <div class="small">Same prompt variant: <strong>${{escapeHtml(selected.prompt_id)}}</strong> · similarity ${{sim}}</div>

      <div class="compare-grid" style="margin-top:10px;">
        <div class="compare-output">
          <span class="badge ${{tempBadgeClass(low.temperature)}}">T=${{low.temperature}}</span>
          <div class="output">${{escapeHtml(low.generated_text)}}</div>
        </div>
        <div class="compare-output">
          <span class="badge ${{tempBadgeClass(high.temperature)}}">T=${{high.temperature}}</span>
          <div class="output">${{escapeHtml(high.generated_text)}}</div>
        </div>
      </div>

      <div class="phrase-row">
        <div class="phrase-box">
          <h4>Shared terms</h4>
          ${{phrases.shared.map(w => `<span class="token">${{escapeHtml(w)}}</span>`).join("")}}
        </div>
        <div class="phrase-box">
          <h4>High-temp distinct terms</h4>
          ${{phrases.onlyB.map(w => `<span class="token red">${{escapeHtml(w)}}</span>`).join("")}}
        </div>
      </div>

      <details>
        <summary>Low-temp distinct terms</summary>
        <div style="margin-top:8px;">
          ${{phrases.onlyA.map(w => `<span class="token red">${{escapeHtml(w)}}</span>`).join("")}}
        </div>
      </details>
    </div>
    ` : ""}}
  `;
}}

function computePhrases(a, b) {{
  const aw = normWords(a);
  const bw = normWords(b);
  const as = new Set(aw);
  const bs = new Set(bw);

  return {{
    shared: [...as].filter(w => bs.has(w)).slice(0, 36),
    onlyA: [...as].filter(w => !bs.has(w)).slice(0, 36),
    onlyB: [...bs].filter(w => !as.has(w)).slice(0, 36),
  }};
}}

function normWords(text) {{
  return String(text).toLowerCase().replace(/[^a-z0-9\\s]/g, " ").split(/\\s+/).filter(Boolean);
}}

function jaccard(a, b) {{
  const as = new Set(normWords(a));
  const bs = new Set(normWords(b));
  const inter = [...as].filter(x => bs.has(x)).length;
  const union = new Set([...as, ...bs]).size;
  return union ? inter / union : 1;
}}

function render() {{
  const records = filteredRecords();
  const edges = filteredEdges(records);
  renderMetrics(records);
  renderMap(records, edges);
  renderDetails(records);
}}

let dragging = false;
let lastX = 0;
let lastY = 0;

svg.addEventListener("mousedown", e => {{
  dragging = true;
  lastX = e.clientX;
  lastY = e.clientY;
  svg.style.cursor = "grabbing";
}});

window.addEventListener("mouseup", () => {{
  dragging = false;
  svg.style.cursor = "grab";
}});

window.addEventListener("mousemove", e => {{
  if (!dragging) return;
  const dx = e.clientX - lastX;
  const dy = e.clientY - lastY;
  state.tx += dx / state.scale;
  state.ty += dy / state.scale;
  lastX = e.clientX;
  lastY = e.clientY;
  render();
}});

svg.addEventListener("wheel", e => {{
  e.preventDefault();
  const delta = e.deltaY > 0 ? 0.9 : 1.1;
  state.scale = Math.max(0.55, Math.min(3.5, state.scale * delta));
  render();
}}, {{ passive: false }});

initControls();
render();
</script>
</body>
</html>
"""


def main() -> None:
    if not RESULTS_PATH.exists():
        raise FileNotFoundError(f"Missing {RESULTS_PATH}. Run trajectory_study.py first.")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(RESULTS_PATH)
    data = prepare_data(df)
    OUTPUT_PATH.write_text(build_html(data), encoding="utf-8")

    print(f"Wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
