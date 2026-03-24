const API_RUN = "/api/run";
const API_COMPARE = "/api/compare";
const API_DENSITY = "/api/experiment/density";
const API_HEALTH = "/api/health";
const API_EXPORT = "/api/export/run.csv";
const API_ML_PREDICT = "/api/ml/predict";
const API_ML_TRAIN = "/api/ml/train";
const API_ML_STATUS = "/api/ml/status";
const API_ML_MEMORY = "/api/ml/memory";

const $ = (id) => document.getElementById(id);

const palette = {
  text: "#F6F0D7",
  muted: "#C7EABB",
  accent: "#84B179",
  accent2: "#A2CB8B",
  accent3: "#C7EABB",
  accent4: "#E8F5BD",
  accent5: "#F6F0D7",
  danger: "#D95C5C",
  paper: "rgba(0,0,0,0)",
  plot: "rgba(0,0,0,0)",
};

const controlDescriptions = {
  f_model: {
    random: "Random: failures are spread across active nodes without any location pattern.",
    clustered: "Clustered: nearby active nodes are more likely to fail together, simulating a local disturbance.",
    region: "Region: failures affect one region of the field more than the others.",
    periodic: "Periodic: failure bursts appear on periodic rounds instead of every round.",
  },
  r_model: {
    greedy_coverage: "Greedy Coverage: wakes backup nodes that recover the most missing coverage first.",
    nearest_backup: "Nearest Backup: wakes the closest backup nodes first, keeping recovery more local.",
  },
  sensor_type: {
    homogeneous: "Homogeneous: all nodes use the same sensing radius and energy cost.",
    heterogeneous: "Heterogeneous: nodes get varied sensing radius and energy cost to mimic uneven hardware.",
    temperature: "Temperature: shorter range with lower energy draw.",
    humidity: "Humidity: balanced sensing and energy behaviour.",
    motion: "Motion: longer range but higher energy draw.",
    mixed: "Mixed Multi-Sensor: combines temperature, humidity, and motion nodes in one deployment.",
  },
};

let runCache = null;
let compareCache = null;
let compareView = "voronoi";
let compareStage = "initial";
let expRows = [];
let latestMLStatus = null;

function bindSlider(sliderId, labelId) {
  const slider = $(sliderId);
  const label = $(labelId);
  label.textContent = slider.value;
  slider.addEventListener("input", () => {
    label.textContent = slider.value;
  });
}

function fmtNumber(value, digits = 2) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "-";
  return Number(value).toFixed(digits);
}

function fmtPct(value, digits = 2) {
  return `${fmtNumber(value, digits)}%`;
}

function setBusy(btn, busy, labelBusy = "Working…") {
  btn.disabled = !!busy;
  btn.dataset.old = btn.dataset.old || btn.textContent;
  btn.textContent = busy ? labelBusy : btn.dataset.old;
}

function baseLayout(title, xTitle = "", yTitle = "") {
  return {
    title: { text: title, x: 0.02, font: { color: palette.text, size: 16 } },
    paper_bgcolor: palette.paper,
    plot_bgcolor: palette.plot,
    font: { color: palette.text, size: 12 },
    margin: { l: 62, r: 26, t: 64, b: 64 },
    xaxis: {
      title: xTitle,
      gridcolor: "rgba(162,203,139,0.10)",
      zerolinecolor: "rgba(162,203,139,0.10)",
      automargin: true,
      tickfont: { size: 11, color: palette.muted },
      titlefont: { size: 12, color: palette.text },
    },
    yaxis: {
      title: yTitle,
      gridcolor: "rgba(162,203,139,0.10)",
      zerolinecolor: "rgba(162,203,139,0.10)",
      automargin: true,
      tickfont: { size: 11, color: palette.muted },
      titlefont: { size: 12, color: palette.text },
    },
    legend: { orientation: "h", y: 1.16, x: 0, font: { size: 11, color: palette.text } },
    hovermode: "closest",
  };
}

function boundedRange(values, { minBound = null, maxBound = null, padFraction = 0.06, minSpan = null } = {}) {
  if (!values.length) return null;
  let min = Math.min(...values);
  let max = Math.max(...values);
  if (min === max) {
    const span = minSpan ?? Math.max(Math.abs(min) * 0.15, 1);
    min -= span / 2;
    max += span / 2;
  } else {
    const pad = Math.max((max - min) * padFraction, 1e-9);
    min -= pad;
    max += pad;
  }
  if (minBound !== null) min = Math.max(minBound, min);
  if (maxBound !== null) max = Math.min(maxBound, max);
  if (minBound !== null && maxBound !== null && max - min < 1e-9) {
    min = minBound;
    max = maxBound;
  }
  return [min, max];
}

function plotConfig() {
  return { displayModeBar: false, responsive: true };
}

function resizeAllPlots() {
  [
    "chartCoverage", "chartActive", "energyBarChart", "energyRoundChart", "lifetimeChart", "energyTradeoffChart",
    "chartEnergyDensity", "chartBackupDensity", "chartCoverageDensity", "chartLifetimeDensity",
  ].forEach((id) => {
    const el = $(id);
    if (el && el.data) Plotly.Plots.resize(el);
  });
}

function average(values) {
  if (!values.length) return 0;
  return values.reduce((sum, value) => sum + value, 0) / values.length;
}

function setTab(tabId) {
  document.querySelectorAll(".tab").forEach((button) => button.classList.remove("active"));
  document.querySelectorAll(".tabpane").forEach((pane) => pane.classList.remove("active"));
  document.querySelector(`.tab[data-tab="${tabId}"]`)?.classList.add("active");
  $(tabId).classList.add("active");
  requestAnimationFrame(() => {
    if (tabId === "tab_run" && runCache) {
      renderRunViz(runCache);
      renderRunCharts(runCache);
    }
    if (tabId === "tab_energy" && runCache) {
      updateEnergyDashboard(runCache, compareCache);
    }
    if (tabId === "tab_compare" && compareCache) {
      renderCompareViz();
    }
    if (tabId === "tab_experiment" && expRows.length) {
      renderExpCharts(expRows);
    }
    resizeAllPlots();
  });
}

function updateControlDescriptions() {
  Object.entries(controlDescriptions).forEach(([id, descriptions]) => {
    const select = $(id);
    const target = $(`${id}_desc`);
    if (!select || !target) return;
    target.textContent = descriptions[select.value] || "";
  });
}

function clampFailureProbability() {
  const input = $("failp");
  const warning = $("failureWarning");
  let value = Number(input.value);
  if (Number.isNaN(value)) value = 0.01;
  if (value < 0) value = 0;
  if (value > 0.02) {
    value = 0.02;
    warning.classList.remove("hidden");
  } else {
    warning.classList.add("hidden");
  }
  input.value = value.toFixed(3);
}

function clampThresholdCoeff() {
  const input = $("th");
  const warning = $("thresholdWarning");
  let value = Number(input.value);
  if (Number.isNaN(value)) value = 0.02;
  if (value < 0) value = 0;
  if (value > 0.02) {
    value = 0.02;
    warning.classList.remove("hidden");
  } else {
    warning.classList.add("hidden");
  }
  input.value = value.toFixed(3);
}

function payloadFromUI() {
  clampFailureProbability();
  clampThresholdCoeff();
  return {
    n_nodes: +$("n_nodes").value,
    width: +$("width").value,
    height: +$("height").value,
    sensing_radius: +$("radius").value,
    threshold_coeff: +$("th").value,
    seed: +$("seed").value,
    enable_fault_tolerance: $("ft").checked,
    failure_prob_per_round: +$("failp").value,
    failure_model: $("f_model").value,
    recovery_model: $("r_model").value,
    n_rounds: +$("rounds").value,
    enable_ai: $("ai").checked,
    sensor_type: $("sensor_type").value,
  };
}

function checkHealth() {
  fetch(API_HEALTH)
    .then((res) => res.json())
    .then((data) => {
      $("statusLine").textContent = `API: ${data.status}`;
    })
    .catch(() => {
      $("statusLine").textContent = "API: offline - start backend";
    });
}

function updateRunCards(metrics) {
  const initial = metrics.initial_snapshot || {};
  const final = metrics.final_snapshot || {};
  $("activeNodes").textContent = initial.active_nodes ?? metrics.n_on;
  $("backupNodes").textContent = initial.backup_nodes ?? metrics.n_off;
  $("initialFailedNodes").textContent = initial.failed_nodes ?? 0;
  $("coverage").textContent = fmtNumber(initial.coverage ?? metrics.coverage_scheduled, 4);
  $("energySaved").textContent = fmtPct(initial.energy_saved_pct ?? metrics.energy_saved_pct, 2);
  $("uncoveredPct").textContent = fmtPct(initial.uncovered_area_pct ?? metrics.uncovered_area_pct, 2);
  $("algoRuntime").textContent = `${fmtNumber(initial.runtime_ms ?? metrics.algorithm_runtime_ms, 2)} ms`;

  $("finalActiveNodes").textContent = final.active_nodes ?? metrics.n_on;
  $("finalBackupNodes").textContent = final.backup_nodes ?? metrics.n_off;
  $("finalFailedNodes").textContent = final.failed_nodes ?? metrics.n_unavailable ?? 0;
  $("finalCoverage").textContent = fmtNumber(final.coverage ?? metrics.coverage_scheduled, 4);
  $("finalEnergySaved").textContent = fmtPct(final.energy_saved_pct ?? metrics.energy_saved_pct, 2);
  $("finalUncoveredPct").textContent = fmtPct(final.uncovered_area_pct ?? metrics.uncovered_area_pct, 2);
  $("finalRuntime").textContent = `${fmtNumber(final.runtime_ms ?? metrics.total_runtime_ms ?? metrics.algorithm_runtime_ms, 2)} ms`;

  $("fieldProportionNote").textContent = `${fmtNumber(metrics.width, 1)} × ${fmtNumber(metrics.height, 1)} field · ${metrics.field_shape} footprint · visualization keeps the same aspect ratio.`;
}

function updateRunLegendInfo(data) {
  const failedNow = (data.unavailable_mask || []).filter(Boolean).length;
  const initialBackups = data.metrics?.round_summary?.initial_backup_nodes ?? data.metrics?.n_off ?? 0;
  const finalBackupPool = data.metrics?.round_summary?.final_backup_available ?? initialBackups;
  $("runLegendNote").innerHTML = `<b>Active</b> nodes are ON and still sensing. <b>Backup / OFF</b> nodes are sleeping standby nodes that can wake up during recovery. <b>Failed</b> nodes are unavailable after injected faults.<br><br><b>This run:</b> initial backup pool = ${initialBackups}, standby backups remaining after the last round = ${finalBackupPool}, failed nodes now = ${failedNow}.`;
}

function updateRunRoundNote(data) {
  const logs = data.fault_logs || [];
  const initialBackups = data.metrics?.round_summary?.initial_backup_nodes ?? 0;
  if (!logs.length) {
    $("runRoundNote").textContent = "Fault tolerance is disabled, so no round-wise failure or recovery curves are shown.";
    return;
  }
  const minCoverage = Math.min(...logs.map((row) => row.coverage));
  const backupSeries = logs.map((row) => row.n_backup_available);
  const backupMessage = initialBackups === 0
    ? "No standby backups were created in the initial schedule, so the backup-available curve stays at 0."
    : `Standby backups remaining vary from ${Math.min(...backupSeries)} to ${Math.max(...backupSeries)}.`;
  $("runRoundNote").textContent = `Round plots use only logged values from this run. Minimum recovered coverage is ${fmtNumber(minCoverage, 4)}. ${backupMessage}`;
}

function metricRows(metrics) {
  const final = metrics.final_snapshot || {};
  const rows = [
    ["Algorithm", metrics.algo],
    ["Field size (m)", `${fmtNumber(metrics.width, 1)} × ${fmtNumber(metrics.height, 1)}`],
    ["Density (nodes/m²)", fmtNumber(metrics.density, 5)],
    ["Sensor mode", metrics.sensor_type],
    ["Sensor explanation", metrics.sensor_explanation],
    ["Coverage (all nodes)", fmtNumber(metrics.coverage_all, 4)],
    ["Coverage (scheduled)", fmtNumber(metrics.coverage_scheduled, 4)],
    ["Target coverage", fmtNumber(metrics.target_coverage, 4)],
    ["Baseline energy / round", fmtNumber(metrics.baseline_energy, 2)],
    ["Scheduled energy / round", fmtNumber(metrics.scheduled_energy, 2)],
    ["Saved energy (%)", fmtPct(metrics.energy_saved_pct, 2)],
    ["Estimated lifetime (rounds)", `${fmtNumber(metrics.estimated_network_lifetime_rounds, 2)} rounds`],
    ["Lifetime improvement", fmtPct(metrics.lifetime_improvement_pct, 2)],
    ["Total failures", metrics.total_failures ?? 0],
    ["Total recoveries", metrics.total_recoveries ?? 0],
    ["Recovery success", fmtPct(metrics.recovery_success_rate_pct ?? 0, 2)],
    ["Final active nodes", final.active_nodes ?? metrics.n_on],
    ["Final backup nodes", final.backup_nodes ?? metrics.n_off],
    ["Final failed nodes", final.failed_nodes ?? metrics.n_unavailable ?? 0],
    ["Final coverage", final.coverage !== undefined ? fmtNumber(final.coverage, 4) : "-"],
    ["Minimum round coverage", metrics.min_round_coverage !== null && metrics.min_round_coverage !== undefined ? fmtNumber(metrics.min_round_coverage, 4) : "-"],
    ["Average round energy saved", metrics.avg_round_energy_saved_pct !== null && metrics.avg_round_energy_saved_pct !== undefined ? fmtPct(metrics.avg_round_energy_saved_pct, 2) : "-"],
  ];
  if (metrics.ml_predicted_metrics) {
    rows.push(["ML predicted coverage", fmtNumber(metrics.ml_predicted_metrics.coverage_scheduled, 4)]);
    rows.push(["ML predicted energy saved", fmtPct(metrics.ml_predicted_metrics.energy_saved_pct, 2)]);
    rows.push(["ML predicted lifetime", fmtNumber(metrics.ml_predicted_metrics.estimated_network_lifetime_rounds, 2)]);
  }
  return rows;
}

function renderMetricsTable(metrics) {
  $("metricsTable").innerHTML = metricRows(metrics).map(([name, value]) => `
    <div class="metric-row">
      <span class="name">${name}</span>
      <span class="val">${value}</span>
    </div>
  `).join("");
}

function fieldTransform(width, height, containerWidth, containerHeight, pad = 26) {
  const scale = Math.min((containerWidth - 2 * pad) / width, (containerHeight - 2 * pad) / height);
  const plotWidth = width * scale;
  const plotHeight = height * scale;
  const offsetX = (containerWidth - plotWidth) / 2;
  const offsetY = (containerHeight - plotHeight) / 2;
  return {
    scale,
    offsetX,
    offsetY,
    x: (value) => offsetX + value * scale,
    y: (value) => offsetY + value * scale,
  };
}

function transformBoundary(boundary, transform) {
  if (!boundary) return [];
  return boundary.map(([x, y]) => [transform.x(x), transform.y(y)]);
}

function drawSvgField(svgId, data) {
  const svg = $(svgId);
  while (svg.firstChild) svg.removeChild(svg.firstChild);

  const w = svg.clientWidth || 1000;
  const h = svg.clientHeight || 560;
  svg.setAttribute("viewBox", `0 0 ${w} ${h}`);

  const transform = fieldTransform(data.metrics.width, data.metrics.height, w, h, 24);
  const boundary = transformBoundary(data.boundary, transform);

  const boundaryPath = document.createElementNS("http://www.w3.org/2000/svg", "polygon");
  boundaryPath.setAttribute("points", boundary.map(([x, y]) => `${x},${y}`).join(" "));
  boundaryPath.setAttribute("fill", "rgba(132,177,121,0.04)");
  boundaryPath.setAttribute("stroke", "rgba(162,203,139,0.20)");
  boundaryPath.setAttribute("stroke-width", "2");
  svg.appendChild(boundaryPath);

  (data.voronoi_cells || []).forEach((cell) => {
    const poly = document.createElementNS("http://www.w3.org/2000/svg", "polygon");
    const coords = cell.coords.map(([x, y]) => `${transform.x(x)},${transform.y(y)}`).join(" ");
    poly.setAttribute("points", coords);
    poly.setAttribute("fill", "rgba(132,177,121,0.08)");
    poly.setAttribute("stroke", "rgba(162,203,139,0.12)");
    poly.setAttribute("stroke-width", "1");
    svg.appendChild(poly);
  });

  data.points.forEach((point, index) => {
    const x = transform.x(point[0]);
    const y = transform.y(point[1]);
    const radius = (data.sensing_radii?.[index] || data.metrics.sensing_radius || 15) * transform.scale;
    const isActive = !!data.active_mask[index];
    const isFailed = !!(data.unavailable_mask && data.unavailable_mask[index]);

    if (isActive && !isFailed) {
      const coverage = document.createElementNS("http://www.w3.org/2000/svg", "circle");
      coverage.setAttribute("cx", x);
      coverage.setAttribute("cy", y);
      coverage.setAttribute("r", radius);
      coverage.setAttribute("fill", "rgba(162,203,139,0.08)");
      coverage.setAttribute("stroke", "rgba(132,177,121,0.18)");
      coverage.setAttribute("stroke-width", "1");
      svg.appendChild(coverage);
    }

    const node = document.createElementNS("http://www.w3.org/2000/svg", "circle");
    node.setAttribute("cx", x);
    node.setAttribute("cy", y);
    node.setAttribute("r", isFailed ? 4.7 : isActive ? 4.5 : 5.4);
    node.setAttribute("fill", isFailed ? palette.danger : isActive ? palette.accent : palette.accent4);
    node.setAttribute("stroke", isFailed ? "#7A2323" : "#102019");
    node.setAttribute("stroke-width", isFailed ? "1.2" : "1");
    const title = document.createElementNS("http://www.w3.org/2000/svg", "title");
    title.textContent = `Node ${index} · ${isFailed ? "FAILED" : isActive ? "ACTIVE" : "BACKUP/OFF"} · ${data.sensor_labels?.[index] || data.metrics.sensor_type}`;
    node.appendChild(title);
    svg.appendChild(node);
  });
}

function renderRunViz(data) {
  updateRunLegendInfo(data);
  drawSvgField("voronoiSvg", data);
}

function renderRunCharts(data) {
  const logs = data.fault_logs || [];
  updateRunRoundNote(data);
  if (!logs.length) {
    Plotly.react("chartCoverage", [], baseLayout("Recovered coverage over rounds", "Round", "Coverage"), plotConfig());
    Plotly.react("chartActive", [], baseLayout("Node states over rounds", "Round", "Nodes"), plotConfig());
    return;
  }
  const rounds = logs.map((row) => row.round);
  const coverages = logs.map((row) => row.coverage);
  const actives = logs.map((row) => row.n_active);
  const backups = logs.map((row) => row.n_backup_available);
  const failed = logs.map((row) => row.n_unavailable);
  const coverageRange = boundedRange([...coverages, data.metrics.target_coverage], { minBound: Math.max(0, Math.min(...coverages) - 0.08), maxBound: 1.02, minSpan: 0.08 });

  Plotly.react(
    "chartCoverage",
    [
      { x: rounds, y: coverages, mode: "lines+markers", line: { color: palette.accent, shape: "hv", width: 2.5 }, marker: { size: 5 }, fill: "tozeroy", fillcolor: "rgba(132,177,121,0.16)", name: "Recovered coverage", hovertemplate: "Round %{x}<br>Coverage %{y:.4f}<extra></extra>" },
      { x: rounds, y: rounds.map(() => data.metrics.target_coverage), mode: "lines", line: { color: palette.accent4, dash: "dash", width: 2 }, name: "Target coverage", hovertemplate: "Target %{y:.4f}<extra></extra>" },
    ],
    { ...baseLayout("Recovered coverage over rounds", "Round", "Coverage ratio"), yaxis: { title: "Coverage ratio", range: coverageRange || [0, 1.02], gridcolor: "rgba(162,203,139,0.10)", automargin: true } },
    plotConfig(),
  );

  Plotly.react(
    "chartActive",
    [
      { x: rounds, y: actives, mode: "lines+markers", line: { color: palette.accent, shape: "hv", width: 2.5 }, marker: { size: 5 }, stackgroup: "states", fill: "tonexty", fillcolor: "rgba(132,177,121,0.18)", name: "Active nodes", hovertemplate: "Round %{x}<br>Active %{y}<extra></extra>" },
      { x: rounds, y: backups, mode: "lines+markers", line: { color: palette.accent4, shape: "hv", width: 2.5 }, marker: { size: 5 }, stackgroup: "states", fill: "tonexty", fillcolor: "rgba(232,245,189,0.18)", name: "Standby backups", hovertemplate: "Round %{x}<br>Standby backups %{y}<extra></extra>" },
      { x: rounds, y: failed, mode: "lines+markers", line: { color: palette.danger, shape: "hv", width: 2.5 }, marker: { size: 5 }, stackgroup: "states", fill: "tonexty", fillcolor: "rgba(217,92,92,0.18)", name: "Failed nodes", hovertemplate: "Round %{x}<br>Failed %{y}<extra></extra>" },
    ],
    { ...baseLayout("Node states over rounds", "Round", "Node count"), yaxis: { title: "Node count", range: [0, data.metrics.n_nodes], gridcolor: "rgba(162,203,139,0.10)", automargin: true } },
    plotConfig(),
  );
}

function updateEnergyCards(metrics, faultLogs = []) {
  const initial = metrics.initial_snapshot || {};
  const final = metrics.final_snapshot || {};
  const initialBaseline = Number(metrics.baseline_energy || 0);
  const initialScheduled = Number(metrics.scheduled_energy || 0);
  const initialSavedAbs = Math.max(0, initialBaseline - initialScheduled);

  const lastLog = faultLogs.length ? faultLogs[faultLogs.length - 1] : null;
  const finalBaseline = Number(lastLog?.baseline_energy ?? metrics.baseline_energy ?? 0);
  const finalScheduled = Number(lastLog?.scheduled_energy ?? metrics.scheduled_energy ?? 0);
  const finalSavedAbs = Math.max(0, finalBaseline - finalScheduled);

  $("initialBaselineEnergy").textContent = fmtNumber(initialBaseline, 2);
  $("initialScheduledEnergy").textContent = fmtNumber(initialScheduled, 2);
  $("initialSavedEnergyAbs").textContent = fmtNumber(initialSavedAbs, 2);
  $("initialEnergySavedPct").textContent = fmtPct(initial.energy_saved_pct ?? metrics.energy_saved_pct, 2);
  $("initialEnergyCoverage").textContent = fmtNumber(initial.coverage ?? metrics.coverage_scheduled, 4);
  $("initialLifetimeImprovement").textContent = fmtPct(metrics.lifetime_improvement_pct, 2);

  $("finalBaselineEnergy").textContent = fmtNumber(finalBaseline, 2);
  $("finalScheduledEnergy").textContent = fmtNumber(finalScheduled, 2);
  $("finalSavedEnergyAbs").textContent = fmtNumber(finalSavedAbs, 2);
  $("finalEnergySavedPctCard").textContent = fmtPct(final.energy_saved_pct ?? metrics.energy_saved_pct, 2);
  $("finalEnergyCoverage").textContent = fmtNumber(final.coverage ?? metrics.coverage_scheduled, 4);
  $("finalEnergyActiveNodes").textContent = final.active_nodes ?? metrics.n_on;

  $("avgRoundEnergySaved").textContent = faultLogs.length ? fmtPct(average(faultLogs.map((row) => row.energy_saved_pct)), 2) : fmtPct(metrics.energy_saved_pct, 2);
  $("recoverySuccess").textContent = fmtPct(metrics.recovery_success_rate_pct || 0, 2);
}

function getCompareRows(stage = "initial") {
  if (!compareCache) return [];
  const group = compareCache[stage] || {};
  return [
    ["Voronoi", group.voronoi],
    ["Random Same OFF", group.random_same_off],
    ["Random Greedy", group.random_greedy_cov],
    ["Greedy Coverage", group.greedy_cov],
    ["AI Based", group.ai_based],
  ].filter((row) => row[1]);
}

function snapshotForStage(payload, stage = "initial") {
  if (!payload?.metrics) return null;
  return stage === "final" ? payload.metrics.final_snapshot : payload.metrics.initial_snapshot;
}

function updateEnergyInsight(compareRows, metrics, faultLogs = []) {
  const rows = compareRows.filter((row) => row[1]);
  const initial = metrics.initial_snapshot || {};
  const final = metrics.final_snapshot || {};
  const avgRound = faultLogs.length ? average(faultLogs.map((row) => row.energy_saved_pct)) : metrics.energy_saved_pct;
  if (!rows.length) {
    $("energyInsight").innerHTML = `<b>Energy story for this field:</b> the run starts at <b>${fmtPct(initial.energy_saved_pct ?? metrics.energy_saved_pct, 2)}</b> saved energy with <b>${fmtNumber(initial.coverage ?? metrics.coverage_scheduled, 4)}</b> coverage and ends at <b>${fmtPct(final.energy_saved_pct ?? metrics.energy_saved_pct, 2)}</b> saved energy with <b>${fmtNumber(final.coverage ?? metrics.coverage_scheduled, 4)}</b> coverage. Average round savings are <b>${fmtPct(avgRound, 2)}</b>. Recovery success is <b>${fmtPct(metrics.recovery_success_rate_pct || 0, 2)}</b>.`;
    return;
  }
  const mapped = rows.map(([name, payload]) => [name, snapshotForStage(payload, "initial"), payload.metrics]);
  const bestSaver = mapped.reduce((best, row) => (row[1].energy_saved_pct > best[1].energy_saved_pct ? row : best));
  const safest = mapped.reduce((best, row) => (row[1].coverage > best[1].coverage ? row : best));
  $("energyInsight").innerHTML = `<b>Energy story for this field:</b> the run starts at <b>${fmtPct(initial.energy_saved_pct ?? metrics.energy_saved_pct, 2)}</b> saved energy with <b>${fmtNumber(initial.coverage ?? metrics.coverage_scheduled, 4)}</b> coverage and ends at <b>${fmtPct(final.energy_saved_pct ?? metrics.energy_saved_pct, 2)}</b> saved energy with <b>${fmtNumber(final.coverage ?? metrics.coverage_scheduled, 4)}</b> coverage. Best initial saver in compare mode is <b>${bestSaver[0]}</b> at <b>${fmtPct(bestSaver[1].energy_saved_pct, 2)}</b>, while the highest initial coverage is <b>${safest[0]}</b> at <b>${fmtNumber(safest[1].coverage, 4)}</b>. Average round savings in this run are <b>${fmtPct(avgRound, 2)}</b>. Recovery success is <b>${fmtPct(metrics.recovery_success_rate_pct || 0, 2)}</b>.`;
}

function updateEnergyDashboard(runData, compareData = compareCache) {
  if (!runData) return;
  const metrics = runData.metrics;
  const logs = runData.fault_logs || [];
  updateEnergyCards(metrics, logs);

  Plotly.react(
    "energyBarChart",
    [{
      x: ["Baseline", "Scheduled"],
      y: [metrics.baseline_energy, metrics.scheduled_energy],
      type: "bar",
      marker: { color: [palette.accent4, palette.accent] },
      hovertemplate: "%{x}<br>%{y:.2f} energy units / round<extra></extra>",
    }],
    {
      ...baseLayout("Initial energy draw", "State", "Energy units / round"),
      yaxis: {
        title: "Energy units / round",
        range: boundedRange([metrics.baseline_energy, metrics.scheduled_energy], { minBound: 0, minSpan: 8 }) || undefined,
        automargin: true,
        gridcolor: "rgba(162,203,139,0.10)",
      },
    },
    plotConfig(),
  );

  Plotly.react(
    "energyRoundChart",
    logs.length ? [
      {
        x: logs.map((row) => row.round),
        y: logs.map((row) => row.scheduled_energy),
        mode: "lines+markers",
        line: { color: palette.accent, shape: "hv", width: 2.5 },
        marker: { size: 5 },
        fill: "tozeroy",
        fillcolor: "rgba(132,177,121,0.10)",
        name: "Scheduled energy / round",
        hovertemplate: "Round %{x}<br>Scheduled %{y:.2f}<extra></extra>",
      },
      {
        x: logs.map((row) => row.round),
        y: logs.map((row) => row.baseline_energy),
        mode: "lines",
        line: { color: palette.accent4, dash: "dash", width: 2 },
        name: "Baseline all-active energy",
        hovertemplate: "Round %{x}<br>Baseline %{y:.2f}<extra></extra>",
      },
    ] : [],
    {
      ...baseLayout("Round-wise energy draw", "Round", "Energy units / round"),
      yaxis: {
        title: "Energy units / round",
        range: logs.length ? boundedRange(logs.flatMap((row) => [row.scheduled_energy, row.baseline_energy]), { minBound: 0, minSpan: 12 }) : undefined,
        automargin: true,
        gridcolor: "rgba(162,203,139,0.10)",
      },
    },
    plotConfig(),
  );

  const compareRows = compareData ? getCompareRows("initial") : [[metrics.algo, { metrics }]];
  updateEnergyInsight(compareRows, metrics, logs);

  const energyVals = compareRows.map(([name, payload]) => ({ name, snap: snapshotForStage(payload, "initial"), metrics: payload.metrics }));

  Plotly.react(
    "lifetimeChart",
    [
      {
        x: energyVals.map((row) => row.name),
        y: energyVals.map((row) => row.snap?.energy_saved_pct ?? row.metrics.energy_saved_pct),
        type: "bar",
        marker: { color: palette.accent },
        name: "Energy saved (%)",
        hovertemplate: "%{x}<br>Energy saved %{y:.2f}%<extra></extra>",
      },
      {
        x: energyVals.map((row) => row.name),
        y: energyVals.map((row) => row.metrics.lifetime_improvement_pct),
        type: "bar",
        marker: { color: palette.accent4 },
        name: "Lifetime gain (%)",
        hovertemplate: "%{x}<br>Lifetime gain %{y:.2f}%<extra></extra>",
      },
    ],
    {
      ...baseLayout(compareData ? "Algorithm-wise energy saving and lifetime gain" : "Network lifetime impact", "Scenario", "Percent"),
      barmode: "group",
      xaxis: { ...baseLayout().xaxis, tickangle: -20, automargin: true, title: "Scenario", gridcolor: "rgba(162,203,139,0.10)" },
      yaxis: {
        title: "Percent",
        range: boundedRange(energyVals.flatMap((row) => [row.snap?.energy_saved_pct ?? 0, row.metrics.lifetime_improvement_pct]), { minBound: 0, minSpan: 8 }) || undefined,
        automargin: true,
        gridcolor: "rgba(162,203,139,0.10)",
      },
    },
    plotConfig(),
  );

  const pointColors = [palette.accent, palette.accent2, palette.accent3, palette.accent4, palette.accent5].slice(0, compareRows.length);
  Plotly.react(
    "energyTradeoffChart",
    [{
      x: energyVals.map((row) => row.snap?.coverage ?? row.metrics.coverage_scheduled),
      y: energyVals.map((row) => row.snap?.energy_saved_pct ?? row.metrics.energy_saved_pct),
      mode: "markers",
      marker: { size: 14, color: pointColors, line: { color: "#102019", width: 1 } },
      customdata: energyVals.map((row) => [row.name, row.snap?.active_nodes ?? row.metrics.n_on, row.snap?.backup_nodes ?? row.metrics.n_off]),
      hovertemplate: "%{customdata[0]}<br>Coverage %{x:.4f}<br>Energy saved %{y:.2f}%<br>Active %{customdata[1]} · Backups %{customdata[2]}<extra></extra>",
      name: "Initial trade-off",
    }],
    {
      ...baseLayout("Coverage vs energy trade-off", "Coverage", "Energy saved (%)"),
      showlegend: false,
      margin: { l: 62, r: 30, t: 64, b: 60 },
      yaxis: {
        title: "Energy saved (%)",
        range: boundedRange(energyVals.map((row) => row.snap?.energy_saved_pct ?? row.metrics.energy_saved_pct), { minBound: 0, maxBound: 100, minSpan: 10 }) || [0, 100],
        automargin: true,
        gridcolor: "rgba(162,203,139,0.10)",
      },
      xaxis: {
        title: "Coverage",
        range: boundedRange(energyVals.map((row) => row.snap?.coverage ?? row.metrics.coverage_scheduled), { minBound: 0.9, maxBound: 1.02, minSpan: 0.03 }) || [0.9, 1.02],
        automargin: true,
        gridcolor: "rgba(162,203,139,0.10)",
      },
    },
    plotConfig(),
  );
}

function renderCompareMetrics(elId, payload, stage) {
  const metrics = payload.metrics;
  const snap = snapshotForStage(payload, stage);
  const runtime = stage === "final"
    ? snap?.runtime_ms ?? metrics.total_runtime_ms ?? metrics.algorithm_runtime_ms
    : snap?.runtime_ms ?? metrics.algorithm_runtime_ms;
  const baselineLabel = stage === "final" ? "Final active" : "Initial active";
  $(elId).innerHTML = `
    <div><b>Active nodes:</b> ${snap?.active_nodes ?? metrics.n_on}</div>
    <div><b>Backup nodes:</b> ${snap?.backup_nodes ?? metrics.n_off}</div>
    <div><b>Failed nodes:</b> ${snap?.failed_nodes ?? metrics.n_unavailable ?? 0}</div>
    <div><b>Coverage:</b> ${fmtNumber(snap?.coverage ?? metrics.coverage_scheduled, 4)}</div>
    <div><b>Energy saved:</b> ${fmtPct(snap?.energy_saved_pct ?? metrics.energy_saved_pct, 2)}</div>
    <div><b>Lifetime gain:</b> ${fmtPct(metrics.lifetime_improvement_pct, 2)}</div>
    <div><b>Runtime:</b> ${fmtNumber(runtime, 2)} ms</div>
    <div class="muted-text" style="margin-top:10px">${baselineLabel} ${snap?.active_nodes ?? metrics.n_on} · Baseline ${fmtNumber(metrics.baseline_energy, 2)} → Scheduled ${fmtNumber(metrics.scheduled_energy, 2)}</div>
  `;
}

function currentCompareGroup() {
  return compareCache ? compareCache[compareStage] : null;
}

function currentCompareSlice() {
  const group = currentCompareGroup();
  if (!group) return null;
  if (compareView === "r1") return group.random_same_off;
  if (compareView === "r2") return group.random_greedy_cov;
  if (compareView === "gc") return group.greedy_cov;
  if (compareView === "ai") return group.ai_based;
  return group.voronoi;
}

function renderCompareViz() {
  const slice = currentCompareSlice();
  if (!slice) return;
  const snap = snapshotForStage(slice, compareStage);
  $("compareAlgoSummary").textContent = `${compareStage === "initial" ? "Initial" : "Final"} ${slice.metrics.algo}: active ${snap?.active_nodes ?? slice.metrics.n_on}, backups ${snap?.backup_nodes ?? slice.metrics.n_off}, failed ${snap?.failed_nodes ?? 0}, coverage ${fmtNumber(snap?.coverage ?? slice.metrics.coverage_scheduled, 4)}, energy saved ${fmtPct(snap?.energy_saved_pct ?? slice.metrics.energy_saved_pct, 2)}.`;
  drawSvgField("simSvg", slice);
}

function renderCompareSections(data) {
  $("compareFairnessNote").textContent = `${data.scenario?.fairness_note || "All algorithms use the same generated field and settings."} The initial section shows pre-failure scheduling only.`;
  renderCompareMetrics("cmp_init_voronoi", data.initial.voronoi, "initial");
  renderCompareMetrics("cmp_init_r1", data.initial.random_same_off, "initial");
  renderCompareMetrics("cmp_init_r2", data.initial.random_greedy_cov, "initial");
  renderCompareMetrics("cmp_init_gc", data.initial.greedy_cov, "initial");
  renderCompareMetrics("cmp_init_ai", data.initial.ai_based, "initial");

  renderCompareMetrics("cmp_final_voronoi", data.final.voronoi, "final");
  renderCompareMetrics("cmp_final_r1", data.final.random_same_off, "final");
  renderCompareMetrics("cmp_final_r2", data.final.random_greedy_cov, "final");
  renderCompareMetrics("cmp_final_gc", data.final.greedy_cov, "final");
  renderCompareMetrics("cmp_final_ai", data.final.ai_based, "final");
}

function renderExpTable(rows) {
  const tbody = document.querySelector("#expTable tbody");
  tbody.innerHTML = "";
  rows.forEach((row) => {
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td>${fmtNumber(row.field_area, 2)}</td>
      <td>${fmtNumber(row.field_width, 1)}</td>
      <td>${fmtNumber(row.field_height, 1)}</td>
      <td>${fmtNumber(row.density, 5)}</td>
      <td>${row.backup_nodes}</td>
      <td>${fmtNumber(row.coverage, 4)}</td>
      <td>${fmtNumber(row.energy_saved_pct, 2)}</td>
      <td>${fmtNumber(row.estimated_network_lifetime_rounds, 2)}</td>`;
    tbody.appendChild(tr);
  });
}

function renderExpCharts(rows) {
  const densities = rows.map((row) => row.density);
  Plotly.react("chartEnergyDensity", [{ x: densities, y: rows.map((row) => row.energy_saved_pct), mode: "lines+markers", line: { color: palette.accent }, name: "Energy saved (%)" }], baseLayout("Energy saved vs density", "Density", "Energy saved (%)"), plotConfig());
  Plotly.react("chartBackupDensity", [{ x: densities, y: rows.map((row) => row.backup_nodes), mode: "lines+markers", line: { color: palette.accent4 }, name: "Backup nodes" }], baseLayout("Backup nodes vs density", "Density", "Backup nodes"), plotConfig());
  Plotly.react("chartCoverageDensity", [{ x: densities, y: rows.map((row) => row.coverage), mode: "lines+markers", line: { color: palette.accent3 }, name: "Coverage" }], { ...baseLayout("Coverage vs density", "Density", "Coverage"), yaxis: { title: "Coverage", range: [0, 1.02], gridcolor: "rgba(162,203,139,0.10)" } }, plotConfig());
  Plotly.react("chartLifetimeDensity", [{ x: densities, y: rows.map((row) => row.estimated_network_lifetime_rounds), mode: "lines+markers", line: { color: palette.accent5 }, name: "Lifetime" }], baseLayout("Lifetime vs density", "Density", "Rounds"), plotConfig());
}

function renderPreviewTable(targetId, dataset) {
  const target = $(targetId);
  if (!dataset || !dataset.preview_rows || !dataset.preview_rows.length) {
    target.textContent = "No stored samples yet.";
    return;
  }
  const headers = [...dataset.feature_names, ...dataset.target_names];
  const rows = dataset.preview_rows.map((row) => `<tr>${headers.map((key) => `<td>${typeof row[key] === "number" ? fmtNumber(row[key], 4) : row[key]}</td>`).join("")}</tr>`).join("");
  target.innerHTML = `<table class="memory-table"><thead><tr>${headers.map((name) => `<th>${name}</th>`).join("")}</tr></thead><tbody>${rows}</tbody></table>`;
}

function updateMLQuickStatus(status) {
  const version = status?.model_version ?? "-";
  const ready = status?.last_trained_at ? `Model v${version} ready` : "No trained model yet";
  $("mlQuickStatus").innerHTML = `<b>${ready}</b><br>${status?.last_trained_at ? `Last trained: ${status.last_trained_at}` : "Train once to enable AI scheduling."}`;
  $("aiReadyBadge").textContent = status?.last_trained_at ? `Model v${version} loaded` : "No model loaded";
  $("aiReadyBadge").classList.toggle("ready", !!status?.last_trained_at);
}

function renderMLStatus(status) {
  latestMLStatus = status;
  updateMLQuickStatus(status);
  $("mlStatusCard").innerHTML = `
    <div class="status-grid">
      <div class="status-chip"><b>Model</b>${status.model_type}</div>
      <div class="status-chip"><b>Version</b>${status.model_version}</div>
      <div class="status-chip"><b>Last trained</b>${status.last_trained_at || "Not trained yet"}</div>
      <div class="status-chip"><b>Node samples</b>${status.total_node_samples}</div>
      <div class="status-chip"><b>Run samples</b>${status.total_run_samples}</div>
      <div class="status-chip"><b>Best classifier accuracy</b>${status.best_classifier_accuracy !== null ? fmtNumber(status.best_classifier_accuracy, 4) : "-"}</div>
      <div class="status-chip"><b>Best regressor R²</b>${status.best_regressor_r2 !== null ? fmtNumber(status.best_regressor_r2, 4) : "-"}</div>
      <div class="status-chip"><b>Saved best kept?</b>classifier=${status.classifier_model_kept ? "yes" : "no"}, regressor=${status.regressor_model_kept ? "yes" : "no"}</div>
    </div>
    <div class="muted-text" style="margin-top:10px">${status.training_note}</div>
  `;
}

function renderMLMemory(memory) {
  const status = memory.status || {};
  $("mlMemorySummary").innerHTML = `
    <div class="status-grid">
      <div class="status-chip"><b>Storage type</b>Local file-based memory</div>
      <div class="status-chip"><b>Training basis</b>${status.training_basis || "Synthetic WSN data"}</div>
      <div class="status-chip"><b>Node dataset size</b>${memory.node_dataset?.count ?? 0}</div>
      <div class="status-chip"><b>Run dataset size</b>${memory.run_dataset?.count ?? 0}</div>
      <div class="status-chip"><b>Node feature count</b>${memory.node_dataset?.feature_names?.length ?? 0}</div>
      <div class="status-chip"><b>Run feature count</b>${memory.run_dataset?.feature_names?.length ?? 0}</div>
    </div>`;

  const history = status.history || [];
  $("mlHistoryList").innerHTML = history.length
    ? history.slice().reverse().map((item) => `<div class="history-item"><b>v${item.model_version}</b> · ${item.trained_at || "-"}<br>node +${item.new_node_samples}, run +${item.new_run_samples}<br>acc ${item.classifier_accuracy ?? "-"}, R² ${item.regressor_r2 ?? "-"}</div>`).join("")
    : "No training history yet.";

  renderPreviewTable("nodeMemoryTable", memory.node_dataset);
  renderPreviewTable("runMemoryTable", memory.run_dataset);
}

async function loadMLStatus() {
  try {
    const res = await fetch(API_ML_STATUS);
    const data = await res.json();
    renderMLStatus(data);
  } catch (_error) {
    $("mlQuickStatus").textContent = "ML status unavailable. Start the backend to train or inspect the model.";
    $("mlStatusCard").textContent = "ML status unavailable. Start the backend to train or inspect the model.";
    $("aiReadyBadge").textContent = "Model unavailable";
  }
}

async function loadMLMemory() {
  try {
    const res = await fetch(API_ML_MEMORY);
    const data = await res.json();
    renderMLMemory(data);
  } catch (_error) {
    $("mlMemorySummary").textContent = "ML memory preview unavailable until the backend is running.";
    $("mlHistoryList").textContent = "ML history unavailable.";
  }
}

async function trainML() {
  const btn = $("btnTrainML");
  setBusy(btn, true, "Training…");
  try {
    const res = await fetch(API_ML_TRAIN, { method: "POST" });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || "ML training failed");
    renderMLStatus(data.training);
    await loadMLMemory();
  } catch (error) {
    $("mlQuickStatus").textContent = `Training failed: ${error.message}`;
    $("mlStatusCard").textContent = `Training failed: ${error.message}`;
  } finally {
    setBusy(btn, false);
  }
}

async function runSim() {
  const btn = $("runBtn");
  setBusy(btn, true);
  try {
    const payload = payloadFromUI();
    const endpoint = payload.enable_ai ? API_ML_PREDICT : API_RUN;
    const res = await fetch(endpoint, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || "Run request failed");
    runCache = data;
    updateRunCards(data.metrics);
    renderMetricsTable(data.metrics);
    renderRunViz(data);
    renderRunCharts(data);
    updateEnergyDashboard(data, compareCache);
    setTab("tab_run");
  } catch (error) {
    console.error(error);
    alert(`Run failed: ${error.message}`);
  } finally {
    setBusy(btn, false);
  }
}

async function runCompare() {
  const btn = $("compareBtn");
  setBusy(btn, true);
  try {
    const res = await fetch(API_COMPARE, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payloadFromUI()),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || "Compare request failed");
    compareCache = data;
    renderCompareSections(data);
    compareView = "voronoi";
    compareStage = "initial";
    $("compareToggles").classList.remove("hidden");
    $("compareStageToggles").classList.remove("hidden");
    document.querySelectorAll("#compareToggles .segbtn").forEach((button) => button.classList.toggle("active", button.dataset.algo === "voronoi"));
    document.querySelectorAll("#compareStageToggles .segbtn").forEach((button) => button.classList.toggle("active", button.dataset.stage === "initial"));
    renderCompareViz();
    if (runCache) updateEnergyDashboard(runCache, compareCache);
    setTab("tab_compare");
  } catch (error) {
    console.error(error);
    alert(`Compare failed: ${error.message}`);
  } finally {
    setBusy(btn, false);
  }
}

async function runExperiment() {
  const btn = $("expBtn");
  setBusy(btn, true);
  try {
    const res = await fetch(API_DENSITY, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payloadFromUI()) });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || "Experiment failed");
    expRows = data.rows || [];
    renderExpTable(expRows);
    renderExpCharts(expRows);
    setTab("tab_experiment");
  } catch (error) {
    console.error(error);
    alert(`Experiment failed: ${error.message}`);
  } finally {
    setBusy(btn, false);
  }
}

function downloadBlob(blob, filename) {
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
}

bindSlider("n_nodes", "n_nodes_val");
updateControlDescriptions();
clampFailureProbability();
clampThresholdCoeff();

$("f_model").addEventListener("change", updateControlDescriptions);
$("r_model").addEventListener("change", updateControlDescriptions);
$("sensor_type").addEventListener("change", updateControlDescriptions);
$("failp").addEventListener("change", clampFailureProbability);
$("th").addEventListener("change", clampThresholdCoeff);

$("runBtn").addEventListener("click", runSim);
$("compareBtn").addEventListener("click", runCompare);
$("expBtn").addEventListener("click", runExperiment);
$("btnTrainML").addEventListener("click", trainML);

$("exportBtn").addEventListener("click", async () => {
  try {
    const res = await fetch(API_EXPORT, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payloadFromUI()) });
    const blob = await res.blob();
    downloadBlob(blob, "wsn_run_export.csv");
  } catch (error) {
    console.error(error);
    alert("CSV export failed.");
  }
});

$("expCsvBtn").addEventListener("click", () => {
  if (!expRows.length) {
    alert("Run experiment first.");
    return;
  }
  const header = ["field_area", "field_width", "field_height", "density", "backup_nodes", "coverage", "energy_saved_pct", "estimated_network_lifetime_rounds"];
  const lines = [header.join(",")];
  expRows.forEach((row) => lines.push([row.field_area, row.field_width, row.field_height, row.density, row.backup_nodes, row.coverage, row.energy_saved_pct, row.estimated_network_lifetime_rounds].join(",")));
  downloadBlob(new Blob([lines.join("\n")], { type: "text/csv" }), "wsn_experiment_density.csv");
});

document.querySelectorAll(".tab").forEach((button) => {
  button.addEventListener("click", () => setTab(button.dataset.tab));
});

document.querySelectorAll("#compareToggles .segbtn").forEach((button) => {
  button.addEventListener("click", () => {
    document.querySelectorAll("#compareToggles .segbtn").forEach((node) => node.classList.remove("active"));
    button.classList.add("active");
    compareView = button.dataset.algo;
    renderCompareViz();
  });
});

document.querySelectorAll("#compareStageToggles .segbtn").forEach((button) => {
  button.addEventListener("click", () => {
    document.querySelectorAll("#compareStageToggles .segbtn").forEach((node) => node.classList.remove("active"));
    button.classList.add("active");
    compareStage = button.dataset.stage;
    renderCompareViz();
  });
});

window.addEventListener("resize", () => {
  if (runCache) renderRunViz(runCache);
  if (compareCache) renderCompareViz();
  resizeAllPlots();
});

checkHealth();
loadMLStatus();
loadMLMemory();
runSim();
