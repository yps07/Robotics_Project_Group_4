// Builds the Group 4 project deck. Run from the scratchpad deck folder:
//   node build_deck.js <repo_root> <out.pptx>
const fs = require("fs");
const path = require("path");
const pptxgen = require("pptxgenjs");
const React = require("react");
const ReactDOMServer = require("react-dom/server");
const sharp = require("sharp");
const fa = require("react-icons/fa");
const { applyTheme } = require(process.env.SKILL_DIR + "/scripts/apply_theme.js");

const REPO = process.argv[2];
const OUT = process.argv[3];
const IMG = path.join(__dirname, "img");

// ------------------------------------------------------------------ theme
const THEME = {
  name: "Group4 Robotics",
  headFontFace: "Cambria",
  bodyFontFace: "Arial",
  colors: {
    dk1: "1B2430", lt1: "FFFFFF", dk2: "16202B", lt2: "EEF1F4",
    accent1: "1F8A8A",   // DeLaN (teal)
    accent2: "E0892E",   // MLP (amber)
    accent3: "4A5868",   // analytical (slate)
    accent4: "A7B1BC",   // PD only (grey)
    accent5: "C0392B",   // failure / diverged
    accent6: "7FC7C0",   // light teal for dark slides
    hlink: "1F8A8A", folHlink: "4A5868",
  },
};
const HEX = THEME.colors;
const MODEL = { analytical: HEX.accent3, delan: HEX.accent1, mlp: HEX.accent2, pd: HEX.accent4 };

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE";            // 13.333 x 7.5 in
pres.theme = { headFontFace: THEME.headFontFace, bodyFontFace: THEME.bodyFontFace };
pres.title = "Deep Lagrangian Network for Computed-Torque Tracking";
pres.subject = "Robotics (ECE670/CSE633) course project, Group 4";
pres.author = "Group 4";
const C = pres.SchemeColor;

const W = 13.333, M = 0.6;
const FOOTER = "Group 4  ·  Robotics (ECE670/CSE633)  ·  DeLaN for computed-torque tracking";

// ------------------------------------------------------------------ layouts
pres.defineSlideMaster({
  title: "TITLE_DARK",
  background: { color: C.text2 },
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: M, y: 1.5, w: 6.6, h: 2.55,
        fontFace: THEME.headFontFace, fontSize: 36, bold: true, color: C.background1, valign: "bottom",
        margin: 0 }, text: "" } },
    { placeholder: { options: { name: "body", type: "body", x: M, y: 4.2, w: 6.6, h: 1.9,
        fontSize: 18, color: C.accent6, valign: "top", margin: 0 }, text: "" } },
  ],
});
pres.defineSlideMaster({
  title: "CONTENT",
  background: { color: C.background1 },
  margin: [0.5, 0.6, 0.6, 0.6],
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: M, y: 0.35, w: W - 2 * M, h: 0.85,
        fontFace: THEME.headFontFace, fontSize: 34, bold: true, color: C.text1, valign: "middle",
        margin: 0 }, text: "" } },
    { text: { text: FOOTER, options: { x: M, y: 7.02, w: 9, h: 0.3, fontSize: 10, color: C.accent3,
        margin: 0, valign: "middle" } } },
  ],
  slideNumber: { x: W - M - 0.6, y: 7.02, w: 0.6, h: 0.3, fontSize: 10, color: C.accent3, align: "right" },
});
pres.defineSlideMaster({
  title: "CLOSING_DARK",
  background: { color: C.text2 },
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: M, y: 0.45, w: W - 2 * M, h: 0.9,
        fontFace: THEME.headFontFace, fontSize: 36, bold: true, color: C.background1, valign: "middle",
        margin: 0 }, text: "" } },
  ],
  slideNumber: { x: W - M - 0.6, y: 7.02, w: 0.6, h: 0.3, fontSize: 10, color: C.accent4, align: "right" },
});

// ------------------------------------------------------------------ helpers
async function iconPng(Icon, color = "FFFFFF", size = 256) {
  const svg = ReactDOMServer.renderToStaticMarkup(React.createElement(Icon, { color: "#" + color, size }));
  const buf = await sharp(Buffer.from(svg)).png().toBuffer();
  return "image/png;base64," + buf.toString("base64");
}
function imgData(file) {
  return "image/png;base64," + fs.readFileSync(file).toString("base64");
}
let objN = 0;
const nm = (s) => `${s}_${++objN}`;

function text(slide, t, o) {
  slide.addText(t, { isTextBox: true, margin: 0, valign: "top", fontSize: 16, color: C.text1,
                     objectName: nm("text"), ...o });
}
function card(slide, x, y, w, h, fill = C.background2) {
  slide.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, fill: { color: fill },
    line: { type: "none" }, rectRadius: 0.08, objectName: nm("card") });
}
async function iconCircle(slide, Icon, x, y, d, fill) {
  slide.addShape(pres.shapes.OVAL, { x, y, w: d, h: d, fill: { color: fill }, line: { type: "none" },
    objectName: nm("icon_bg") });
  const pad = d * 0.24;
  slide.addImage({ data: await iconPng(Icon), x: x + pad, y: y + pad, w: d - 2 * pad, h: d - 2 * pad,
    objectName: nm("icon"), altText: "" });
}
// a card with an icon circle, a bold header and body text
async function iconCard(slide, { x, y, w, h, Icon, color, head, body, bodySize = 14 }) {
  card(slide, x, y, w, h);
  await iconCircle(slide, Icon, x + 0.25, y + 0.25, 0.6, color);
  text(slide, head, { x: x + 1.0, y: y + 0.25, w: w - 1.2, h: 0.6, fontSize: 18, bold: true,
                      valign: "middle" });
  text(slide, body, { x: x + 0.25, y: y + 1.0, w: w - 0.5, h: h - 1.15, fontSize: bodySize });
}
function stat(slide, { x, y, w, big, label, color, bigSize = 44, labelSize = 14, labelColor }) {
  text(slide, big, { x, y, w, h: 0.85, fontSize: bigSize, bold: true, color, fontFace: THEME.headFontFace,
                     valign: "bottom" });
  text(slide, label, { x, y: y + 0.95, w, h: 0.8, fontSize: labelSize, color: labelColor || C.text1 });
}
function bullets(items, size = 16, extra = {}) {
  return items.map((it, i) => {
    const runs = Array.isArray(it) ? it : [{ text: it }];
    return runs.map((r, k) => ({
      text: r.text,
      options: { ...(k === 0 ? { bullet: { indent: 18 }, paraSpaceAfter: 8 } : {}), fontSize: size,
                 ...(r.options || {}), ...extra, ...(k === runs.length - 1 && i < items.length - 1
                   ? { breakLine: true } : {}) },
    }));
  }).flat();
}
// one style for every chart in the deck
function chartStyle(extra) {
  return {
    catAxisLabelColor: HEX.accent3, valAxisLabelColor: HEX.accent3,
    catAxisLabelFontFace: "+mn-lt", valAxisLabelFontFace: "+mn-lt",
    catAxisLabelFontSize: 13, valAxisLabelFontSize: 12,
    valGridLine: { color: "E3E7EB", size: 0.75 }, catGridLine: { style: "none" },
    catAxisLineShow: true, valAxisLineShow: false, catAxisLineColor: "C9D0D7",
    showLegend: true, legendPos: "b", legendFontSize: 13, legendFontFace: "+mn-lt", legendColor: HEX.dk1,
    dataLabelFontFace: "+mn-lt", dataLabelFontSize: 11, dataLabelColor: HEX.dk1,
    titleFontFace: "+mn-lt", titleFontSize: 15, titleColor: HEX.dk1,
    valAxisTitleFontFace: "+mn-lt", valAxisTitleFontSize: 12, valAxisTitleColor: HEX.accent3,
    ...extra,
  };
}
function content(title, section) {
  const s = pres.addSlide({ masterName: "CONTENT", sectionTitle: section });
  s.addText(title, { placeholder: "title" });
  return s;
}

// ------------------------------------------------------------------ slides
async function build() {
  const S1 = "Introduction", S2 = "Setup", S3 = "Results", S4 = "Demo and conclusions";

  // 1. Title ---------------------------------------------------------------
  pres.addSection({ title: S1 });
  let s = pres.addSlide({ masterName: "TITLE_DARK", sectionTitle: S1 });
  s.addImage({ path: path.join(IMG, "hero.png"), x: 7.4, y: 0, w: 5.933, h: 7.5,
    sizing: { type: "cover", w: 5.933, h: 7.5 }, objectName: "hero_render",
    altText: "MuJoCo render of the dm_control manipulator arm" });
  text(s, "ROBOTICS (ECE670/CSE633)  ·  COURSE PROJECT  ·  GROUP 4", { x: M, y: 1.0, w: 6.6, h: 0.4,
    fontSize: 13, bold: true, color: C.accent6, charSpacing: 1 });
  s.addText("Deep Lagrangian Network for computed-torque tracking of the dm_control manipulator arm",
    { placeholder: "title" });
  s.addText("Physics-informed inverse dynamics, compared with a black-box MLP and the analytical "
    + "Euler–Lagrange model, in prediction and in closed-loop tracking", { placeholder: "body" });
  text(s, "Five workstreams: data · analytical model · MLP · DeLaN · evaluation and control",
    { x: M, y: 6.55, w: 6.6, h: 0.4, fontSize: 13, color: C.accent4 });
  s.addNotes("Introduce the project: we learn the inverse dynamics of the 4-joint arm of the dm_control "
    + "manipulator with a Deep Lagrangian Network, compare it with a black-box MLP and the analytical "
    + "Euler-Lagrange model, and use each one inside a computed-torque controller in MuJoCo.");

  // 2. Project in one slide --------------------------------------------------
  s = content("Physics structure makes the model usable", S1);
  text(s, "Same data, same training subsets, same metrics. Three results stand out:",
    { x: M, y: 1.4, w: 12, h: 0.45, fontSize: 18, color: C.accent3 });
  const cards2 = [
    { big: "100 %", vs: "vs ≤ 11 %", label: "of mass matrices positive-definite: DeLaN by construction, "
      + "the MLP's implied ∂τ/∂q̈ almost never", c: MODEL.delan, Icon: fa.FaShieldAlt },
    { big: "0 %", vs: "vs 31 %", label: "of 2 s open-loop rollouts diverge (in range); 92 % for the MLP "
      + "outside the training range", c: MODEL.delan, Icon: fa.FaChartLine },
    { big: "2.7×", vs: "vs 38×", label: "growth of the tracking error outside the training range "
      + "(DeLaN vs MLP); the MLP ends up worse than PD", c: MODEL.delan, Icon: fa.FaBullseye },
  ];
  for (let i = 0; i < 3; i++) {
    const x = M + i * 4.1, y = 2.15, w = 3.85, h = 4.4, cd = cards2[i];
    card(s, x, y, w, h);
    await iconCircle(s, cd.Icon, x + 0.3, y + 0.35, 0.65, cd.c);
    text(s, cd.big, { x: x + 0.3, y: y + 1.25, w: w - 0.6, h: 1.0, fontSize: 54, bold: true, color: cd.c,
                      fontFace: THEME.headFontFace, valign: "bottom" });
    text(s, cd.vs, { x: x + 0.3, y: y + 2.3, w: w - 0.6, h: 0.5, fontSize: 22, bold: true,
                     color: MODEL.mlp, fontFace: THEME.headFontFace });
    text(s, cd.label, { x: x + 0.3, y: y + 2.95, w: w - 0.6, h: 1.3, fontSize: 15 });
  }
  text(s, [{ text: "DeLaN  ", options: { color: MODEL.delan, bold: true } },
           { text: "vs  ", options: { color: C.accent3 } },
           { text: "MLP", options: { color: MODEL.mlp, bold: true } },
           { text: "   (100 % of the training data; MuJoCo ground truth)", options: { color: C.accent3 } }],
    { x: M, y: 6.6, w: 12, h: 0.35, fontSize: 13 });
  s.addNotes("Executive summary. One-step torque accuracy alone hides the difference: the MLP is only 3x "
    + "worse than DeLaN in range. The structure (SPD inertia, gravity from a potential, explicit damping) "
    + "is what makes DeLaN invertible for simulation and robust outside the training range.");

  // 3. Problem statement ------------------------------------------------------
  pres.addSection({ title: S2 });
  s = content("Problem statement", S2);
  const ps = [
    { Icon: fa.FaRobot, c: MODEL.analytical, head: "System", body: "dm_control manipulator (bring_ball): "
      + "4 actuated hinge joints of a planar arm, τ = gear × action, gear = [12, 8, 4, 2] N·m" },
    { Icon: fa.FaBrain, c: MODEL.delan, head: "What is learned", body: "Inverse dynamics τ = f(q, q̇, q̈); "
      + "DeLaN also yields the mass matrix M(q) and potential energy V(q)" },
    { Icon: fa.FaExchangeAlt, c: MODEL.mlp, head: "Inputs → outputs", body: "q, q̇, q̈ of the 4 arm joints "
      + "→ τ (4 joint torques); DeLaN also outputs the 4×4 M(q) and V(q)" },
    { Icon: fa.FaCheckCircle, c: MODEL.analytical, head: "Measures of success", body: "One-step torque "
      + "error · rollout RMSE at 0.5/1/2 s · tracking RMSE on 10 trajectories · 10/30/50/100 % data · "
      + "OOD error · M(q) positive-definite" },
  ];
  for (let i = 0; i < 4; i++) {
    const col = i % 2, row = Math.floor(i / 2);
    await iconCard(s, { x: M + col * 4.25, y: 1.5 + row * 2.75, w: 4.0, h: 2.5, ...ps[i], color: ps[i].c,
                        bodySize: 14 });
  }
  s.addImage({ path: path.join(IMG, "arm.png"), x: 9.15, y: 1.5, w: 3.58, h: 2.685,
    objectName: "arm_render", altText: "Render of the 4-joint arm" });
  text(s, "Why physics should help: τ = M(q)q̈ + C(q,q̇)q̇ + g(q) + Dq̇ has a known structure. "
    + "Building it in should improve data efficiency, extrapolation and controller suitability.",
    { x: 9.15, y: 4.4, w: 3.58, h: 2.35, fontSize: 14, color: C.text1 });
  s.addNotes("Inputs and outputs follow our approved statement. The physics argument: an SPD mass matrix, "
    + "gravity as the gradient of a potential and non-negative damping constrain the hypothesis space.");

  // 4. System ------------------------------------------------------------------
  s = content("The arm: damped, and stiff at the wrist", S2);
  const rows = [["Joint", "Range (rad)", "Damping D (N·m·s/rad)", "Gear (N·m)"],
    ["root", "unlimited", "2.0", "12"], ["shoulder", "±2.79", "1.5", "8"],
    ["elbow", "±2.79", "1.0", "4"], ["wrist", "±2.44", "0.5", "2"]];
  s.addTable(rows.map((r, i) => r.map((v) => ({ text: v, options: {
      bold: i === 0, color: i === 0 ? C.background1 : C.text1, fill: { color: i === 0 ? C.accent3 : (i % 2
        ? C.background1 : C.background2) }, fontSize: 15, valign: "middle", margin: [0, 0.1, 0, 0.1] } }))),
    { x: M, y: 1.5, w: 7.2, colW: [1.5, 1.7, 2.5, 1.5], rowH: 0.48, border: { type: "none" },
      objectName: "joint_table" });
  text(s, bullets([
    "Data model: bring_ball MJCF with the gripper hinges removed (fingers rigid) and collisions off, so "
      + "the 4 arm coordinates fully determine τ",
    "Physics step 1 ms, control step 10 ms (zero-order hold)",
    [{ text: "Wrist: M/D ≈ 60 µs, ", options: { bold: true } },
     { text: "far below the 10 ms control step. This shapes both integration and control (slides 12 and 13)" }],
  ], 15), { x: M, y: 4.2, w: 7.2, h: 2.6 });
  s.addImage({ path: path.join(IMG, "arm.png"), x: 8.3, y: 1.5, w: 4.43, h: 3.32,
    objectName: "arm_render_2", altText: "Render of the arm model" });
  text(s, "Mass-matrix eigenvalues span 0.00003 to 0.027 (condition number ≈ 900)",
    { x: 8.3, y: 5.0, w: 4.43, h: 0.8, fontSize: 14, color: C.accent3 });
  s.addNotes("Parameters come from the MuJoCo model (Member 2's extraction): no armature or friction loss, "
    + "only viscous damping. The tiny wrist inertia relative to its damping is the key physical fact that "
    + "shapes the rollout integrator and the controller.");

  // 5. Pipeline -------------------------------------------------------------------
  s = content("Five workstreams, one shared pipeline", S2);
  const ws = [
    { t: "A · Data", d: "Reduced model, excitation, 4 splits", Icon: fa.FaDatabase,
      c: MODEL.analytical },
    { t: "B · Analytical", d: "Euler–Lagrange M, C, g, D from MuJoCo; validated vs mj_inverse",
      Icon: fa.FaSquareRootAlt, c: MODEL.analytical },
    { t: "C · MLP", d: "Black-box inverse dynamics; 10–100 % data", Icon: fa.FaProjectDiagram,
      c: MODEL.mlp },
    { t: "D · DeLaN", d: "M = LLᵀ + εI, V(q), learnable damping", Icon: fa.FaAtom, c: MODEL.delan },
    { t: "E · Evaluation & control", d: "Shared interface, rollouts, computed-torque tracking",
      Icon: fa.FaCogs, c: MODEL.analytical },
  ];
  // A (left) fans out to B, C, D (middle), which fan in to E (right); 0.55" gaps hold the arrows
  const bx = [M, 3.75, 3.75, 3.75, 8.35], by = [3.05, 1.5, 3.05, 4.6, 3.05];
  const bw = [2.6, 4.0, 4.0, 4.0, 4.38];
  for (let i = 0; i < 5; i++) {
    const x = bx[i], y = by[i], w = bw[i], h = 1.35;
    card(s, x, y, w, h);
    await iconCircle(s, ws[i].Icon, x + 0.2, y + 0.36, 0.62, ws[i].c);
    text(s, ws[i].t, { x: x + 0.95, y: y + 0.15, w: w - 1.1, h: 0.42, fontSize: 17, bold: true });
    text(s, ws[i].d, { x: x + 0.95, y: y + 0.58, w: w - 1.1, h: 0.7, fontSize: 13, color: C.accent3 });
  }
  const arrow = (x1, y1, x2, y2) => s.addShape(pres.shapes.LINE, { x: x1, y: Math.min(y1, y2),
    w: x2 - x1, h: Math.max(Math.abs(y2 - y1), 0.001), flipV: y2 < y1,
    line: { color: HEX.accent4, width: 2, endArrowType: "triangle" }, objectName: nm("arrow") });
  const midY = (i) => by[i] + 0.675;
  for (const i of [1, 2, 3]) arrow(bx[0] + bw[0] + 0.05, midY(0), bx[i] - 0.05, midY(i));
  for (const i of [1, 2, 3]) arrow(bx[i] + bw[i] + 0.05, midY(i), bx[4] - 0.05, midY(4));
  text(s, "Every model is trained and scored on the same official splits, identical nested subsets "
    + "(seed 42) and the same metric code.", { x: M, y: 6.2, w: 12.1, h: 0.5, fontSize: 15, color: C.accent3 });
  s.addNotes("Explain who owned what (Members 1 to 5) and how the hand-offs work: Member 1's dataset feeds "
    + "B, C and D; Member 5 wraps all three models behind one interface for rollouts and control.");

  // 6. Data ---------------------------------------------------------------------------
  s = content("Data: three excitations and an OOD split", S2);
  const ex = [
    { h: "Random", d: "uniform actions, amplitude 0.35, new value every 10 ms" },
    { h: "Multi-sine", d: "0.1–1 Hz per joint, amplitude 0.15–0.35" },
    { h: "Chirp", d: "0.05–0.2 Hz sweeping up to 0.5–2 Hz" },
  ];
  for (let i = 0; i < 3; i++) {
    const y = 1.5 + i * 1.15;
    card(s, M, y, 5.6, 1.0);
    text(s, ex[i].h, { x: M + 0.3, y: y + 0.12, w: 1.6, h: 0.76, fontSize: 17, bold: true, valign: "middle" });
    text(s, ex[i].d, { x: M + 2.0, y: y + 0.12, w: 3.4, h: 0.76, fontSize: 14, valign: "middle" });
  }
  text(s, bullets([
    "OOD: amplitude 0.65 and higher frequencies; every OOD row lies outside the training min/max in at "
      + "least one of q, q̇, q̈",
    "No measurement noise; contacts asserted to be zero",
    "Normalisation statistics from the training split only",
  ], 14), { x: M, y: 5.05, w: 5.6, h: 1.8 });
  s.addChart(pres.charts.BAR, [{ name: "Samples", labels: ["Train (40 ep.)", "Validation (10)",
      "Test (10)", "OOD (10)"], values: [20000, 2000, 2000, 2000] }],
    chartStyle({ x: 6.75, y: 1.5, w: 5.98, h: 5.3, barDir: "col", chartColors: [HEX.accent3],
      showLegend: false, showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: "#,##0",
      showTitle: true, title: "Samples per split (episodes of 10 s)", valAxisLabelFormatCode: "#,##0",
      objectName: "split_chart" }));
  s.addNotes("Samples are spread over each episode rather than taken as a prefix. Even the 10 % subset "
    + "contains all 40 training episodes, which explains the flat DeLaN data-efficiency curve later.");

  // 7. Analytical model -------------------------------------------------------------------
  s = content("Analytical model: exact on this data", S2);
  card(s, M, 1.5, 7.3, 2.0, C.text2);
  text(s, "τ = M(q) q̈ + C(q, q̇) q̇ + g(q) + D q̇", { x: M + 0.4, y: 1.75, w: 6.5, h: 0.8, fontSize: 30,
    bold: true, color: C.background1, fontFace: THEME.headFontFace, valign: "middle" });
  text(s, "M from link Jacobians · C from Christoffel symbols of ∂M/∂q · g = ∂V/∂q · D from MuJoCo",
    { x: M + 0.4, y: 2.65, w: 6.5, h: 0.6, fontSize: 14, color: C.accent6 });
  text(s, bullets([
    "Masses, centres of mass and inertias extracted from the MuJoCo model; finger bodies lumped into the wrist",
    "∂M/∂q by complex-step differentiation (exact to machine precision)",
    "Used by Workstream E as the ground-truth model and as the floor for rollouts and tracking",
  ], 15), { x: M, y: 3.85, w: 7.3, h: 2.9 });
  stat(s, { x: 8.4, y: 1.5, w: 4.3, big: [{ text: "8.9 × 10" }, { text: "−16", options: { superscript: true } }], label: "N·m maximum torque error against "
    + "mujoco.mj_inverse on 1,000 random states", color: MODEL.analytical, bigSize: 40 });
  stat(s, { x: 8.4, y: 3.85, w: 4.3, big: [{ text: "1.7 × 10" }, { text: "−16", options: { superscript: true } }], label: [{ text: "N·m RMSE on the official test split (3.1 × 10" }, { text: "−16", options: { superscript: true } }, { text: " on OOD)" }], color: MODEL.analytical, bigSize: 40 });
  s.addNotes("This mirrors Assignment 2, with parameters from MuJoCo instead of the URDF. Because the data "
    + "come from the same rigid-body model, the analytical model is exact; it is the upper bound for every "
    + "learned model.");

  // 8. Two learned models ------------------------------------------------------------------
  s = content("Two learned models, trained on identical data", S2);
  const cols = [
    { head: "Black-box MLP", c: MODEL.mlp, Icon: fa.FaProjectDiagram, items: [
      "13 → 256 → 256 → 128 → 4, tanh",
      "Inputs z-scored; root angle as (sin, cos)",
      "AdamW, lr 0.001, weight decay 0.0001, early stopping",
      "Learns τ directly: no structure, no M(q)",
    ] },
    { head: "Deep Lagrangian Network", c: MODEL.delan, Icon: fa.FaAtom, items: [
      "L(q) network (128 × 3, tanh) → M = LLᵀ + εI, SPD by construction",
      "V(q) network → gravity g = ∂V/∂q by autograd",
      "Coriolis from ∂M/∂q (torch.func), learnable damping D = softplus(d) ≥ 0",
      "Raw physical units, float64; same loss (normalised torque MSE)",
    ] },
  ];
  for (let i = 0; i < 2; i++) {
    const x = M + i * 6.2, w = 5.9;
    card(s, x, 1.5, w, 3.95);
    await iconCircle(s, cols[i].Icon, x + 0.3, 1.75, 0.7, cols[i].c);
    text(s, cols[i].head, { x: x + 1.2, y: 1.75, w: w - 1.4, h: 0.7, fontSize: 22, bold: true,
      color: cols[i].c, valign: "middle", fontFace: THEME.headFontFace });
    text(s, bullets(cols[i].items, 17, { paraSpaceAfter: 12 }), { x: x + 0.3, y: 2.75, w: w - 0.6, h: 3.1 });
  }
  text(s, "Both: nested 10 / 30 / 50 / 100 % subsets (seed 42), validation-only tuning, test and OOD "
    + "untouched until the end.", { x: M, y: 5.75, w: 12.1, h: 0.5, fontSize: 16, color: C.accent3 });
  s.addNotes("DeLaN needs the damping term: a pure Lagrangian is conservative and cannot fit the data "
    + "(ablation: R² 0.14 without damping). Damping got a 10x learning rate so the 10 % run converges.");

  // 9. One-step error -------------------------------------------------------------------------
  pres.addSection({ title: S3 });
  s = content("One-step error: 3× better, 18× out of range", S3);
  s.addChart(pres.charts.BAR, [
      { name: "MLP", labels: ["Test split", "OOD split"], values: [0.0440, 0.450] },
      { name: "DeLaN", labels: ["Test split", "OOD split"], values: [0.0147, 0.0250] }],
    chartStyle({ x: M, y: 1.45, w: 7.4, h: 5.35, barDir: "col", chartColors: [MODEL.mlp, MODEL.delan],
      showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: "0.000", barGapWidthPct: 60,
      valAxisLogScaleBase: 10, valAxisMinVal: 0.01, valAxisMaxVal: 1, valAxisLabelFormatCode: "0.00",
      showValAxisTitle: true, valAxisTitle: "Torque RMSE (N·m, log scale)", showTitle: true,
      title: "Torque RMSE, 100 % training data", objectName: "one_step_chart" }));
  stat(s, { x: 8.5, y: 1.5, w: 4.2, big: "1.7×", label: "DeLaN error growth from test to OOD "
    + "(MLP: 10×)", color: MODEL.delan });
  stat(s, { x: 8.5, y: 3.45, w: 4.2, big: "R² > 0.9999", label: [{ text: "DeLaN on test and OOD; analytical model exact (≈ 10" }, { text: "−16", options: { superscript: true } }, { text: " N·m)" }], color: MODEL.delan, bigSize: 36 });
  text(s, "Per-joint and per-fraction results: delan/README.md, black_box_mlp/README.md",
    { x: 8.5, y: 5.9, w: 4.2, h: 0.8, fontSize: 12, color: C.accent3 });
  s.addNotes("These are the official splits, recomputed by Workstream E with Member 3's metric function and "
    + "cross-checked against Members 3 and 4. Analytical error is ~1e-16 and is off the chart.");

  // 10. Physics recovered -------------------------------------------------------------------------
  s = content("DeLaN recovers the dominant physics", S3);
  s.addImage({ path: path.join(REPO, "delan/results/physics_comparison.png"), x: M, y: 1.45, w: 12.1,
    h: 3.48, objectName: "physics_comparison", altText: "Learned vs analytical mass matrix, gravity, damping" });
  const ph = [
    { big: "0.1 %", label: "max error of the learned damping D = [1.999, 1.500, 1.000, 0.500]" },
    { big: "6.8 %", label: "relative error of gravity g(q); RMSE 0.014 N·m on torques up to 0.62 N·m" },
    { big: "33 %", label: "error of M(q): small elbow/wrist inertias are weakly excited by the data" },
  ];
  for (let i = 0; i < 3; i++) {
    stat(s, { x: M + i * 4.1, y: 5.0, w: 3.85, big: ph[i].big, label: ph[i].label,
              color: i === 2 ? MODEL.mlp : MODEL.delan, bigSize: 32, labelSize: 13 });
  }
  s.addNotes("Damping dominates the torque (root: 2.37 N·m RMS from D q̇ vs 0.61 from M q̈), so it is "
    + "identified almost exactly. The wrist/elbow inertia moves the torque by only ~0.01-0.03 N·m, the same "
    + "order as the model error, so it is over-estimated. Figure from Member 4 (delan/results).");

  // 11. Mass matrix PD ------------------------------------------------------------------------------
  s = content("Positive-definite inertia: DeLaN only", S3);
  s.addChart(pres.charts.BAR, [
      { name: "MLP (implied ∂τ/∂q̈)", labels: ["10 %", "30 %", "50 %", "100 %"], values: [0.1, 0.4, 0.5, 1.8] },
      { name: "DeLaN (L Lᵀ + εI)", labels: ["10 %", "30 %", "50 %", "100 %"], values: [100, 100, 100, 100] }],
    chartStyle({ x: M, y: 1.45, w: 7.4, h: 5.35, barDir: "col", chartColors: [MODEL.mlp, MODEL.delan],
      showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: "0.0", valAxisMaxVal: 110,
      valAxisMinVal: 0, valAxisMajorUnit: 25, showValAxisTitle: true, valAxisTitle: "Positive-definite (%)",
      showTitle: true, title: "10,000 uniform configurations within the joint limits, by training data",
      objectName: "pd_chart" }));
  stat(s, { x: 8.5, y: 1.5, w: 4.2, big: "56,000", label: "DeLaN mass matrices checked by Member 4: all "
    + "SPD, Cholesky succeeds, exactly symmetric", color: MODEL.delan, bigSize: 40 });
  stat(s, { x: 8.5, y: 3.55, w: 4.2, big: "≤ 10.6 %", label: "MLP states with a PD implied inertia (test "
    + "split); it is also non-symmetric (asymmetry ≈ 1.4)", color: MODEL.mlp, bigSize: 40 });
  text(s, "Why it matters: forward dynamics and simulation need to solve M(q) q̈ = τ − c − g − Dq̇. An "
    + "indefinite M makes that solve ill-posed.", { x: 8.5, y: 5.65, w: 4.2, h: 1.15, fontSize: 14, color: C.text1 });
  s.addNotes("The MLP has no mass matrix, so we use the Jacobian of its torque w.r.t. q̈ (autograd). It is "
    + "not symmetric and almost never positive-definite, and its error vs the analytical M is 2.7-3.8 "
    + "(relative Frobenius) against 0.6 for DeLaN.");

  // 12. Rollouts --------------------------------------------------------------------------------------
  s = content("Rollouts: DeLaN tracks, the MLP diverges", S3);
  s.addImage({ path: path.join(IMG, "rollout_example.png"), x: M, y: 1.45, w: 7.4, h: 2.84,
    objectName: "rollout_example", altText: "Example 2 s rollout against MuJoCo" });
  s.addChart(pres.charts.BAR, [
      { name: "Analytical (integration floor)", labels: ["0.5 s", "1 s", "2 s"], values: [2.13e-4, 3.28e-4, 3.66e-4] },
      { name: "DeLaN", labels: ["0.5 s", "1 s", "2 s"], values: [5.27e-3, 6.62e-3, 7.86e-3] },
      { name: "MLP", labels: ["0.5 s", "1 s", "2 s"], values: [2.17e-2, 2.89e-2, 5.13e-2] }],
    chartStyle({ x: M, y: 4.4, w: 7.4, h: 2.45, barDir: "col", chartColors: [MODEL.analytical, MODEL.delan,
      MODEL.mlp], valAxisLogScaleBase: 10, valAxisMinVal: 1e-4, valAxisMaxVal: 0.1,
      valAxisLabelFormatCode: "0.0E+0", showValue: false, legendPos: "r", legendFontSize: 12,
      showValAxisTitle: true, valAxisTitle: "rad (log)", objectName: "rollout_chart" }));
  stat(s, { x: 8.5, y: 1.5, w: 4.2, big: "0 % vs 31 %", label: "rollouts diverged within 2 s, in range "
    + "(DeLaN vs MLP)", color: MODEL.delan, bigSize: 36 });
  stat(s, { x: 8.5, y: 3.4, w: 4.2, big: "0 % vs 92 %", label: "the same, outside the training range",
    color: MODEL.delan, bigSize: 36 });
  text(s, "Implicit-in-velocity integration (the wrist is stiff): solve ID(q, v + h·a, a) = τ by Newton, "
    + "h = 10 ms. Chart: median position RMSE over 90 windows (in range); diverged = ∞.",
    { x: 8.5, y: 5.3, w: 4.2, h: 1.5, fontSize: 13, color: C.accent3 });
  s.addNotes("Ground truth: fresh contiguous MuJoCo episodes (the dataset is subsampled). The same integrator "
    + "with the exact model reproduces MuJoCo to 3e-4 rad, so the rest is model error. The MLP inversion "
    + "uses Newton with line search from 3 starts; it still has no well-behaved root where its implied "
    + "inertia is indefinite.");

  // 13. Sampled-data computed torque -----------------------------------------------------------------
  s = content("Computed torque on a sampled, damped arm", S3);
  card(s, M, 1.5, 5.9, 1.5);
  text(s, [{ text: "Classic:  ", options: { bold: true } }, { text: "τ = ID(q, q̇, a_cmd)" }],
    { x: M + 0.3, y: 1.6, w: 5.4, h: 0.55, fontSize: 20, fontFace: THEME.headFontFace, valign: "middle" });
  text(s, "Within a 10 ms hold the wrist velocity settles to (τ − g)/D, so feedback changes it by only "
    + "(M/D)·a_cmd ≈ 0.6 % of Δt·a_cmd", { x: M + 0.3, y: 2.2, w: 5.4, h: 0.75, fontSize: 13, color: C.accent3 });
  card(s, M, 3.2, 5.9, 1.5, "DDEFEE");
  text(s, [{ text: "Sampled:  ", options: { bold: true, color: MODEL.delan } },
           { text: "τ = ID(q, q̇ + Δt·a_cmd, a_cmd)" }],
    { x: M + 0.3, y: 3.3, w: 5.4, h: 0.55, fontSize: 20, fontFace: THEME.headFontFace, valign: "middle" });
  text(s, "Same model, evaluated at the velocity the joint should reach by the end of the hold "
    + "(backward Euler over one control step)", { x: M + 0.3, y: 3.9, w: 5.4, h: 0.75, fontSize: 13,
    color: C.accent3 });
  text(s, bullets([
    "a_cmd = q̈_ref + K_d ė + K_p e; ω_n = 15 rad/s, ζ = 1; control at 100 Hz",
    "Uses only inverse dynamics: all three models plug in unchanged",
  ], 15), { x: M, y: 4.95, w: 5.9, h: 1.8 });
  s.addChart(pres.charts.BAR, [
      { name: "Classic CT", labels: ["root", "shoulder", "elbow", "wrist"], values: [0.0669, 0.204, 0.255, 0.401] },
      { name: "Sampled CT", labels: ["root", "shoulder", "elbow", "wrist"], values: [0.0013, 0.00105, 0.00074, 0.00056] }],
    chartStyle({ x: 6.85, y: 1.45, w: 5.88, h: 5.35, barDir: "col", chartColors: [HEX.accent5, MODEL.analytical],
      valAxisLogScaleBase: 10, valAxisMinVal: 1e-4, valAxisMaxVal: 1, valAxisLabelFormatCode: "0.0E+0",
      showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: "0.000",
      showValAxisTitle: true, valAxisTitle: "Tracking RMSE (rad, log)", showTitle: true,
      title: "Exact model: error grows as M/D shrinks", objectName: "ct_chart" }));
  s.addNotes("Even the exact model tracks at 0.26 rad with the classic law, worse than PD. The error grows "
    + "from root to wrist as M/D shrinks, which confirms the mechanism. DeLaN under the classic law does "
    + "better (0.017 rad) only because it over-estimates the tiny wrist inertia. Relation to Assignment 3.");

  // 14. Tracking results ------------------------------------------------------------------------------
  s = content("Tracking: only DeLaN holds up out of range", S3);
  s.addChart(pres.charts.BAR, [
      { name: "Analytical", labels: ["10 trajectories inside the training range", "4 trajectories outside it"], values: [1.00e-3, 5.91e-3] },
      { name: "DeLaN", labels: ["10 trajectories inside the training range", "4 trajectories outside it"], values: [3.72e-3, 1.02e-2] },
      { name: "MLP", labels: ["10 trajectories inside the training range", "4 trajectories outside it"], values: [5.33e-3, 2.02e-1] },
      { name: "PD only", labels: ["10 trajectories inside the training range", "4 trajectories outside it"], values: [5.11e-2, 1.26e-1] }],
    chartStyle({ x: M, y: 1.45, w: 7.9, h: 5.35, barDir: "col",
      chartColors: [MODEL.analytical, MODEL.delan, MODEL.mlp, MODEL.pd], valAxisLogScaleBase: 10,
      valAxisMinVal: 1e-4, valAxisMaxVal: 1, valAxisLabelFormatCode: "0.0E+0", showValue: true,
      dataLabelPosition: "outEnd", dataLabelFormatCode: "0.0E+0", barGapWidthPct: 50,
      showValAxisTitle: true, valAxisTitle: "Mean tracking RMSE (rad, log)", showTitle: true,
      title: "Sampled computed torque in MuJoCo, 5 s per trajectory", objectName: "tracking_chart" }));
  stat(s, { x: 9.0, y: 1.5, w: 3.73, big: "3.7 mrad", label: "DeLaN in range; MLP 5.3 mrad, "
    + "PD 51 mrad", color: MODEL.delan, bigSize: 40 });
  stat(s, { x: 9.0, y: 3.45, w: 3.73, big: "0.20 rad", label: "MLP outside the range: 38× worse, and "
    + "worse than PD", color: MODEL.mlp, bigSize: 40 });
  text(s, "No torque saturation for the sampled controllers. With a 0.1 rad initial error all three "
    + "models converge (≈ 0.013 rad).", { x: 9.0, y: 5.4, w: 3.73, h: 1.4, fontSize: 13, color: C.accent3 });
  s.addNotes("Computed torque needs only inverse dynamics, so the MLP's poor inversion does not hurt it in "
    + "range. Outside the range its torque error (0.45 N·m) shows up directly as tracking error. DeLaN "
    + "degrades only 2.7x, consistent with its 1.7x one-step degradation.");

  // 15. OOD traces ----------------------------------------------------------------------------------
  s = content("Out of range, the MLP loses the trajectory", S3);
  s.addImage({ path: path.join(IMG, "tracking_ood.png"), x: M, y: 1.45, w: 12.1, h: 4.64,
    objectName: "ood_traces", altText: "Tracking error over time on an OOD trajectory" });
  text(s, "Trajectory ood_fourier_1 (only 8 % of its samples lie inside the training range). DeLaN and the "
    + "analytical model stay within ±0.03 rad; the MLP's errors reach 0.2–0.4 rad, as large as PD alone.",
    { x: M, y: 6.2, w: 12.1, h: 0.7, fontSize: 15 });
  s.addNotes("Show this before the live demo; the OOD demo video shows the same trajectory side by side.");

  // 16. Data efficiency -------------------------------------------------------------------------------
  s = content("Data efficiency: DeLaN is flat from 10 %", S3);
  const fr = ["10 %", "30 %", "50 %", "100 %"];
  s.addChart(pres.charts.LINE, [
      { name: "MLP", labels: fr, values: [6.61e-2, 8.07e-3, 6.93e-3, 5.33e-3] },
      { name: "DeLaN", labels: fr, values: [3.53e-3, 3.63e-3, 3.65e-3, 3.72e-3] }],
    chartStyle({ x: M, y: 1.45, w: 5.95, h: 4.6, chartColors: [MODEL.mlp, MODEL.delan], lineSize: 3,
      lineDataSymbol: "circle", lineDataSymbolSize: 9, valAxisLogScaleBase: 10, valAxisMinVal: 1e-3,
      valAxisMaxVal: 0.1, valAxisLabelFormatCode: "0.000", showValAxisTitle: true,
      valAxisTitle: "rad (log)", showTitle: true, title: "Tracking RMSE, 10 trajectories in range",
      objectName: "eff_tracking_chart" }));
  s.addChart(pres.charts.LINE, [
      { name: "MLP", labels: fr, values: [67.8, 41.1, 43.3, 31.1] },
      { name: "DeLaN", labels: fr, values: [0, 0, 0, 0] }],
    chartStyle({ x: 6.78, y: 1.45, w: 5.95, h: 4.6, chartColors: [MODEL.mlp, MODEL.delan], lineSize: 3,
      lineDataSymbol: "circle", lineDataSymbolSize: 9, valAxisMinVal: 0, valAxisMaxVal: 100,
      valAxisMajorUnit: 25, showValAxisTitle: true, valAxisTitle: "%", showTitle: true,
      title: "Rollouts diverged within 2 s (in range)", objectName: "eff_div_chart" }));
  text(s, "At 10 % data the MLP tracks 19× worse than DeLaN. One-step torque shows the same pattern "
    + "(MLP 0.069 → 0.044 N·m; DeLaN 0.014–0.015 N·m at every fraction).",
    { x: M, y: 6.2, w: 12.1, h: 0.7, fontSize: 15 });
  s.addNotes("Caveat for the viva: the nested subsets sample rows at random, so 10 % still covers all 40 "
    + "training episodes. The curve measures sample density along the same trajectories, not new regions of "
    + "state space.");

  // 17. Demo --------------------------------------------------------------------------------------------
  pres.addSection({ title: S4 });
  s = content("Live demo", S4);
  s.addMedia({ type: "video", path: path.join(REPO, "evaluation_control/results/tracking_demo_ood.mp4"),
    x: M, y: 1.5, w: 7.9, h: 1.975, cover: "data:image/png;base64," +
      fs.readFileSync(path.join(IMG, "video_cover.png")).toString("base64"), objectName: "demo_video" });
  text(s, "Analytical · MLP · DeLaN tracking the same OOD trajectory in MuJoCo (5 s)",
    { x: M, y: 3.6, w: 7.9, h: 0.4, fontSize: 13, color: C.accent3 });
  const steps = [
    { n: "1", h: "Rollouts against MuJoCo", c: "python -m evaluation_control.rollout --quick" },
    { n: "2", h: "Closed-loop tracking + video", c: "python -m evaluation_control.tracking_experiment --quick --video" },
    { n: "3", h: "Tests (30 pass)", c: "pytest -q evaluation_control delan" },
  ];
  for (let i = 0; i < 3; i++) {
    const y = 4.3 + i * 0.85;
    s.addShape(pres.shapes.OVAL, { x: M, y: y + 0.08, w: 0.55, h: 0.55, fill: { color: C.accent1 },
      line: { type: "none" }, objectName: nm("step_bg") });
    text(s, steps[i].n, { x: M, y: y + 0.08, w: 0.55, h: 0.55, fontSize: 18, bold: true,
      color: C.background1, align: "center", valign: "middle" });
    text(s, steps[i].h, { x: M + 0.8, y, w: 3.4, h: 0.7, fontSize: 16, bold: true, valign: "middle" });
    text(s, steps[i].c, { x: M + 4.2, y, w: 7.9, h: 0.7, fontSize: 14, fontFace: "Courier New",
      color: C.accent3, valign: "middle" });
  }
  card(s, 8.75, 1.5, 3.98, 2.55);
  text(s, "Full pipeline", { x: 9.0, y: 1.65, w: 3.55, h: 0.45, fontSize: 17, bold: true });
  text(s, "bash evaluation_control/run_all.sh\n    --video", { x: 9.0, y: 2.15, w: 3.6, h: 0.75, fontSize: 12,
    fontFace: "Courier New", color: C.accent3 });
  text(s, "≈ 15–20 min on a laptop CPU; writes every table and figure", { x: 9.0, y: 2.95, w: 3.55,
    h: 0.9, fontSize: 13 });
  s.addNotes("Run from the repository root with .venv/bin/python and MUJOCO_GL=egl. The quick rollout takes "
    + "about 1 minute and the quick tracking run about 1 minute. If time is short, play the embedded video.");

  // 18. Discussion -------------------------------------------------------------------------------------
  s = content("Discussion", S4);
  const disc = [
    { Icon: fa.FaCheckCircle, c: MODEL.delan, h: "When the prior helps", b: "Extrapolation (OOD), "
      + "small data, and any use that needs forward dynamics: SPD M makes the inversion well posed" },
    { Icon: fa.FaExclamationTriangle, c: HEX.accent5, h: "Failure cases", b: "Weakly excited inertia "
      + "(33 % M error); a purely conservative Lagrangian fails without damping (R² 0.14)" },
    { Icon: fa.FaClock, c: MODEL.analytical, h: "Computational cost", b: "Per state on CPU: MLP 0.1 ms, "
      + "analytical 1.9 ms, DeLaN 4.4 ms. All fit a 10 ms control period" },
    { Icon: fa.FaBook, c: MODEL.mlp, h: "Limitations · A2 & A3", b: "Noise-free simulated data, one seed, "
      + "no contacts. Extends the A2 model to learning and the A3 controller to sampled control" },
  ];
  for (let i = 0; i < 4; i++) {
    const col = i % 2, row = Math.floor(i / 2);
    await iconCard(s, { x: M + col * 6.2, y: 1.5 + row * 2.7, w: 5.9, h: 2.45, Icon: disc[i].Icon,
                        color: disc[i].c, head: disc[i].h, body: disc[i].b, bodySize: 17 });
  }
  s.addNotes("Limitations to admit: one seed (42), no measurement noise, a 10 ms integrator with a 3e-4 rad "
    + "floor, and analytical inertia is exact only because the data share its model.");

  // 19. Conclusion ---------------------------------------------------------------------------------------
  s = pres.addSlide({ masterName: "CLOSING_DARK", sectionTitle: S4 });
  s.addText("Conclusions and next steps", { placeholder: "title" });
  const concl = [
    { big: "Structure", d: "DeLaN's SPD inertia, potential and damping give accurate torques, a usable "
      + "forward model and stable OOD behaviour" },
    { big: "Control", d: "Sampled computed torque with DeLaN tracks at 3.7 mrad in range and 10 mrad out of "
      + "range, close to the exact model" },
    { big: "Data", d: "DeLaN reaches its accuracy with 10 % of the data; the MLP needs ≥ 30 % and still "
      + "fails outside the range" },
  ];
  for (let i = 0; i < 3; i++) {
    const x = M + i * 4.1;
    text(s, concl[i].big, { x, y: 1.75, w: 3.8, h: 0.7, fontSize: 28, bold: true, color: C.accent6,
      fontFace: THEME.headFontFace });
    text(s, concl[i].d, { x, y: 2.55, w: 3.8, h: 1.7, fontSize: 16, color: C.background1 });
  }
  text(s, "Next steps", { x: M, y: 4.65, w: 6, h: 0.5, fontSize: 20, bold: true, color: C.background1 });
  text(s, bullets([
    "Episode-wise data subsets and multiple seeds",
    "Measurement noise; excitation that targets the elbow/wrist inertia",
    "Model-predictive control and the OpenMANIPULATOR-X model",
  ], 16, { color: C.background1 }), { x: M, y: 5.2, w: 12, h: 1.6 });
  s.addNotes("Close on the main message and invite questions. Each member should be ready to explain any "
    + "part: data, analytical model, MLP, DeLaN, rollouts and control.");

  await pres.writeFile({ fileName: OUT });
  await applyTheme(OUT, THEME);
  console.log("wrote", OUT);
}

build().catch((e) => { console.error(e); process.exit(1); });
