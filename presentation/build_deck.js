// Builds the Group 4 project deck.
//   NODE_PATH=<folder with node_modules> SKILL_DIR=<pptx skill dir> node build_deck.js <repo_root> <out.pptx>
// Needs pptxgenjs, react, react-dom, react-icons and sharp (0.33.x on Node 18).
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
const R = (p) => path.join(REPO, p);

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
    accent5: "C0392B",   // failure / out of range
    accent6: "7FC7C0",   // light teal for dark slides
    hlink: "1F8A8A", folHlink: "4A5868",
  },
};
const HEX = THEME.colors;
const MODEL = { analytical: HEX.accent3, delan: HEX.accent1, mlp: HEX.accent2, pd: HEX.accent4 };
const TINT = { delan: "DDEFEE", mlp: "FBEBD9", bad: "F6DEDB" };

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE";            // 13.333 x 7.5 in
pres.theme = { headFontFace: THEME.headFontFace, bodyFontFace: THEME.bodyFontFace };
pres.title = "Deep Lagrangian Network for Computed-Torque Tracking";
pres.subject = "Robotics (ECE670/CSE633) course project, Group 4";
pres.author = "Group 4";
const C = pres.SchemeColor;

const W = 13.333, M = 0.6, CW = W - 2 * M;
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
  title: "DIVIDER_DARK",
  background: { color: C.text2 },
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: M, y: 2.55, w: 7.8, h: 1.2,
        fontFace: THEME.headFontFace, fontSize: 44, bold: true, color: C.background1, valign: "middle",
        margin: 0 }, text: "" } },
    { placeholder: { options: { name: "body", type: "body", x: M, y: 3.85, w: 7.8, h: 1.0,
        fontSize: 20, color: C.accent6, valign: "top", margin: 0 }, text: "" } },
  ],
  slideNumber: { x: W - M - 0.6, y: 7.02, w: 0.6, h: 0.3, fontSize: 10, color: C.accent4, align: "right" },
});
const contentObjects = (titleW) => [
  { placeholder: { options: { name: "title", type: "title", x: M, y: 0.35, w: titleW, h: 0.85,
      fontFace: THEME.headFontFace, fontSize: 32, bold: true, color: C.text1, valign: "middle",
      margin: 0 }, text: "" } },
  { text: { text: FOOTER, options: { x: M, y: 7.02, w: 9, h: 0.3, fontSize: 10, color: C.accent3,
      margin: 0, valign: "middle" } } },
];
pres.defineSlideMaster({
  title: "CONTENT", background: { color: C.background1 }, margin: [0.5, 0.6, 0.6, 0.6],
  objects: contentObjects(CW),
  slideNumber: { x: W - M - 0.6, y: 7.02, w: 0.6, h: 0.3, fontSize: 10, color: C.accent3, align: "right" },
});
// methodology slides: narrower title, step tracker on the right
pres.defineSlideMaster({
  title: "STEP", background: { color: C.background1 }, margin: [0.5, 0.6, 0.6, 0.6],
  objects: contentObjects(7.5),
  slideNumber: { x: W - M - 0.6, y: 7.02, w: 0.6, h: 0.3, fontSize: 10, color: C.accent3, align: "right" },
});
pres.defineSlideMaster({
  title: "CLOSING_DARK",
  background: { color: C.text2 },
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: M, y: 0.45, w: CW, h: 0.9,
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
async function iconCircle(slide, Icon, x, y, d, fill, iconColor = "FFFFFF") {
  slide.addShape(pres.shapes.OVAL, { x, y, w: d, h: d, fill: { color: fill }, line: { type: "none" },
    objectName: nm("icon_bg") });
  const pad = d * 0.24;
  slide.addImage({ data: await iconPng(Icon, iconColor), x: x + pad, y: y + pad, w: d - 2 * pad,
    h: d - 2 * pad, objectName: nm("icon"), altText: "" });
}
// a card with an icon circle, a bold header and body text
async function iconCard(slide, { x, y, w, h, Icon, color, head, body, bodySize = 15, fill }) {
  card(slide, x, y, w, h, fill);
  await iconCircle(slide, Icon, x + 0.25, y + 0.25, 0.6, color);
  text(slide, head, { x: x + 1.0, y: y + 0.25, w: w - 1.2, h: 0.6, fontSize: 18, bold: true,
                      valign: "middle" });
  text(slide, body, { x: x + 0.25, y: y + 1.0, w: w - 0.5, h: h - 1.15, fontSize: bodySize });
}
function stat(slide, { x, y, w, big, label, color, bigSize = 40, labelSize = 14, labelColor, labelH = 0.85 }) {
  text(slide, big, { x, y, w, h: 0.8, fontSize: bigSize, bold: true, color, fontFace: THEME.headFontFace,
                     valign: "bottom" });
  text(slide, label, { x, y: y + 0.9, w, h: labelH, fontSize: labelSize, color: labelColor || C.text1 });
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
function img(slide, file, x, y, w, h, alt) {
  slide.addImage({ path: file, x, y, w, h, objectName: nm("image"), altText: alt });
}
function caption(slide, t, x, y, w, h = 0.4) {
  text(slide, t, { x, y, w, h, fontSize: 12, color: C.accent3, italic: true });
}
function arrow(slide, x1, y1, x2, y2, color = HEX.accent4) {
  slide.addShape(pres.shapes.LINE, { x: Math.min(x1, x2), y: Math.min(y1, y2), w: Math.max(Math.abs(x2 - x1), 0.001),
    h: Math.max(Math.abs(y2 - y1), 0.001), flipV: (y2 < y1) !== (x2 < x1), flipH: x2 < x1,
    line: { color, width: 2.25, endArrowType: "triangle" }, objectName: nm("arrow") });
}
// rounded box with centred text, for diagrams
function box(slide, x, y, w, h, t, { fill = C.background2, color = C.text1, size = 15, bold = false } = {}) {
  card(slide, x, y, w, h, fill);
  text(slide, t, { x: x + 0.12, y, w: w - 0.24, h, fontSize: size, bold, color, align: "center",
                   valign: "middle" });
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
    dataLabelFontFace: "+mn-lt", dataLabelFontSize: 12, dataLabelColor: HEX.dk1,
    titleFontFace: "+mn-lt", titleFontSize: 15, titleColor: HEX.dk1,
    valAxisTitleFontFace: "+mn-lt", valAxisTitleFontSize: 12, valAxisTitleColor: HEX.accent3,
    ...extra,
  };
}

const STEPS = [
  { k: "A", w: "Data", c: MODEL.analytical }, { k: "B", w: "Physics", c: MODEL.analytical },
  { k: "C", w: "MLP", c: MODEL.mlp }, { k: "D", w: "DeLaN", c: MODEL.delan },
  { k: "E", w: "Control", c: MODEL.analytical },
];
// step tracker: five pills, the current one filled
function stepper(slide, cur, { x0 = 8.3, y = 0.45, pw = 0.84, ph = 0.66, gap = 0.05, dark = false } = {}) {
  STEPS.forEach((st, i) => {
    const x = x0 + i * (pw + gap), on = i === cur;
    slide.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w: pw, h: ph, rectRadius: 0.1,
      fill: { color: on ? C.text2 : (dark ? "2A3642" : C.background2) }, line: { type: "none" },
      objectName: nm("step_pill") });
    const fg = on ? C.background1 : (dark ? C.accent4 : C.accent3);
    text(slide, st.k, { x, y: y + 0.05, w: pw, h: 0.32, fontSize: 15, bold: true, align: "center",
      valign: "middle", color: on ? C.accent6 : fg, fontFace: THEME.headFontFace });
    text(slide, st.w, { x, y: y + 0.35, w: pw, h: 0.26, fontSize: 10, align: "center", valign: "middle",
      color: fg });
  });
}
function content(title, section) {
  const s = pres.addSlide({ masterName: "CONTENT", sectionTitle: section });
  s.addText(title, { placeholder: "title" });
  return s;
}
function step(title, section, cur) {
  const s = pres.addSlide({ masterName: "STEP", sectionTitle: section });
  s.addText(title, { placeholder: "title" });
  stepper(s, cur);
  return s;
}
async function divider(num, title, sub, section, Icon) {
  const s = pres.addSlide({ masterName: "DIVIDER_DARK", sectionTitle: section });
  await iconCircle(s, Icon, 8.9, 1.6, 3.6, "243140", HEX.accent6);
  text(s, num, { x: M, y: 1.35, w: 3, h: 1.1, fontSize: 60, bold: true, color: C.accent6,
                 fontFace: THEME.headFontFace, valign: "bottom" });
  s.addText(title, { placeholder: "title" });
  s.addText(sub, { placeholder: "body" });
  return s;
}

const deg = (rad) => rad * 180 / Math.PI;

// ------------------------------------------------------------------ slides
async function build() {
  const S0 = "Introduction", S1 = "The project", S2 = "Methodology", S3 = "Results", S4 = "Final thoughts";

  // 1. Title ---------------------------------------------------------------------------------
  pres.addSection({ title: S0 });
  let s = pres.addSlide({ masterName: "TITLE_DARK", sectionTitle: S0 });
  s.addImage({ path: path.join(IMG, "hero.png"), x: 7.4, y: 0, w: 5.933, h: 7.5,
    sizing: { type: "cover", w: 5.933, h: 7.5 }, objectName: "hero_render",
    altText: "MuJoCo render of the dm_control manipulator arm" });
  text(s, "ROBOTICS (ECE670/CSE633)  ·  COURSE PROJECT  ·  GROUP 4", { x: M, y: 1.0, w: 6.6, h: 0.4,
    fontSize: 13, bold: true, color: C.accent6, charSpacing: 1 });
  s.addText("Teaching a robot arm its own physics", { placeholder: "title" });
  s.addText("A Deep Lagrangian Network for computed-torque tracking of the dm_control manipulator arm, "
    + "compared with a plain neural network and the textbook physics model", { placeholder: "body" });
  s.addNotes("Introduce the project in one sentence: we teach a neural network how a robot arm moves, but "
    + "we build the laws of mechanics into the network, and we test whether that helps when the model is used "
    + "to simulate and to control the arm.");

  // 2. Agenda ---------------------------------------------------------------------------------
  s = content("How this talk is organised", S0);
  const parts = [
    { n: "1", h: "The project", d: "The problem, the three models we compare, how we judge them, and the "
      + "overall pipeline", Icon: fa.FaRobot },
    { n: "2", h: "Methodology", d: "The five steps, A to E, each with what we did and what we found",
      Icon: fa.FaCogs },
    { n: "3", h: "Results side by side", d: "All three models compared on every measure, plus a live demo",
      Icon: fa.FaChartBar },
    { n: "4", h: "Final thoughts", d: "What we learned, the limits of our study, and conclusions",
      Icon: fa.FaLightbulb },
  ];
  for (let i = 0; i < 4; i++) {
    const x = M + i * 3.1, y = 1.6, w = 2.85, h = 4.6;
    card(s, x, y, w, h);
    text(s, parts[i].n, { x: x + 0.3, y: y + 0.3, w: 1.2, h: 1.0, fontSize: 54, bold: true,
      color: C.accent1, fontFace: THEME.headFontFace, valign: "bottom" });
    await iconCircle(s, parts[i].Icon, x + w - 0.95, y + 0.45, 0.65, C.accent3);
    text(s, parts[i].h, { x: x + 0.3, y: y + 1.55, w: w - 0.6, h: 0.9, fontSize: 21, bold: true,
      fontFace: THEME.headFontFace });
    text(s, parts[i].d, { x: x + 0.3, y: y + 2.5, w: w - 0.6, h: 1.9, fontSize: 15, color: C.accent3 });
  }
  s.addNotes("Give the audience the map: first what and why, then the method step by step (each step has its "
    + "own results), then everything side by side, then conclusions.");

  // 3. What is the project about ---------------------------------------------------------------
  pres.addSection({ title: S1 });
  s = content("What is this project about?", S1);
  img(s, path.join(IMG, "arm.png"), M, 1.5, 4.4, 3.3, "Render of the 4-joint robot arm");
  caption(s, "The robot: a 4-joint arm from DeepMind's dm_control suite, simulated in MuJoCo",
    M, 4.9, 4.4, 0.6);
  text(s, "To move an arm along a planned path, its controller must know what motor torque each joint "
    + "needs at every instant. The rule that turns a wanted motion into torques is the arm's dynamics model.",
    { x: 5.4, y: 1.5, w: 7.33, h: 1.3, fontSize: 17 });
  // flow: motion -> model -> torque
  const fy = 3.05, fh = 1.35;
  box(s, 5.4, fy, 2.15, fh, "Wanted motion\nposition, speed,\nacceleration", { size: 14 });
  box(s, 8.0, fy, 2.15, fh, "Dynamics\nmodel", { fill: C.text2, color: C.background1, size: 17, bold: true });
  box(s, 10.6, fy, 2.13, fh, "Motor torques\n(twisting force\nat each joint)", { size: 14 });
  arrow(s, 7.6, fy + fh / 2, 7.95, fy + fh / 2, HEX.accent3);
  arrow(s, 10.2, fy + fh / 2, 10.55, fy + fh / 2, HEX.accent3);
  card(s, 5.4, 4.75, 7.33, 2.05, TINT.delan);
  text(s, [{ text: "Our goal:  ", options: { bold: true, color: C.accent1 } },
           { text: "learn this model from recorded data with a neural network that has the laws of "
             + "mechanics built in (a Deep Lagrangian Network, DeLaN). Then check whether it is accurate, "
             + "needs little data, copes with unfamiliar motions, and works inside a real controller." }],
    { x: 5.7, y: 4.95, w: 6.8, h: 1.7, fontSize: 16, valign: "middle" });
  s.addNotes("Plain-language framing. Inverse dynamics: given position q, speed q-dot and acceleration "
    + "q-double-dot, return the torque tau. A controller that uses such a model is called computed-torque "
    + "control. The rest of the talk explains how we build three versions of this model and test them.");

  // 4. Three ways ---------------------------------------------------------------------------------
  s = content("Three ways to get the dynamics model", S1);
  const three = [
    { Icon: fa.FaSquareRootAlt, c: MODEL.analytical, h: "Textbook physics", tag: "Analytical model",
      d: "Write down Newton–Euler/Lagrange equations using the exact masses and sizes of every link.",
      pro: "Exact, if every parameter is known", con: "Real robots rarely give us exact parameters" },
    { Icon: fa.FaProjectDiagram, c: MODEL.mlp, h: "Pure learning", tag: "Black-box MLP",
      d: "A standard neural network learns the mapping from motion to torque purely from examples.",
      pro: "Needs no physics knowledge", con: "Knows nothing outside its training data" },
    { Icon: fa.FaAtom, c: MODEL.delan, h: "Learning + physics", tag: "DeLaN (our focus)",
      d: "Neural networks learn the arm's energy; the equations of motion are then derived from it.",
      pro: "Learns from data, obeys physics", con: "More complex, slower to compute" },
  ];
  for (let i = 0; i < 3; i++) {
    const x = M + i * 4.1, y = 1.5, w = 3.85, h = 5.3, t = three[i];
    card(s, x, y, w, h, i === 2 ? TINT.delan : C.background2);
    await iconCircle(s, t.Icon, x + 0.3, y + 0.3, 0.75, t.c);
    text(s, t.h, { x: x + 1.25, y: y + 0.3, w: w - 1.45, h: 0.42, fontSize: 19, bold: true,
      fontFace: THEME.headFontFace });
    text(s, t.tag, { x: x + 1.25, y: y + 0.72, w: w - 1.45, h: 0.35, fontSize: 14, bold: true, color: t.c });
    text(s, t.d, { x: x + 0.3, y: y + 1.35, w: w - 0.6, h: 1.55, fontSize: 16 });
    await iconCircle(s, fa.FaPlus, x + 0.3, y + 3.15, 0.42, C.accent1);
    text(s, t.pro, { x: x + 0.9, y: y + 3.05, w: w - 1.2, h: 0.65, fontSize: 15, valign: "middle" });
    await iconCircle(s, fa.FaMinus, x + 0.3, y + 4.1, 0.42, C.accent5);
    text(s, t.con, { x: x + 0.9, y: y + 4.0, w: w - 1.2, h: 0.65, fontSize: 15, valign: "middle" });
  }
  s.addNotes("The analytical model is our reference: here it is exact because the data come from the same "
    + "simulated arm. The real question is MLP versus DeLaN: does building the physics into a network pay "
    + "off?");

  // 5. Physics built in ---------------------------------------------------------------------------
  s = content("The physics that DeLaN builds in", S1);
  card(s, M, 1.5, 7.4, 1.25, C.text2);
  text(s, "τ  =  M(q) q̈  +  C(q, q̇) q̇  +  g(q)  +  D q̇", { x: M + 0.3, y: 1.5, w: 6.8, h: 1.25,
    fontSize: 26, bold: true, color: C.background1, fontFace: THEME.headFontFace, valign: "middle",
    align: "center" });
  const terms = [
    { h: "Inertia  M(q) q̈", d: "torque needed to speed a joint up; depends on the arm's pose" },
    { h: "Coriolis  C(q, q̇) q̇", d: "forces that moving joints exert on one another" },
    { h: "Gravity  g(q)", d: "torque needed just to hold the arm up" },
    { h: "Damping  D q̇", d: "joint friction that grows with speed" },
  ];
  for (let i = 0; i < 4; i++) {
    const col = i % 2, row = Math.floor(i / 2), x = M + col * 3.75, y = 2.95 + row * 1.2;
    card(s, x, y, 3.65, 1.05);
    text(s, terms[i].h, { x: x + 0.2, y: y + 0.1, w: 3.3, h: 0.38, fontSize: 15, bold: true });
    text(s, terms[i].d, { x: x + 0.2, y: y + 0.48, w: 3.3, h: 0.5, fontSize: 13, color: C.accent3 });
  }
  text(s, [{ text: "Built-in rule: ", options: { bold: true, color: C.accent1 } },
           { text: "the inertia matrix M must be positive-definite: moving costs energy, and a push always "
             + "gives a sensible acceleration. DeLaN guarantees this; a plain network does not." }],
    { x: M, y: 5.45, w: 7.4, h: 1.3, fontSize: 15 });
  s.addChart(pres.charts.BAR, [{ name: "Torque size", labels: ["Damping", "Inertia", "Gravity", "Coriolis"],
      values: [2.37, 0.61, 0.37, 0.012] }],
    chartStyle({ x: 8.3, y: 1.45, w: 4.43, h: 5.35, barDir: "col",
      chartColors: [MODEL.delan, MODEL.analytical, MODEL.analytical, MODEL.analytical],
      showLegend: false, showValue: true, dataLabelPosition: "outEnd",
      dataLabelFormatCode: "0.00", valAxisMinVal: 0, valAxisMaxVal: 2.8, valAxisMajorUnit: 0.5,
      showValAxisTitle: true, valAxisTitle: "Typical torque (N·m)", showTitle: true,
      title: "Which term matters most? (root joint)", objectName: "terms_chart" }));
  s.addNotes("Read the equation term by term. Key fact about this arm: joint friction (damping) dominates "
    + "the torque, and Coriolis forces are tiny. Values are RMS contributions in the training data, root joint, "
    + "from Member 4's analysis. This will explain several results later.");

  // 6. Measures of success -------------------------------------------------------------------------
  s = content("How we judge each model", S1);
  const meas = [
    { Icon: fa.FaCrosshairs, h: "Torque accuracy", d: "Error of the predicted torque on unseen test data "
      + "(one step)" },
    { Icon: fa.FaFilm, h: "Simulation", d: "Run the model forward for 0.5, 1 and 2 s and compare the motion "
      + "with the true simulator" },
    { Icon: fa.FaBullseye, h: "Tracking", d: "Use the model inside a controller to follow 10 planned "
      + "paths; measure how far the arm strays" },
    { Icon: fa.FaDatabase, h: "Data efficiency", d: "Train on 10, 30, 50 and 100 % of the data and see how "
      + "much the model needs" },
    { Icon: fa.FaExpandArrowsAlt, h: "Out-of-range motions", d: "Test on faster, larger motions than any "
      + "in the training data (called OOD)" },
    { Icon: fa.FaBalanceScale, h: "Physical sense", d: "Is the learned inertia matrix positive-definite, "
      + "as real physics requires?" },
  ];
  for (let i = 0; i < 6; i++) {
    const col = i % 3, row = Math.floor(i / 3);
    await iconCard(s, { x: M + col * 4.1, y: 1.5 + row * 2.7, w: 3.85, h: 2.45, Icon: meas[i].Icon,
      color: i === 4 ? HEX.accent5 : C.accent1, head: meas[i].h, body: meas[i].d });
  }
  s.addNotes("These six measures come from our approved project statement. Every model is scored on the same "
    + "data, with the same code.");

  // 7. Pipeline ----------------------------------------------------------------------------------------
  s = content("Overall pipeline: five steps, five team members", S1);
  const ws = [
    { t: "A · Data", d: "Simulate the arm and record the data", Icon: fa.FaDatabase,
      c: MODEL.analytical, who: "Member 1" },
    { t: "B · Physics model", d: "Exact equations from the simulator's parameters",
      Icon: fa.FaSquareRootAlt, c: MODEL.analytical, who: "Member 2" },
    { t: "C · Black-box MLP", d: "Plain neural network baseline", Icon: fa.FaProjectDiagram,
      c: MODEL.mlp, who: "Member 3" },
    { t: "D · DeLaN", d: "Neural network with physics built in", Icon: fa.FaAtom, c: MODEL.delan,
      who: "Member 4" },
    { t: "E · Simulation & control", d: "Test all three models in the same simulations and controllers",
      Icon: fa.FaCogs, c: MODEL.analytical, who: "Member 5" },
  ];
  const bx = [M, 4.0, 4.0, 4.0, 8.75], by = [2.85, 1.45, 2.85, 4.25, 2.85];
  const bw = [2.85, 4.1, 4.1, 4.1, 3.98], bh = 1.2;
  for (let i = 0; i < 5; i++) {
    const x = bx[i], y = by[i], w = bw[i];
    card(s, x, y, w, bh, i === 3 ? TINT.delan : C.background2);
    await iconCircle(s, ws[i].Icon, x + 0.2, y + 0.29, 0.62, ws[i].c);
    text(s, ws[i].t, { x: x + 0.95, y: y + 0.12, w: w - 1.1, h: 0.4, fontSize: 16, bold: true });
    text(s, ws[i].d, { x: x + 0.95, y: y + 0.52, w: w - 1.1, h: 0.6, fontSize: 13, color: C.accent3 });
  }
  const midY = (i) => by[i] + bh / 2;
  for (const i of [1, 2, 3]) arrow(s, bx[0] + bw[0] + 0.08, midY(0), bx[i] - 0.08, midY(i));
  for (const i of [1, 2, 3]) arrow(s, bx[i] + bw[i] + 0.08, midY(i), bx[4] - 0.08, midY(4));
  // ownership strip and the shared rules
  card(s, M, 5.7, CW, 1.1, C.text2);
  text(s, [{ text: "Fair comparison:  ", options: { bold: true, color: C.accent6 } },
           { text: "every model uses the same data splits, the same training subsets, the same random seed "
             + "(42) and the same scoring code.", options: { color: C.background1 } }],
    { x: M + 0.3, y: 5.7, w: CW - 0.6, h: 1.1, fontSize: 16, valign: "middle" });
  text(s, "Members 1–5 own steps A–E", { x: 8.75, y: 4.35, w: 3.98, h: 0.4, fontSize: 13, color: C.accent3 });
  s.addNotes("Step A's dataset feeds the three models (B, C, D). Step E wraps all three behind one common "
    + "interface and runs the same simulation and control experiments on each. Member 1: data; Member 2: "
    + "analytical model; Member 3: MLP; Member 4: DeLaN; Member 5: evaluation and control.");

  // ================================================================= PART 2: METHODOLOGY
  pres.addSection({ title: S2 });
  s = await divider("02", "Methodology, step by step", "Five steps, A to E. Each one shows what we did and "
    + "what we found.", S2, fa.FaCogs);
  stepper(s, -1, { x0: M, y: 5.2, pw: 1.45, ph: 0.9, gap: 0.12, dark: true });
  s.addNotes("Now walk through the five steps. The tracker in the top-right corner of each slide shows which "
    + "step we are on.");

  // A1. Robot and simulator -------------------------------------------------------------------------
  s = step("A · The robot and the simulator", S2, 0);
  const rows = [["Joint", "Range of motion", "Max torque", "Friction"],
    ["Root (base)", "full turn", "12 N·m", "2.0"], ["Shoulder", "±160°", "8 N·m", "1.5"],
    ["Elbow", "±160°", "4 N·m", "1.0"], ["Wrist", "±140°", "2 N·m", "0.5"]];
  s.addTable(rows.map((r, i) => r.map((v) => ({ text: v, options: {
      bold: i === 0, color: i === 0 ? C.background1 : C.text1, fill: { color: i === 0 ? C.accent3 : (i % 2
        ? C.background1 : C.background2) }, fontSize: 15, valign: "middle", margin: [0, 0.1, 0, 0.1] } }))),
    { x: M, y: 1.5, w: 7.3, colW: [1.9, 2.0, 1.7, 1.7], rowH: 0.48, border: { type: "none" },
      objectName: "joint_table" });
  caption(s, "Friction (damping) in N·m per rad/s, read from the simulator model", M, 3.95, 7.3);
  text(s, bullets([
    [{ text: "Simplified for clean data: ", options: { bold: true } },
     { text: "the fingers are locked to the wrist and collisions are switched off, so only the 4 arm joints "
       + "decide the torque" }],
    [{ text: "Timing: ", options: { bold: true } },
     { text: "physics computed every 1 ms; controller acts every 10 ms" }],
    [{ text: "Key property: ", options: { bold: true } },
     { text: "the wrist is very light but strongly damped, so it reacts within 0.06 ms. This shapes how we "
       + "simulate and control it (step E)" }],
  ], 15), { x: M, y: 4.45, w: 7.3, h: 2.4 });
  img(s, path.join(IMG, "arm.png"), 8.3, 1.5, 4.43, 3.32, "Render of the 4-joint arm");
  caption(s, "The arm in MuJoCo (the 4 joints all turn in one vertical plane)", 8.3, 4.9, 4.43, 0.6);
  s.addNotes("Plant: dm_control manipulator/bring_ball. Data model: the gripper hinges are removed (finger "
    + "bodies rigidly attached), the gripper tendon is removed, collisions are disabled, and the contact count is "
    + "checked to be zero. Wrist mass/damping ratio is about 6e-5 s, far below the 10 ms control step.");

  // A2. Collecting data -----------------------------------------------------------------------------
  s = step("A · Collecting the data", S2, 0);
  text(s, "We drive the motors with three kinds of test signals and record position, speed, acceleration "
    + "and torque every 10 ms.", { x: M, y: 1.5, w: 6.0, h: 0.95, fontSize: 16 });
  const ex = [
    { Icon: fa.FaRandom, h: "Random", d: "a new random torque every 10 ms" },
    { Icon: fa.FaWaveSquare, h: "Smooth waves", d: "sums of sine waves, 0.1–1 Hz" },
    { Icon: fa.FaSignal, h: "Sweeps (chirps)", d: "waves that speed up over time" },
  ];
  for (let i = 0; i < 3; i++) {
    const y = 2.6 + i * 1.0;
    card(s, M, y, 6.0, 0.85);
    await iconCircle(s, ex[i].Icon, M + 0.2, y + 0.15, 0.55, C.accent3);
    text(s, ex[i].h, { x: M + 0.95, y, w: 2.0, h: 0.85, fontSize: 16, bold: true, valign: "middle" });
    text(s, ex[i].d, { x: M + 2.95, y, w: 2.9, h: 0.85, fontSize: 14, valign: "middle" });
  }
  text(s, bullets([
    "Out-of-range (OOD) set: stronger and faster signals, so every sample lies beyond the training data",
    "Clean simulated data: no sensor noise, no contacts",
  ], 15), { x: M, y: 5.65, w: 6.0, h: 1.2 });
  s.addChart(pres.charts.BAR, [{ name: "Samples", labels: ["Training", "Validation", "Test", "OOD test"],
      values: [20000, 2000, 2000, 2000] }],
    chartStyle({ x: 7.0, y: 1.45, w: 5.73, h: 5.35, barDir: "col", chartColors: [HEX.accent3],
      showLegend: false, showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: "#,##0",
      showTitle: true, title: "Samples in each split (40 + 10 + 10 + 10 episodes of 10 s)",
      valAxisLabelFormatCode: "#,##0", valAxisMaxVal: 24000, objectName: "split_chart" }));
  s.addNotes("Validation is used only to tune and stop training; test and OOD are touched only at the end. "
    + "Training amplitude 0.35, OOD amplitude 0.65 with higher frequencies. Normalisation statistics come from "
    + "the training split only.");

  // A3. OOD -----------------------------------------------------------------------------------------
  s = step("A · What “out of range” means", S2, 0);
  img(s, path.join(IMG, "ood_scatter.png"), M, 1.45, 8.6, 3.58, "Training data and OOD data, speed against "
    + "acceleration, for the shoulder and wrist joints");
  caption(s, "Each dot is one recorded sample. Grey: what the models learn from. Red: the OOD test set.",
    M, 5.1, 8.6);
  card(s, 9.5, 1.5, 3.23, 5.3, C.background2);
  await iconCircle(s, fa.FaExpandArrowsAlt, 9.75, 1.75, 0.65, C.accent5);
  text(s, "Why test this?", { x: 9.75, y: 2.6, w: 2.8, h: 0.45, fontSize: 18, bold: true });
  text(s, "A robot in use will sooner or later move faster or further than in any recorded data. A good "
    + "model should still give sensible torques there.", { x: 9.75, y: 3.1, w: 2.8, h: 2.4, fontSize: 15 });
  text(s, "The red points sit at speeds the models never saw.", { x: 9.75, y: 5.6, w: 2.8, h: 1.0,
    fontSize: 15, bold: true, color: C.accent5 });
  text(s, "Out-of-distribution (OOD) = every sample has at least one joint position, speed or acceleration "
    + "outside the training minimum/maximum.", { x: M, y: 5.6, w: 8.6, h: 1.0, fontSize: 15 });
  s.addNotes("Figure generated from dataset/data/train.npz and ood.npz. OOD is a hard test for a black box, "
    + "which can only interpolate, and a natural test for a physics prior.");

  // B. Analytical ---------------------------------------------------------------------------------------
  s = step("B · The textbook physics model", S2, 1);
  text(s, "What we did", { x: M, y: 1.5, w: 7.3, h: 0.45, fontSize: 20, bold: true, color: C.accent3,
    fontFace: THEME.headFontFace });
  text(s, bullets([
    "Read the mass, centre of mass and inertia of every link from the simulator's robot description",
    "Build each term of the equation (inertia, Coriolis, gravity, friction) by hand, using Lagrange's method",
    "Check it against the simulator's own built-in torque calculation on 1,000 random states",
  ], 16), { x: M, y: 2.05, w: 7.3, h: 2.6 });
  card(s, M, 4.85, 7.3, 1.95, TINT.delan);
  text(s, [{ text: "Its role in the project:  ", options: { bold: true, color: C.accent1 } },
           { text: "since it is exact here, it is the reference. It shows the best any model could do, and "
             + "lets us check what DeLaN learned against the true physics." }],
    { x: M + 0.3, y: 4.85, w: 6.7, h: 1.95, fontSize: 16, valign: "middle" });
  text(s, "What we found", { x: 8.4, y: 1.5, w: 4.3, h: 0.45, fontSize: 20, bold: true, color: C.accent3,
    fontFace: THEME.headFontFace });
  stat(s, { x: 8.4, y: 2.0, w: 4.3, big: "Exact", label: "Largest error vs the simulator is below a "
    + "millionth of a billionth of a N·m: computer rounding only", color: MODEL.analytical, bigSize: 48, labelH: 1.1 });
  stat(s, { x: 8.4, y: 4.3, w: 4.3, big: "0 N·m", label: "error on the test and OOD sets (to 15 decimal "
    + "places)", color: MODEL.analytical, bigSize: 48 });
  s.addNotes("This is Member 2's model (analytical_euler_lagrange). Max error 8.9e-16 N·m against "
    + "mujoco.mj_inverse; RMSE 1.7e-16 on test and 3.1e-16 on OOD. It is exact because the data come from the "
    + "same rigid-body model. On a real robot the parameters would be uncertain.");

  // C. MLP ----------------------------------------------------------------------------------------------
  s = step("C · The black-box network (MLP)", S2, 2);
  // architecture diagram
  const layers = [{ n: "13", l: "inputs" }, { n: "256", l: "" }, { n: "256", l: "" }, { n: "128", l: "" },
                  { n: "4", l: "torques" }];
  const lh = [1.4, 2.5, 2.5, 2.0, 1.1];
  for (let i = 0; i < 5; i++) {
    const x = M + 0.2 + i * 1.35, cy = 2.95, h = lh[i];
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y: cy - h / 2, w: 0.75, h, rectRadius: 0.1,
      fill: { color: i === 0 || i === 4 ? C.accent3 : C.accent2 }, line: { type: "none" },
      objectName: nm("layer") });
    text(s, layers[i].n, { x, y: cy - 0.25, w: 0.75, h: 0.5, fontSize: 14, bold: true,
      color: C.background1, align: "center", valign: "middle" });
    if (i < 4) arrow(s, x + 0.82, cy, x + 1.28, cy, HEX.accent4);
  }
  text(s, "position, speed, acceleration", { x: M, y: 4.3, w: 2.4, h: 0.4, fontSize: 12, color: C.accent3 });
  text(s, "4 joint torques", { x: M + 4.9, y: 4.3, w: 2.0, h: 0.4, fontSize: 12, color: C.accent3,
    align: "right" });
  text(s, bullets([
    "Learns torque directly from examples; no physics inside",
    "Size and settings picked by a search on validation data only",
    "Trained on 10, 30, 50 and 100 % of the training data",
  ], 15), { x: M, y: 4.9, w: 6.6, h: 1.9 });
  img(s, R("black_box_mlp/results/data_efficiency_rmse.png"), 7.5, 1.45, 5.23, 3.43,
    "MLP torque error against the fraction of training data, test and OOD");
  stat(s, { x: 7.5, y: 5.0, w: 2.5, big: "0.044", label: "N·m error on normal test data", color: MODEL.mlp,
    bigSize: 34, labelSize: 13 });
  stat(s, { x: 10.2, y: 5.0, w: 2.53, big: "0.45", label: "N·m out of range: 10× worse", color: HEX.accent5,
    bigSize: 34, labelSize: 13 });
  s.addNotes("Member 3's model: 13 inputs (sin and cos of the root angle, other positions, 4 speeds, 4 "
    + "accelerations, all normalised), three tanh hidden layers 256-256-128, AdamW, early stopping. Figure from "
    + "black_box_mlp/results. Torque RMSE on the test set falls from 0.069 (10 %) to 0.044 N·m (100 %); OOD is "
    + "10 times worse and does not improve with more data.");

  // D1. DeLaN how ---------------------------------------------------------------------------------------
  s = step("D · How DeLaN works", S2, 3);
  text(s, "Instead of learning torque directly, DeLaN learns the arm's energy. The equation of motion "
    + "is then derived from it automatically.", { x: M, y: 1.45, w: CW, h: 0.75, fontSize: 17 });
  // diagram: q -> three branches -> equation -> tau
  box(s, M, 3.55, 1.6, 1.0, "Joint\npositions q", { size: 14 });
  const br = [
    { y: 2.35, h: "Network 1", d: "inertia matrix M(q)\nbuilt as L·Lᵀ: always\npositive-definite" },
    { y: 3.6, h: "Network 2", d: "potential energy V(q)\n→ gravity = its slope" },
    { y: 4.85, h: "4 learned numbers", d: "joint friction D\n(kept ≥ 0)" },
  ];
  for (const b of br) {
    card(s, 3.0, b.y, 3.6, 1.1, TINT.delan);
    text(s, b.h, { x: 3.2, y: b.y + 0.08, w: 3.2, h: 0.32, fontSize: 14, bold: true, color: C.accent1 });
    text(s, b.d, { x: 3.2, y: b.y + 0.38, w: 3.2, h: 0.7, fontSize: 12 });
    arrow(s, M + 1.68, 4.05, 2.92, b.y + 0.55, HEX.accent4);
    arrow(s, 6.68, b.y + 0.55, 7.42, 4.05, HEX.accent4);
  }
  box(s, 7.5, 3.3, 2.6, 1.5, "Lagrange's equations\n(automatic\ndifferentiation)",
    { fill: C.text2, color: C.background1, size: 14, bold: true });
  arrow(s, 10.18, 4.05, 10.6, 4.05, HEX.accent3);
  box(s, 10.68, 3.55, 2.05, 1.0, "Torque τ", { size: 16, bold: true });
  text(s, bullets([
    "Same data, subsets, seed and scoring as the MLP",
    "Friction is essential: a pure energy model cannot explain friction (see next slide)",
  ], 14), { x: 7.5, y: 5.2, w: 5.23, h: 1.6 });
  s.addNotes("Member 4's model. M = L L-transpose + epsilon I, with L lower-triangular and a softplus "
    + "(positive) diagonal: symmetric positive-definite by construction. Gravity g = dV/dq by autograd. "
    + "Coriolis is computed from dM/dq with torch.func; nothing is derived by hand. Inputs are sin and cos of "
    + "the angles; 128x3 tanh networks chosen by a 16-configuration validation sweep. Forward dynamics is a "
    + "linear solve with M.");

  // D2. DeLaN accuracy ----------------------------------------------------------------------------------
  s = step("D · How accurate is DeLaN?", S2, 3);
  img(s, R("delan/results/torque_scatter.png"), M, 1.45, 8.1, 3.78, "DeLaN predicted against true torque for "
    + "each joint, test and OOD");
  caption(s, "Predicted against true torque per joint. Top row: normal test data, bottom row: out-of-range "
    + "data. Points on the diagonal are perfect.", M, 5.3, 8.1, 0.6);
  stat(s, { x: 9.1, y: 1.5, w: 3.63, big: "0.015", label: "N·m error on normal test data (3× better than "
    + "the MLP)", color: MODEL.delan, bigSize: 40 });
  stat(s, { x: 9.1, y: 3.35, w: 3.63, big: "0.025", label: "N·m out of range: only 1.7× worse (MLP: 10×)",
    color: MODEL.delan, bigSize: 40 });
  card(s, 9.1, 5.2, 3.63, 1.6, TINT.bad);
  text(s, [{ text: "Without the friction term: ", options: { bold: true, color: C.accent5 } },
           { text: "error 1.49 N·m, about 100× worse. A pure energy model cannot fit this arm." }],
    { x: 9.3, y: 5.2, w: 3.3, h: 1.6, fontSize: 14, valign: "middle" });
  text(s, "Accuracy is almost the same with 10 % of the data as with 100 %.", { x: M, y: 6.0, w: 8.1,
    h: 0.5, fontSize: 15, bold: true, color: C.accent1 });
  s.addNotes("Torque RMSE at 100 % data: 0.0147 N·m test, 0.0250 OOD; R-squared above 0.9999. With 10 % "
    + "data: 0.0144 / 0.0261. Ablation without damping: test RMSE 1.49, R-squared 0.14. Figure from "
    + "delan/results.");

  // D3. Physics recovered --------------------------------------------------------------------------------
  s = step("D · Did DeLaN learn the real physics?", S2, 3);
  img(s, R("delan/results/physics_comparison.png"), M, 1.45, CW, 3.48, "Learned against true mass matrix, "
    + "gravity and damping");
  caption(s, "DeLaN's learned physics against the exact model. Blue: true values; orange: learned (right). "
    + "On the diagonal = correct (left, middle).", M, 4.95, CW);
  const ph = [
    { big: "0.1 %", label: "error in the learned friction: almost exact", c: MODEL.delan },
    { big: "7 %", label: "error in gravity: close", c: MODEL.delan },
    { big: "33 %", label: "error in inertia: the light wrist and elbow barely affect the torque, so the data "
      + "hardly reveal them", c: MODEL.mlp },
  ];
  for (let i = 0; i < 3; i++) {
    stat(s, { x: M + i * 4.1, y: 5.25, w: 3.85, big: ph[i].big, label: ph[i].label, color: ph[i].c,
              bigSize: 30, labelSize: 13, labelH: 0.75 });
  }
  s.addNotes("Learned damping [1.999, 1.500, 1.000, 0.500] against true [2, 1.5, 1, 0.5]. Gravity RMSE 0.014 "
    + "N·m. The inertia of the wrist and elbow changes the torque by only 0.01-0.03 N·m, the same size as the "
    + "fit error, so it is weakly identifiable. This is a limit of the data, not the model. Accurate torques do "
    + "not automatically mean accurate physics.");

  // E1. Two ways to use a model --------------------------------------------------------------------------
  s = step("E · Two ways to use a learned model", S2, 4);
  const panels = [
    { x: M, h: "Test 1 · Simulation (open loop)", Icon: fa.FaFilm, flow: ["Recorded\nmotor torques",
      "Model predicts\nthe motion", "Compare with\nthe true simulator"],
      d: "Needs the model “run backwards”: from torque to motion. Errors pile up over time, so this is a "
        + "harsh test.", need: "Model used backwards: torque → motion" },
    { x: 6.8, h: "Test 2 · Control (closed loop)", Icon: fa.FaBullseye, flow: ["Planned\npath",
      "Controller +\nmodel → torque", "Simulated arm\n(MuJoCo)"],
      d: "The arm's measured position is fed back every 10 ms, so the controller can correct mistakes.",
      need: "Model used forwards: motion → torque" },
  ];
  for (const p of panels) {
    card(s, p.x, 1.5, 5.93, 5.3);
    await iconCircle(s, p.Icon, p.x + 0.3, 1.75, 0.65, C.accent1);
    text(s, p.h, { x: p.x + 1.15, y: 1.75, w: 4.6, h: 0.65, fontSize: 19, bold: true, valign: "middle",
      fontFace: THEME.headFontFace });
    for (let k = 0; k < 3; k++) {
      const bx2 = p.x + 0.3 + k * 1.9;
      box(s, bx2, 2.8, 1.6, 1.25, p.flow[k], { fill: k === 1 ? C.text2 : C.background1,
        color: k === 1 ? C.background1 : C.text1, size: 13, bold: k === 1 });
      if (k < 2) arrow(s, bx2 + 1.63, 3.42, bx2 + 1.87, 3.42, HEX.accent3);
    }
    if (p.x > 1) {   // feedback arrow under the control loop
      s.addShape(pres.shapes.LINE, { x: p.x + 1.1, y: 4.35, w: 4.6, h: 0, line: { color: HEX.accent1,
        width: 2.25, beginArrowType: "triangle" }, objectName: nm("feedback") });
      s.addShape(pres.shapes.LINE, { x: p.x + 5.7, y: 4.05, w: 0, h: 0.3, line: { color: HEX.accent1,
        width: 2.25 }, objectName: nm("feedback") });
      s.addShape(pres.shapes.LINE, { x: p.x + 1.1, y: 4.05, w: 0, h: 0.3, line: { color: HEX.accent1,
        width: 2.25, beginArrowType: "triangle" }, objectName: nm("feedback") });
      text(s, "measured position fed back", { x: p.x + 1.9, y: 4.4, w: 3.0, h: 0.3, fontSize: 12,
        color: C.accent1, align: "center" });
    }
    text(s, p.d, { x: p.x + 0.3, y: 4.85, w: 5.33, h: 1.0, fontSize: 15 });
    card(s, p.x + 0.3, 5.95, 5.33, 0.6, p.x > 1 ? TINT.delan : TINT.bad);
    text(s, p.need, { x: p.x + 0.5, y: 5.95, w: 4.93, h: 0.6, fontSize: 14, bold: true, valign: "middle",
      color: p.x > 1 ? C.accent1 : C.accent5 });
  }
  s.addNotes("Member 5's step. All three models are wrapped behind one shared interface (inverse dynamics, "
    + "its derivatives, forward dynamics), so exactly the same simulation and control code runs on each. "
    + "Simulation needs forward dynamics (torque to acceleration); for the MLP this means numerically inverting "
    + "it. Computed-torque control only needs inverse dynamics.");

  // E2. Rollouts -------------------------------------------------------------------------------------------
  s = step("E · Test 1: predicting motion 2 s ahead", S2, 4);
  img(s, path.join(IMG, "rollout_example.png"), M, 1.45, 7.4, 2.84, "Example 2 s prediction against the "
    + "true simulator");
  caption(s, "One example: thick grey is the true motion. DeLaN (teal) stays on it; the MLP (amber) drifts.",
    M, 4.3, 7.4);
  s.addChart(pres.charts.BAR, [
      { name: "Exact model", labels: ["after 0.5 s", "after 1 s", "after 2 s"], values: [0.0122, 0.0188, 0.021] },
      { name: "DeLaN", labels: ["after 0.5 s", "after 1 s", "after 2 s"], values: [0.302, 0.379, 0.450] },
      { name: "MLP", labels: ["after 0.5 s", "after 1 s", "after 2 s"], values: [1.243, 1.656, 2.94] }],
    chartStyle({ x: M, y: 4.7, w: 7.4, h: 2.15, barDir: "col", chartColors: [MODEL.analytical, MODEL.delan,
      MODEL.mlp], showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: "0.00",
      dataLabelFontSize: 11, valAxisMinVal: 0, valAxisMaxVal: 3.5, valAxisMajorUnit: 1, legendPos: "r",
      legendFontSize: 12, showValAxisTitle: true, valAxisTitle: "error (°)", objectName: "rollout_chart" }));
  stat(s, { x: 8.5, y: 1.5, w: 4.23, big: "0 % vs 31 %", label: "of predictions that blow up within 2 s "
    + "(DeLaN vs MLP)", color: MODEL.delan, bigSize: 36 });
  stat(s, { x: 8.5, y: 3.35, w: 4.23, big: "0 % vs 92 %", label: "the same for out-of-range motions",
    color: MODEL.delan, bigSize: 36 });
  card(s, 8.5, 5.15, 4.23, 1.65);
  text(s, [{ text: "Why the MLP blows up: ", options: { bold: true } },
           { text: "running it backwards means solving for acceleration, and its implied inertia is not "
             + "positive-definite, so the answer can be wrong or missing." }],
    { x: 8.7, y: 5.15, w: 3.85, h: 1.65, fontSize: 13, valign: "middle" });
  s.addNotes("Ground truth: fresh 10 s MuJoCo episodes (the dataset itself is subsampled), 90 windows of "
    + "2 s per distribution. Chart: median joint-angle error in degrees, normal data; a blown-up prediction "
    + "counts as infinite. Because the wrist is stiff we integrate implicitly in velocity, at 10 ms steps; the "
    + "exact model through the same integrator stays within 0.02 degrees, so the integrator is not the "
    + "problem. The MLP inversion uses Newton's method with line search from 3 starting points; failures "
    + "remain because the model has no well-behaved solution there. None of DeLaN's 720 rollouts diverged.");

  // E3. Computed torque --------------------------------------------------------------------------------------
  s = step("E · Test 2: computed-torque control", S2, 4);
  text(s, "Every 10 ms, the controller asks the model: “what torque gives the acceleration I need to stay "
    + "on the path?” A small correction pulls the arm back when it strays.", { x: M, y: 1.45, w: 6.0, h: 1.3,
    fontSize: 16 });
  card(s, M, 2.95, 6.0, 1.55);
  text(s, "Standard version", { x: M + 0.3, y: 3.05, w: 5.4, h: 0.45, fontSize: 17, bold: true,
    valign: "middle" });
  text(s, "Uses the joint's current speed. Fails here: the damped wrist settles to a new speed within the "
    + "10 ms step, so most of the correction is lost.", { x: M + 0.3, y: 3.5, w: 5.4, h: 0.9, fontSize: 14, color: C.accent3 });
  card(s, M, 4.7, 6.0, 1.55, TINT.delan);
  text(s, "Our sampled version", { x: M + 0.3, y: 4.8, w: 5.4, h: 0.45, fontSize: 17, bold: true,
    color: C.accent1, valign: "middle" });
  text(s, "Uses the speed the joint will have at the end of the step, when the next command arrives. Same "
    + "model, one change.",
    { x: M + 0.3, y: 5.25, w: 5.4, h: 0.9, fontSize: 14, color: C.accent3 });
  text(s, "Works with any of the three models, unchanged", { x: M, y: 6.4, w: 6.0, h: 0.4, fontSize: 14,
    bold: true, color: C.accent3 });
  s.addChart(pres.charts.BAR, [
      { name: "Standard", labels: ["Root", "Shoulder", "Elbow", "Wrist"], values: [3.83, 11.69, 14.61, 22.98] },
      { name: "Sampled (ours)", labels: ["Root", "Shoulder", "Elbow", "Wrist"], values: [0.0745, 0.0602, 0.0424, 0.0321] }],
    chartStyle({ x: 7.0, y: 1.45, w: 5.73, h: 5.35, barDir: "col", chartColors: [HEX.accent5, MODEL.delan],
      valAxisLogScaleBase: 10, valAxisMinVal: 0.01, valAxisMaxVal: 100, valAxisLabelFormatCode: "0.0#",
      showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: "0.0#",
      showValAxisTitle: true, valAxisTitle: "Tracking error (degrees, log scale)", showTitle: true,
      title: "Exact model, standard vs sampled control", objectName: "ct_chart" }));
  s.addNotes("Outer loop: a_cmd = qdd_ref + Kd*e_dot + Kp*e with natural frequency 15 rad/s and damping "
    + "ratio 1, at 100 Hz. Classic law: tau = ID(q, qd, a_cmd). Sampled law: tau = ID(q, qd + dt*a_cmd, "
    + "a_cmd). The elbow and wrist have M/D of 1e-3 and 6e-5 s against a 10 ms hold, so under the classic law "
    + "even the exact model tracks at 15 degrees RMS, worse than PD alone; the error grows from root to wrist "
    + "as M/D shrinks. For an inertia-dominated joint the two laws coincide. Links to Assignment 3.");

  // E4. Trajectories -----------------------------------------------------------------------------------------
  s = step("E · The paths the arm must follow", S2, 4);
  img(s, R("evaluation_control/results/figures/reference_trajectories_main.png"), M, 1.45, 7.6, 4.49,
    "The 10 main reference trajectories, joint angle against time for each joint");
  caption(s, "The 10 main paths, one panel per joint (angles in radians)", M, 6.0, 7.6);
  const tsets = [
    { h: "10 main paths", c: C.accent1, d: "5 smooth wave patterns and 5 point-to-point moves, 5 s each. "
      + "Entirely inside the range of the training data." },
    { h: "4 out-of-range paths", c: C.accent5, d: "Faster waves and quicker moves. Only 8–67 % of each "
      + "path lies inside the training range." },
    { h: "Safety checks", c: C.accent3, d: "Every path stays 0.2 rad (11°) away from the joint limits and "
      + "needs at most 70 % of the motor strength." },
  ];
  for (let i = 0; i < 3; i++) {
    const y = 1.5 + i * 1.78;
    card(s, 8.5, y, 4.23, 1.6);
    text(s, tsets[i].h, { x: 8.75, y: y + 0.15, w: 3.8, h: 0.4, fontSize: 17, bold: true, color: tsets[i].c });
    text(s, tsets[i].d, { x: 8.75, y: y + 0.55, w: 3.8, h: 1.0, fontSize: 13 });
  }
  s.addNotes("Fourier series with 3 harmonics of a 0.1-0.3 Hz base, and quintic minimum-jerk moves through 4 "
    + "waypoints; closed-form position, speed and acceleration. Seed 2026. OOD set: 0.4-0.56 Hz base, or 6 "
    + "waypoints with 1 s segments, up to 90 % of the torque limit. Figure from evaluation_control/results.");

  // E5. Tracking results ---------------------------------------------------------------------------------------
  s = step("E · Tracking results", S2, 4);
  const trLabels = ["10 normal paths", "4 out-of-range paths"];
  s.addChart(pres.charts.BAR, [
      { name: "Exact model", labels: trLabels, values: [0.057, 0.339] },
      { name: "DeLaN", labels: trLabels, values: [0.213, 0.584] },
      { name: "MLP", labels: trLabels, values: [0.305, 11.57] },
      { name: "No model (PD only)", labels: trLabels, values: [2.93, 7.22] }],
    chartStyle({ x: M, y: 1.45, w: 7.9, h: 5.35, barDir: "col",
      chartColors: [MODEL.analytical, MODEL.delan, MODEL.mlp, MODEL.pd], valAxisLogScaleBase: 10,
      valAxisMinVal: 0.01, valAxisMaxVal: 100, valAxisLabelFormatCode: "0.0#", showValue: true,
      dataLabelPosition: "outEnd", dataLabelFormatCode: "0.0#", barGapWidthPct: 50,
      showValAxisTitle: true, valAxisTitle: "Average tracking error (degrees, log scale)", showTitle: true,
      title: "How far the arm strays from the path (sampled control)", objectName: "tracking_chart" }));
  stat(s, { x: 9.0, y: 1.5, w: 3.73, big: "0.21°", label: "DeLaN on normal paths; 14× better than "
    + "no model", color: MODEL.delan, bigSize: 44 });
  stat(s, { x: 9.0, y: 3.35, w: 3.73, big: "0.31°", label: "MLP on normal paths: also good, because control "
    + "doesn't need the model run backwards", color: MODEL.mlp, bigSize: 44, labelH: 1.1 });
  stat(s, { x: 9.0, y: 5.35, w: 3.73, big: "11.6°", label: "MLP out of range: worse than no model at all",
    color: HEX.accent5, bigSize: 44, labelH: 0.6 });
  s.addNotes("Mean RMS joint error over the trajectories, in degrees (radians: exact 1.0e-3, DeLaN 3.7e-3, "
    + "MLP 5.3e-3, PD 5.1e-2; out of range 5.9e-3, 1.0e-2, 0.20, 0.13). Out of range, DeLaN degrades 2.7x and "
    + "the MLP 38x. With a 0.1 rad initial error all three models end within about 12 % of each other "
    + "(0.73-0.81 degrees), since the controller's correction dominates. No motor saturation on the main set.");

  // E6. OOD traces -------------------------------------------------------------------------------------------
  s = step("E · Out of range, the MLP fails", S2, 4);
  img(s, path.join(IMG, "tracking_ood.png"), M, 1.45, CW, 4.64, "Tracking error over time on an "
    + "out-of-range path, root and elbow joints");
  text(s, "One out-of-range path (only 8 % of it inside the training range). The exact model and DeLaN stay "
    + "within ±0.03 rad (±2°). The MLP's error reaches 0.2–0.4 rad (11–23°), as bad as using no model.",
    { x: M, y: 6.2, w: CW, h: 0.7, fontSize: 15 });
  s.addNotes("Trajectory ood_fourier_1. This is the same trajectory as the demo video later.");

  // ================================================================= PART 3: RESULTS
  pres.addSection({ title: S3 });
  s = await divider("03", "Results side by side", "All three models on every measure, then a live demo",
    S3, fa.FaChartBar);
  s.addNotes("Now bring everything together in one place.");

  // Scorecard ------------------------------------------------------------------------------------------------
  s = content("Scorecard: DeLaN vs the plain network", S3);
  const HD = { bold: true, color: C.background1, fill: { color: C.accent3 } };
  const cell = (t, o = {}) => ({ text: t, options: { fontSize: 14, valign: "middle", color: C.text1,
    margin: [0, 0.12, 0, 0.12], ...o } });
  const win = (who) => ({ fill: { color: who === "d" ? TINT.delan : TINT.mlp }, bold: true });
  const sc = [
    ["Measure", "Exact model", "DeLaN", "MLP", "Better"],
    ["Torque error, normal data", "0", "0.015 N·m", "0.044 N·m", "d"],
    ["Torque error, out of range", "0", "0.025 N·m", "0.45 N·m", "d"],
    ["2-s predictions that blow up (normal / out of range)", "0 / 0 %", "0 / 0 %", "31 / 92 %", "d"],
    ["Tracking error, normal paths", "0.06°", "0.21°", "0.31°", "d"],
    ["Tracking error, out-of-range paths", "0.34°", "0.58°", "11.6°", "d"],
    ["Tracking error with only 10 % of the data", "–", "0.20°", "3.8°", "d"],
    ["Inertia physically valid (positive-definite)", "100 %", "100 %", "under 2 %", "d"],
    ["Computing time per control step", "3.0 ms", "4.7 ms", "0.15 ms", "m"],
  ];
  const scRows = sc.map((r, i) => {
    if (i === 0) return r.map((v) => cell(v, HD));
    const w = r[4];
    return [cell(r[0], { fill: { color: i % 2 ? C.background1 : C.background2 } }),
      cell(r[1], { color: C.accent3, fill: { color: i % 2 ? C.background1 : C.background2 } }),
      cell(r[2], w === "d" ? win("d") : { fill: { color: i % 2 ? C.background1 : C.background2 } }),
      cell(r[3], w === "m" ? win("m") : { fill: { color: i % 2 ? C.background1 : C.background2 } }),
      cell(w === "d" ? "DeLaN" : "MLP", { bold: true, color: w === "d" ? C.accent1 : C.accent2,
        fill: { color: i % 2 ? C.background1 : C.background2 } })];
  });
  s.addTable(scRows, { x: M, y: 1.45, w: CW, colW: [5.4, 1.65, 1.65, 1.75, 1.65], rowH: 0.52,
    border: { type: "none" }, objectName: "scorecard" });
  text(s, "DeLaN wins on 7 of 8 measures; the MLP is only faster to compute. The exact model (grey) is the "
    + "reference: no learned model can beat it here.", { x: M, y: 6.3, w: CW, h: 0.6, fontSize: 15,
    color: C.accent3 });
  s.addNotes("All numbers are for 100 % training data unless stated. Tracking errors are average RMS "
    + "joint errors in degrees. Times are per state on a laptop CPU; all three fit easily inside the 10 ms "
    + "control period.");

  // Data efficiency -----------------------------------------------------------------------------------------
  s = content("How much data does each model need?", S3);
  const fr = ["10 %", "30 %", "50 %", "100 %"];
  s.addChart(pres.charts.LINE, [
      { name: "MLP", labels: fr, values: [3.79, 0.462, 0.397, 0.305] },
      { name: "DeLaN", labels: fr, values: [0.202, 0.208, 0.209, 0.213] }],
    chartStyle({ x: M, y: 1.45, w: 5.95, h: 4.6, chartColors: [MODEL.mlp, MODEL.delan], lineSize: 3,
      lineDataSymbol: "circle", lineDataSymbolSize: 9, valAxisLogScaleBase: 10, valAxisMinVal: 0.1,
      valAxisMaxVal: 10, valAxisLabelFormatCode: "0.0#", showValAxisTitle: true,
      valAxisTitle: "degrees (log scale)", showTitle: true, title: "Tracking error, 10 normal paths",
      catAxisTitle: "share of training data used", showCatAxisTitle: true, catAxisTitleFontSize: 12,
      catAxisTitleColor: HEX.accent3, catAxisTitleFontFace: "+mn-lt", objectName: "eff_tracking_chart" }));
  s.addChart(pres.charts.LINE, [
      { name: "MLP", labels: fr, values: [67.8, 41.1, 43.3, 31.1] },
      { name: "DeLaN", labels: fr, values: [0, 0, 0, 0] }],
    chartStyle({ x: 6.78, y: 1.45, w: 5.95, h: 4.6, chartColors: [MODEL.mlp, MODEL.delan], lineSize: 3,
      lineDataSymbol: "circle", lineDataSymbolSize: 9, valAxisMinVal: 0, valAxisMaxVal: 100,
      valAxisMajorUnit: 25, showValAxisTitle: true, valAxisTitle: "% of predictions", showTitle: true,
      title: "2-s predictions that blow up (normal data)", catAxisTitle: "share of training data used",
      showCatAxisTitle: true, catAxisTitleFontSize: 12, catAxisTitleColor: HEX.accent3,
      catAxisTitleFontFace: "+mn-lt", objectName: "eff_div_chart" }));
  text(s, [{ text: "DeLaN is as good with 10 % of the data as with 100 %. ", options: { bold: true,
             color: C.accent1 } },
           { text: "With 10 %, the MLP tracks 19× worse and two in three of its predictions blow up." }],
    { x: M, y: 6.2, w: CW, h: 0.7, fontSize: 15 });
  s.addNotes("Caveat: the nested subsets are random rows, so even 10 % still covers all 40 training "
    + "episodes. More data adds density along the same motions, not new kinds of motion; a stricter test would "
    + "drop whole episodes.");

  // PD inertia -----------------------------------------------------------------------------------------------
  s = content("Physically valid inertia: only DeLaN", S3);
  s.addChart(pres.charts.BAR, [
      { name: "MLP (inertia implied by the network)", labels: fr, values: [0.1, 0.4, 0.5, 1.8] },
      { name: "DeLaN (built-in inertia)", labels: fr, values: [100, 100, 100, 100] }],
    chartStyle({ x: M, y: 1.45, w: 7.4, h: 5.35, barDir: "col", chartColors: [MODEL.mlp, MODEL.delan],
      showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: "0.0", valAxisMaxVal: 110,
      valAxisMinVal: 0, valAxisMajorUnit: 25, showValAxisTitle: true, valAxisTitle: "Positive-definite (%)",
      showTitle: true, title: "10,000 random arm poses, by share of training data", catAxisTitle:
      "share of training data used", showCatAxisTitle: true, catAxisTitleFontSize: 12,
      catAxisTitleColor: HEX.accent3, catAxisTitleFontFace: "+mn-lt", objectName: "pd_chart" }));
  stat(s, { x: 8.5, y: 1.5, w: 4.23, big: "56,000", label: "DeLaN inertia matrices checked: every one is "
    + "valid and exactly symmetric", color: MODEL.delan, bigSize: 44 });
  stat(s, { x: 8.5, y: 3.45, w: 4.23, big: "under 2 %", label: "of poses where the MLP's implied inertia is "
    + "valid; it is not even symmetric", color: MODEL.mlp, bigSize: 44 });
  card(s, 8.5, 5.4, 4.23, 1.4);
  text(s, "This is why only DeLaN can be run backwards reliably in simulation (Test 1).",
    { x: 8.7, y: 5.4, w: 3.85, h: 1.4, fontSize: 15, valign: "middle", bold: true });
  s.addNotes("The MLP has no inertia matrix, so we use the derivative of its torque with respect to "
    + "acceleration (computed by autograd). On the test states it is positive-definite at 0-10.6 % of states. "
    + "Its error against the true inertia is 2.7-3.8 (relative), against about 0.6 for DeLaN.");

  // Demo --------------------------------------------------------------------------------------------------------
  s = content("Live demo", S3);
  s.addMedia({ type: "video", path: R("evaluation_control/results/tracking_demo_ood.mp4"),
    x: M, y: 1.5, w: 7.9, h: 1.975, cover: "data:image/png;base64," +
      fs.readFileSync(path.join(IMG, "video_cover.png")).toString("base64"), objectName: "demo_video" });
  text(s, "Exact model · MLP · DeLaN following the same out-of-range path in MuJoCo (5 s)",
    { x: M, y: 3.6, w: 7.9, h: 0.4, fontSize: 13, color: C.accent3 });
  const steps = [
    { n: "1", h: "Simulation test", c: "python -m evaluation_control.rollout --quick" },
    { n: "2", h: "Control test + video", c: "python -m evaluation_control.tracking_experiment --quick --video" },
    { n: "3", h: "Automated checks (30 pass)", c: "pytest -q evaluation_control delan" },
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
  text(s, "About 15–20 min on a laptop; rebuilds every table and figure", { x: 9.0, y: 2.95, w: 3.55,
    h: 0.9, fontSize: 14 });
  s.addNotes("Run from the repository root with .venv/bin/python and MUJOCO_GL=egl. The quick simulation test "
    + "and the quick tracking run take about a minute each. If time is short, just play the embedded video.");

  // ================================================================= PART 4: FINAL THOUGHTS
  pres.addSection({ title: S4 });
  s = await divider("04", "Final thoughts", "What we learned, where our study stops, and what comes next",
    S4, fa.FaLightbulb);
  s.addNotes("Wrap up.");

  // Discussion --------------------------------------------------------------------------------------------------
  s = content("What we learned", S4);
  const disc = [
    { Icon: fa.FaCheckCircle, c: MODEL.delan, h: "Physics helps most where data run out",
      b: "Out-of-range motions, small datasets, and simulation. Its valid inertia makes the model safe to "
        + "run backwards." },
    { Icon: fa.FaBullseye, c: MODEL.mlp, h: "A plain network is enough in familiar territory",
      b: "Inside the training range, the MLP controls the arm almost as well, and it computes 30× faster." },
    { Icon: fa.FaExclamationTriangle, c: HEX.accent5, h: "Good torques ≠ correct physics",
      b: "DeLaN's inertia is 33 % off, because the data barely exercise the light wrist and elbow." },
    { Icon: fa.FaSlidersH, c: MODEL.analytical, h: "The controller must suit the robot",
      b: "On this heavily damped arm, standard computed-torque control fails even with a perfect model." },
  ];
  for (let i = 0; i < 4; i++) {
    const col = i % 2, row = Math.floor(i / 2);
    await iconCard(s, { x: M + col * 6.2, y: 1.5 + row * 2.7, w: 5.9, h: 2.45, Icon: disc[i].Icon,
                        color: disc[i].c, head: disc[i].h, body: disc[i].b, bodySize: 16 });
  }
  s.addNotes("Timing per state: MLP 0.15 ms, exact model 3 ms, DeLaN 4.7 ms per control step; all fit inside "
    + "the 10 ms control period. DeLaN under the standard control law does better than the exact model only "
    + "because it over-estimates the wrist inertia: a lucky side effect, not accuracy.");

  // Limitations ----------------------------------------------------------------------------------------------
  s = content("Limits of this study and next steps", S4);
  const lim = [
    { Icon: fa.FaDice, h: "One random seed", d: "Small differences, e.g. 0.20° vs 0.21°, are not significant" },
    { Icon: fa.FaFlask, h: "Clean simulated data", d: "No sensor noise, no contacts, fingers locked" },
    { Icon: fa.FaLayerGroup, h: "Subsets share all episodes", d: "So the flat data curve partly reflects "
      + "the sampling, not just DeLaN" },
    { Icon: fa.FaClock, h: "Simulation step of 10 ms", d: "Adds about 0.06° of error by itself" },
  ];
  text(s, "Limits", { x: M, y: 1.45, w: 6, h: 0.45, fontSize: 20, bold: true, color: C.accent3,
    fontFace: THEME.headFontFace });
  for (let i = 0; i < 4; i++) {
    const y = 2.05 + i * 1.2;
    await iconCircle(s, lim[i].Icon, M, y + 0.1, 0.6, C.accent3);
    text(s, lim[i].h, { x: M + 0.85, y, w: 5.2, h: 0.4, fontSize: 16, bold: true });
    text(s, lim[i].d, { x: M + 0.85, y: y + 0.42, w: 5.2, h: 0.6, fontSize: 14, color: C.accent3 });
  }
  card(s, 6.9, 1.45, 5.83, 5.35, TINT.delan);
  text(s, "Next steps", { x: 7.2, y: 1.7, w: 5.2, h: 0.45, fontSize: 20, bold: true, color: C.accent1,
    fontFace: THEME.headFontFace });
  text(s, bullets([
    "Repeat with several seeds and with subsets that drop whole episodes",
    "Add sensor noise to the data",
    "Design motions that exercise the wrist and elbow, to pin down their inertia",
    "Try model-predictive control, and a real robot model (OpenMANIPULATOR-X)",
  ], 18, { paraSpaceAfter: 24 }), { x: 7.2, y: 2.45, w: 5.25, h: 4.1 });
  s.addNotes("Be upfront about the limits during questions. The 10 ms implicit integrator gives a floor of "
    + "about 1e-3 rad over 2 s; with 1 ms steps the floor drops to 2e-5 rad, but the MLP then fails at once.");

  // Conclusion ----------------------------------------------------------------------------------------------------
  s = pres.addSlide({ masterName: "CLOSING_DARK", sectionTitle: S4 });
  s.addText("Conclusions", { placeholder: "title" });
  const concl = [
    { big: "Structure", d: "Building the physics into the network gives accurate torques and a model that "
      + "never blew up in 720 simulations" },
    { big: "Control", d: "With DeLaN the arm follows its path to 0.2° on normal paths and 0.6° on "
      + "out-of-range ones, close to the exact model" },
    { big: "Data", d: "DeLaN reaches full accuracy with 10 % of the data; the MLP needs more and still fails "
      + "out of range" },
  ];
  for (let i = 0; i < 3; i++) {
    const x = M + i * 4.1;
    text(s, concl[i].big, { x, y: 1.75, w: 3.8, h: 0.7, fontSize: 30, bold: true, color: C.accent6,
      fontFace: THEME.headFontFace });
    text(s, concl[i].d, { x, y: 2.6, w: 3.8, h: 1.9, fontSize: 17, color: C.background1 });
  }
  card(s, M, 4.95, CW, 1.4, "243140");
  text(s, [{ text: "Bottom line:  ", options: { bold: true, color: C.accent6 } },
           { text: "when a learned model must work outside its training data or be used as a simulator, "
             + "building physics into it is worth the extra cost.", options: { color: C.background1 } }],
    { x: M + 0.35, y: 4.95, w: CW - 0.7, h: 1.4, fontSize: 19, valign: "middle" });
  text(s, "Thank you. Questions?", { x: M, y: 6.55, w: 6, h: 0.45, fontSize: 18, bold: true,
    color: C.accent4 });
  s.addNotes("Close on the bottom line and invite questions. Each member should be ready to explain their "
    + "step: data (A), physics model (B), MLP (C), DeLaN (D), simulation and control (E).");

  await pres.writeFile({ fileName: OUT });
  await applyTheme(OUT, THEME);
  console.log("wrote", OUT);
}

build().catch((e) => { console.error(e); process.exit(1); });
