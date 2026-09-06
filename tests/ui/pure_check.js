const P = module.exports;
const assert = require("assert");
const view = {s0: 0.5, s1: 3.5};
assert.strictEqual(P.viewBoxOf(view, 120), "500 0 3000 120");
assert.ok(Math.abs(P.sOf(view, 800, P.pxOf(view, 800, 2.0)) - 2.0) < 1e-12);
const lim = P.limitsFor(10);
const z = P.zoomAt({s0: 0, s1: 10}, 800, 400, 2, lim);
assert.ok(Math.abs((z.s1 - z.s0) - 5) < 1e-12 && Math.abs(P.sOf(z, 800, 400) - 5) < 1e-9, "zoom keeps the anchor");
const t = P.axisTicks({s0: 0, s1: 6.6}, 900);
assert.ok([1, 2, 5, 10].some(m => Math.abs(t.step - m * Math.pow(10, Math.floor(Math.log10(t.step)))) < 1e-12));
assert.ok(t.major.length >= 3 && t.major[0] >= 0);
const arc = P.arcPoints(0, 0, 0, Math.PI / 2, Math.PI / 2, 8);
assert.ok(Math.abs(arc[arc.length - 1][0] + 1) < 1e-9 && Math.abs(arc[arc.length - 1][1] - 1) < 1e-9, "a +90 deg bend of rho=1 ends at (X=-1, Z=1)");
const arc2 = P.arcPoints(0, 0, 0, Math.PI, Math.PI, 16);
assert.ok(Math.abs(arc2[arc2.length - 1][0] + 2) < 1e-9 && Math.abs(arc2[arc2.length - 1][1]) < 1e-9, "180 deg");
const lay = P.layoutLabels([{i: 0, cx: 10, w: 30}, {i: 1, cx: 20, w: 30}, {i: 2, cx: 30, w: 30}, {i: 3, cx: 100, w: 30}]);
assert.strictEqual(lay.get(0), 0); assert.strictEqual(lay.get(1), 1); assert.strictEqual(lay.get(2), undefined); assert.strictEqual(lay.get(3), 0);
const kinds = ["Drift","Quadrupole","Sextupole","Octupole","Multipole","Bend","Solenoid","RFCavity","FieldMap","NCells","RFQCell","Kicker","Collimator","Marker","Instrument","Foil","Taylor","Patch","ReferenceChange","Freq","Directive","Superposition"];
for (const kind of kinds) for (const L of [0, 0.3]) {
  const row = {i: 0, kind, s_in: 1.0, s_out: 1.0 + L, L, glyph: {sign: 1, plane: "h", angle: 0.1, k1: 2, volt: 1e6, hx: 1, vy: -1, family: "BPM", map_kind: "rf", integrated: true, order: 2}, contrib: {BnL2: 1, BnL3: 1}};
  const g = P.glyphFor(row, {}, {k1: 2, voltage_V: 1e6, k2: 1, k3: 1});
  assert.ok(g.hidden || g.body || g.px.length, kind + " draws something");
}
assert.strictEqual(P.fmt.num(1234.5678, 6), "1234.57"); assert.strictEqual(P.fmt.num(0), "0"); assert.strictEqual(P.fmt.num(1.5e-7, 3), "1.5e-7");
console.log("pure ok");
