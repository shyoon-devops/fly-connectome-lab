// The NeuroMechFly Live game, in the browser. Pilot the fly through a slalom
// track to the finish line at three levels of neural abstraction:
//
//   Level 1 "CPG"    -- steer (W/A/S/D); coupled CPG oscillators coordinate all
//                       six legs automatically. Easy.
//   Level 2 "Tripod" -- drive the two tripod groups (G/H forward, F/J back).
//   Level 3 "Legs"   -- drive each of the six legs (T G B Z H N / R F V U J M).
//
// The physics is real: MuJoCo (compiled to WebAssembly, shared under
// ../shared/) runs the same legs-only position-actuated NeuroMechFly model as
// the desktop game, at dt=1e-4 with leg adhesion -- so, like the desktop game,
// it plays at well below real time (the achieved factor is shown top-right).
// The control logic ported here (CPGNetwork + PreprogrammedSteps + the three
// controllers, from neuromechfly-live / flygym) is fed the *baked* tables in
// model_meta.json, so the browser needs no SciPy. Three.js renders the solved
// state; the camera chases the fly; the finish is a geometric path/line crossing.

import * as THREE from 'three';
import {
  loadScene, makeFailOverlay, makeStepper, makeStatsMeter, buildMeshes, syncMeshes,
} from '../shared/scene.js';

const ASSETS = './assets';
const TAU = Math.PI * 2;
const overlayEl = document.getElementById('overlay');
const overlayMsg = document.getElementById('overlay-msg');

// Target playback speed: sim seconds advanced per wall second. The desktop game
// plays well below real time; we pin it at a fixed slow-motion factor (rather
// than "as fast as the machine allows") so the fly is controllable and the pace
// is the same on every machine.
const PLAYBACK_SPEED = 0.1;
// Safety cap on physics steps per animation frame. At PLAYBACK_SPEED=0.1 and
// ~60 fps a frame only needs ~17 steps, well under this; the cap just prevents a
// long stall from spiralling.
const MAX_SUBSTEPS = 60;

const LEVELS = {
  brain:  { name: 'MaleCNS whole-brain autopilot', chip: '#58f2b0' },
  CPG:    { name: 'CPG control',        chip: '#8ace00' },
  tripod: { name: 'Tripod gait',        chip: '#4d66ff' },
  single: { name: 'Individual legs',    chip: '#ff5a5a' },
};
const STIMULUS_DEFS = {
  food:     { label:'먹이 냄새', color:0xffc857, channel:'후각·미각', range:9,  radius:.42 },
  light:    { label:'밝은 빛',   color:0x73d8ff, channel:'시각',      range:14, radius:.38 },
  threat:   { label:'포식자',    color:0xff4f70, channel:'시각·위협', range:10, radius:.65 },
  wind:     { label:'바람',      color:0x75f4da, channel:'기계감각',   range:8,  radius:.7 },
  heat:     { label:'뜨거운 곳', color:0xff8a3d, channel:'온도',      range:6,  radius:1.25 },
  obstacle: { label:'장애물',    color:0xa99cff, channel:'시각·접촉', range:4,  radius:1.05 },
  boundary: { label:'경기장 경계',color:0x58f2b0, channel:'기계감각',   range:3,  radius:0 },
};
const HELP = {
  brain: '<b>전체 MaleCNS 자동조종</b> · 감각 → 166,700 뉴런 → DNp09 / MDN / DNa → 6-leg CPG' +
       '<div class="row2">MuJoCo dt=0.1 ms · 중력 9.81 m/s² · 접촉·마찰·관절·부착력 활성</div>',
  CPG: '<kbd>W</kbd> forward · <kbd>S</kbd> back · <kbd>A</kbd>/<kbd>D</kbd> turn · <kbd>Q</kbd> stop' +
       '<div class="row2">Switch level: <kbd>1</kbd>/<kbd>2</kbd>/<kbd>3</kbd> · Restart: <kbd>Space</kbd></div>',
  tripod: '<kbd>G</kbd>/<kbd>H</kbd> step left/right tripod forward · <kbd>F</kbd>/<kbd>J</kbd> backward' +
       '<div class="row2">Each tripod is 3 alternating legs. Switch: <kbd>1</kbd>/<kbd>2</kbd>/<kbd>3</kbd> · Restart: <kbd>Space</kbd></div>',
  single: 'Forward <kbd>T</kbd><kbd>G</kbd><kbd>B</kbd> <kbd>Z</kbd><kbd>H</kbd><kbd>N</kbd> · ' +
       'Back <kbd>R</kbd><kbd>F</kbd><kbd>V</kbd> <kbd>U</kbd><kbd>J</kbd><kbd>M</kbd>' +
       '<div class="row2">Six legs, one key each (L/R front·mid·hind). Switch: <kbd>1</kbd>/<kbd>2</kbd>/<kbd>3</kbd> · Restart: <kbd>Space</kbd></div>',
};
// Joystick hint appended to the help line while a gamepad is connected. Mirrors
// the desktop game's joystick layout (see the Gamepad class / controls.py).
const HELP_PAD = {
  brain: '🧠 자율주행 중 — MANUAL CPG 탭에서 직접 조종할 수 있습니다',
  CPG: '🎮 Push the stick to walk · left/right to steer',
  tripod: '🎮 Left & right tripod buttons step each tripod (forward / reverse rows)',
  single: '🎮 One button per leg — forward and reverse button rows',
};

const fail = makeFailOverlay(overlayEl, 'game', 'p');

main().catch((e) => fail('Unexpected error while starting up.', e));

async function main() {
  const { mj, model, data, meta } = await loadScene({
    assetsDir: ASSETS, xmlName: 'fly.xml',
    onStage: (msg) => { overlayMsg.textContent = msg; },
  });
  new Game(mj, model, data, meta).start();
}

// --- the ported controllers -------------------------------------------------
// One object integrates the CPG (level 1) and the per-leg / per-tripod step
// state machines (levels 2-3), then scatters the resulting joint angles +
// adhesion flags into MuJoCo's `data.ctrl` via the baked (leg, dof) map.
class Controller {
  constructor(meta) {
    this.dt = meta.timestep;
    this.legs = meta.control.leg_order;            // 6 leg names
    this.cmap = meta.ctrl_index_by_leg_dof;        // [6][7] -> ctrl index
    this.adh = meta.adhesion;                       // [6] -> adhesion ctrl index
    this.tripodMap = meta.control.tripod_map;       // [6] -> 0/1
    const cpg = meta.control.cpg;
    this.freqs0 = cpg.intrinsic_freqs.slice();      // base |freq| per leg
    this.W = cpg.coupling_weights;                  // [6][6]
    this.PB = cpg.phase_biases;                     // [6][6]
    this.conv = cpg.convergence_coefs;              // [6]
    this.phaseInc = (this.dt / meta.control.leg_step_time) * TAU;

    const pp = meta.preprogrammed;
    this.N = pp.n_samples;
    this.tab = this.legs.map((l) => pp.legs[l]);    // {angles:[N][7], neutral:[7], swing:[2]}

    // scratch
    this._cpgAmps = new Float64Array(6);
    this._cpgFreqs = new Float64Array(6);
    this._d6 = new Float64Array(6);
    this._a7 = new Float64Array(7);
    this.reset();
  }

  reset() {
    this.phases = new Float64Array(6).map(() => Math.random() * TAU); // CPG phases
    this.mags = new Float64Array(6);                                  // CPG magnitudes
    this.legPhases = new Float64Array(6);                             // single-leg
    this.stepDir = new Float64Array(6);
    this.tripodPhases = new Float64Array(2);                          // tripod groups
    this.tripodDir = new Float64Array(2);
  }

  // Joint angles for one leg at a phase / magnitude, by periodic-lerp of the
  // baked table: angle = neutral + magnitude * (table(phase) - neutral).
  _anglesInto(li, phase, mag, out) {
    const t = this.tab[li], N = this.N;
    let x = ((phase % TAU) + TAU) % TAU / TAU * N;
    const i0 = Math.floor(x) % N, i1 = (i0 + 1) % N, f = x - Math.floor(x);
    const a0 = t.angles[i0], a1 = t.angles[i1], nu = t.neutral;
    for (let d = 0; d < 7; d++) {
      const samp = a0[d] * (1 - f) + a1[d] * f;
      out[d] = nu[d] + mag * (samp - nu[d]);
    }
  }

  _adhesionOn(li, phase) {
    const [s, e] = this.tab[li].swing;             // swing = adhesion OFF
    const p = ((phase % TAU) + TAU) % TAU;
    return !(p > s && p < e);
  }

  _writeLeg(ctrl, li, phase, mag) {
    this._anglesInto(li, phase, mag, this._a7);
    const row = this.cmap[li];
    for (let d = 0; d < 7; d++) ctrl[row[d]] = this._a7[d];
    ctrl[this.adh[li]] = this._adhesionOn(li, phase) ? 1 : 0;
  }

  // Level 1: descending signal action=[gainL,gainR] modulates CPG amplitude
  // (|action|) and stepping direction (sign), then one Euler integration step.
  stepCPG(ctrl, gainL, gainR) {
    const amps = this._cpgAmps, freqs = this._cpgFreqs;
    const aL = Math.abs(gainL), aR = Math.abs(gainR);
    amps[0] = amps[1] = amps[2] = aL; amps[3] = amps[4] = amps[5] = aR;
    const sL = gainL > 0 ? 1 : -1, sR = gainR > 0 ? 1 : -1;
    for (let i = 0; i < 6; i++) freqs[i] = this.freqs0[i] * (i < 3 ? sL : sR);

    // dtheta = 2pi*freq + sum_j mags_j*W_ij*sin(theta_j - theta_i - PB_ij);  dr = conv*(amp - r)
    const ph = this.phases, mg = this.mags, dt = this.dt;
    const dph = this._d6;
    for (let i = 0; i < 6; i++) {
      let coupling = 0;
      for (let j = 0; j < 6; j++)
        coupling += mg[j] * this.W[i][j] * Math.sin(ph[j] - ph[i] - this.PB[i][j]);
      dph[i] = TAU * freqs[i] + coupling;
    }
    for (let i = 0; i < 6; i++) {
      ph[i] += dph[i] * dt;
      mg[i] += this.conv[i] * (amps[i] - mg[i]) * dt;
    }
    for (let i = 0; i < 6; i++) this._writeLeg(ctrl, i, ph[i], mg[i]);
  }

  // Shared per-step state machine for the on-demand modes: a leg/tripod at rest
  // (phase<=0) starts a step when its action is non-zero, runs the cycle to
  // completion (forward to 2pi, or backward to 0), then returns to rest.
  _advance(phaseArr, dirArr, i, act) {
    if (phaseArr[i] >= TAU || (phaseArr[i] <= 0 && dirArr[i] < 0)) {
      phaseArr[i] = 0; dirArr[i] = 0;
    } else if (phaseArr[i] <= 0) {
      if (act > 0) { phaseArr[i] += this.phaseInc; dirArr[i] = 1; }
      else if (act < 0) { phaseArr[i] = TAU - this.phaseInc; dirArr[i] = -1; }
    } else {
      phaseArr[i] += this.phaseInc * dirArr[i];
    }
  }

  // Level 3: action is a 6-vector (one trigger per leg).
  stepSingle(ctrl, action) {
    for (let i = 0; i < 6; i++) {
      this._advance(this.legPhases, this.stepDir, i, action[i]);
      this._writeLeg(ctrl, i, this.legPhases[i], 1);
    }
  }

  // Level 2: action is a 2-vector (one trigger per tripod group).
  stepTripod(ctrl, action) {
    for (let g = 0; g < 2; g++) this._advance(this.tripodPhases, this.tripodDir, g, action[g]);
    for (let i = 0; i < 6; i++)
      this._writeLeg(ctrl, i, this.tripodPhases[this.tripodMap[i]], 1);
  }
}

// --- keyboard input ---------------------------------------------------------
// CPG gains persist (set on key-down, like the desktop's prev_gain); the
// on-demand modes read the currently-held keys each substep.
class Input {
  constructor(onLevel, onRestart, onMove) {
    this.held = new Set();
    this.gainL = 0; this.gainR = 0;
    const MOVE = 'wsadqtgbzhnrfvujm';
    // Level 1 (CPG) arrow-key aliases for WASD.
    const ARROW = { arrowup: 'w', arrowdown: 's', arrowleft: 'a', arrowright: 'd' };
    addEventListener('keydown', (e) => {
      const k = ARROW[e.key.toLowerCase()] || e.key.toLowerCase();
      if (e.repeat) { e.preventDefault(); return; }
      if (k === '0' || k === 'b') return onLevel('brain');
      if (k === '1' || k === 'i') return onLevel('CPG');
      if (k === '2' || k === 'o') return onLevel('tripod');
      if (k === '3' || k === 'p') return onLevel('single');
      // Restart is Space only: in Level 3 'r' is the left-front leg's "backward"
      // key (see singleAction), so it must fall through to `held` below.
      if (k === ' ') { e.preventDefault(); return onRestart(); }
      this.held.add(k);
      this._cpgKey(k);
      if (MOVE.includes(k)) { e.preventDefault(); onMove(); }
    });
    addEventListener('keyup', (e) => {
      const k = e.key.toLowerCase();
      this.held.delete(ARROW[k] || k);
    });
    addEventListener('blur', () => { this.held.clear(); });
  }

  _cpgKey(k) {
    const back = this.gainL < 0 || this.gainR < 0;
    if (k === 'w') { this.gainL = 1; this.gainR = 1; }
    else if (k === 's') { this.gainL = -1; this.gainR = -1; }
    else if (k === 'q') { this.gainL = 0; this.gainR = 0; }
    else if (k === 'a') { if (back) { this.gainR = -0.6; this.gainL = -1.2; } else { this.gainL = 0.4; this.gainR = 1.2; } }
    else if (k === 'd') { if (back) { this.gainL = -0.6; this.gainR = -1.2; } else { this.gainR = 0.4; this.gainL = 1.2; } }
  }

  resetGains() { this.gainL = 0; this.gainR = 0; }

  // Level 3: per-leg trigger from held keys (forward / backward sets).
  singleAction(out) {
    const F = 'tgbzhn', B = 'rfvujm';
    for (let i = 0; i < 6; i++)
      out[i] = this.held.has(F[i]) ? 1 : this.held.has(B[i]) ? -1 : 0;
    return out;
  }

  // Level 2: per-tripod trigger. Group 0 = G/F, group 1 = H/J.
  tripodAction(out) {
    out[0] = this.held.has('g') ? 1 : this.held.has('f') ? -1 : 0;
    out[1] = this.held.has('h') ? 1 : this.held.has('j') ? -1 : 0;
    return out;
  }
}

// --- gamepad / joystick input ----------------------------------------------
// Faithful port of the desktop game's JoystickControl (neuromechfly-live
// controls.py) onto the browser Gamepad API: the *same* raw button indices and
// the *same* CPG axis math, so the physical joystick used at outreach events
// behaves identically in the browser. Button indices are device-specific (they
// match the joystick the desktop game targets); tweak PAD if you use another.
const PAD = {
  // leg order LF, LM, LH, RF, RM, RH (== meta.control.leg_order). controls.py:
  //   joystick_buttons_order          = [10, 11, 12, 4, 5, 6]  -> step forward
  //   backward_joystick_buttons_order = [15, 14, 13, 9, 8, 7]  -> step backward
  fwdButtons:  [10, 11, 12, 4, 5, 6],
  backButtons: [15, 14, 13, 9, 8, 7],
  axisX: 0, axisY: 1,     // analog stick: X = turn, Y = forward/back (fwd = -1)
  deadzone: 0.15,         // ignore stick drift (desktop polled raw axes)
};

class Gamepad {
  constructor(onChange) {
    this.index = null;
    this._disconnected = false; // active pad was unplugged; don't auto-adopt another
    this.single = new Float64Array(6);
    this.tripod = new Float64Array(2);
    this._legs = new Float64Array(6);
    addEventListener('gamepadconnected', (e) => {
      this.index = e.gamepad.index; this._disconnected = false; onChange?.();
    });
    addEventListener('gamepaddisconnected', (e) => {
      if (this.index === e.gamepad.index) { this.index = null; this._disconnected = true; }
      onChange?.();
    });
  }

  get connected() { return this._pad() != null; }

  _pad() {
    const pads = navigator.getGamepads ? navigator.getGamepads() : [];
    if (this.index != null && pads[this.index]) return pads[this.index];
    // Discover a pad already connected before page load (no 'gamepadconnected'
    // event). Skip once the active pad has been unplugged, so input doesn't
    // silently jump to a different controller the player isn't holding.
    if (!this._disconnected) {
      for (const p of pads) if (p) { this.index = p.index; return p; }
    }
    return null;
  }

  _axis(pad, i) {
    const v = pad.axes[i] || 0;
    return Math.abs(v) < PAD.deadzone ? 0 : v;
  }

  // controls.py retrieve_joystick_buttons: +1 forward / -1 backward per leg.
  _legPresses(pad) {
    const legs = this._legs; legs.fill(0);
    const down = (b) => pad.buttons[b] && pad.buttons[b].pressed;
    for (let j = 0; j < 6; j++) if (down(PAD.fwdButtons[j])) legs[j] = 1;
    for (let j = 0; j < 6; j++) if (down(PAD.backButtons[j])) legs[j] = -1;
    return legs;
  }

  // One snapshot for the active level. Returns null when no pad is connected;
  // otherwise { active, gainL, gainR } plus this.single / this.tripod filled in.
  // `active` (any input past the deadzone) is used to auto-start the countdown.
  sample(level) {
    const pad = this._pad();
    if (!pad) return null;
    this.single.fill(0); this.tripod.fill(0);
    let gainL = 0, gainR = 0, active = false;

    if (level === 'CPG') {
      // controls.py CPG branch: ||axis||/sqrt(2)*1.2 sets the forward/back
      // magnitude, axisY sign its direction (stick forward reads negative), and
      // |axisX|*0.6 subtracts from one side to turn (right: -=right, left: -=left).
      const ax = this._axis(pad, PAD.axisX), ay = this._axis(pad, PAD.axisY);
      const norm = Math.hypot(ax, ay) / Math.SQRT2 * 1.2;
      const sy = ay > 0 ? 1 : ay < 0 ? -1 : 0;
      gainL = gainR = norm * -1 * sy;
      const off = Math.abs(ax) * 0.6;
      if (ax > 0) gainR -= off; else if (ax < 0) gainL -= off;
      active = ax !== 0 || ay !== 0;
    } else {
      const legs = this._legPresses(pad);
      if (level === 'single') {                       // 6-vector, one per leg
        for (let i = 0; i < 6; i++) this.single[i] = legs[i];
      } else {                                        // tripod: LH btn -> left, RH btn -> right
        this.tripod[0] = legs[2];                     // group 0 = legs LF/LH/RM
        this.tripod[1] = legs[5];                     // group 1 = legs LM/RF/RH
      }
      for (let i = 0; i < 6; i++) if (legs[i]) { active = true; break; }
    }
    return { active, gainL, gainR };
  }
}

// A repeating greyscale checkerboard texture (white / mid-grey). Tinted by the
// ground material's per-level colour; mipmapped + anisotropic so it doesn't
// shimmer into the distance.
function makeCheckerTexture() {
  const N = 256, n = 8, s = N / n;       // 8x8 squares, 32 px each
  const c = document.createElement('canvas');
  c.width = c.height = N;
  const ctx = c.getContext('2d');
  ctx.fillStyle = '#ffffff'; ctx.fillRect(0, 0, N, N);
  ctx.fillStyle = '#9c9c9c';
  for (let i = 0; i < n; i++)
    for (let j = 0; j < n; j++)
      if ((i + j) & 1) ctx.fillRect(i * s, j * s, s, s);
  const tex = new THREE.CanvasTexture(c);
  tex.wrapS = tex.wrapT = THREE.RepeatWrapping;
  tex.colorSpace = THREE.SRGBColorSpace;
  tex.anisotropy = 8;
  return tex;
}

// --- the game ---------------------------------------------------------------
class Game {
  constructor(mj, model, data, meta) {
    this.mj = mj; this.model = model; this.data = data; this.meta = meta;
    this.dt = meta.timestep;
    this.level = 'CPG';
    this.controller = new Controller(meta);
    this.input = new Input(
      (lv) => this.setLevel(lv),
      () => this.restart(),
      () => { if (this.phase === 'ready') this._startCountdown(); });
    this.pad = new Gamepad(() => this._renderHelp());
    this._padState = null;
    this.brain = { gain_left: 0, gain_right: 0, confidence: 0 };
    this.brainLearning = true;
    this.brainThreat = 0;
    this.brainState = null;
    this.brainSocket = null;
    this._lastSenseWall = 0;
    this._lastTargetDistance = null;
    this.gateIndex = 0;
    this.collisionCount = 0;
    this.waypoints = Array.from({ length: meta.arena.n_gates }, (_, i) => [
      (i + 1) * meta.arena.gate_spacing, i % 2 === 0 ? 2.0 : -2.0,
    ]);
    this.arenaBounds = { minX:-4, maxX:56, minY:-9, maxY:9 };

    this.finish = meta.arena.finish_line;          // [[x,y1],[x,y2]]
    this.phase = 'ready';                           // ready | countdown | running | paused | finished
    this.simTime = 0;                               // sim seconds since GO
    this.prevXY = [meta.arena.spawn[0], meta.arena.spawn[1]];
    this.bodyId = this._findFlyRootBody();
    this.rootJointId = this._findFlyRootJoint();
    this.rootQposAdr = Number(model.jnt_qposadr?.[this.rootJointId]) || 0;
    this.rootDofAdr = Number(model.jnt_dofadr?.[this.rootJointId]) || 0;
    this.bodyMass = Number(model.body_subtreemass?.[this.bodyId]) ||
      Array.from(model.body_mass || []).reduce((sum, value) => sum + Number(value || 0), 0) || .001;
    this.gravity = Math.abs(Number(model.opt?.gravity?.[2])) || 9810;
    this.mobilityMode = 'auto';
    this.flightThrottle = 0;
    this.flightTargetZ = 6.5;
    this.flightReason = '';
    this._flightHeading = null;
    this._flightSteer = 0;
    this._flightBank = 0;
    this._invertedFor = 0;
    this._righting = false;
    this._manualRighting = 0;
    this.stimuli = [];
    this.placementKind = null;
    this.senses = { odor:0, visual:0, touch:0, taste:0, temperature:0, humidity:0,
      threat:0, wind:0, targetDistance:0, targetBearing:0, target:null, reaction:'자극 없음' };
    this._lastSenseHud = 0;

    this._singleAct = new Float64Array(6);
    this._tripodAct = new Float64Array(2);
    this._stepper = makeStepper(this.dt, MAX_SUBSTEPS);
    this._statsMeter = makeStatsMeter(this.dt, ({ fps, rtf }) => {
      document.getElementById('stats').innerHTML =
        `${fps.toFixed(0)} fps · ${rtf.toFixed(2)}× realtime<br>${this.data.ncon} contacts`;
    });
    this._buildScene();
    this._wireUi();
    this._connectBrain();
    addEventListener('message',(event)=>{if(event.origin===location.origin&&event.data?.type==='nmf-parent-command')this._handleParentCommand(event.data.message)});
  }

  _findFlyRootBody() {
    const m = this.model;
    for (let j = 0; j < m.njnt; j++) if (m.jnt_type[j] === 0) return m.jnt_bodyid[j]; // free joint
    return 1;
  }

  start() {
    this._resetSim();
    overlayEl.classList.add('hidden');
    this._showReady();
    requestAnimationFrame((t) => this._frame(t));
  }

  _findFlyRootJoint() {
    const m=this.model;
    for(let j=0;j<m.njnt;j++)if(m.jnt_type[j]===0)return j;
    return 0;
  }

  // --- MaleCNS bridge -----------------------------------------------------
  _connectBrain() {
    const proto = location.protocol === 'https:' ? 'wss' : 'ws';
    this.brainSocket = new WebSocket(`${proto}://${location.host}/ws/sim`);
    this.brainSocket.onopen = () => {
      document.getElementById('brain-link').textContent = 'LIVE';
      if (this.phase === 'running' && this.level === 'brain') this._brainSend({ type:'run', value:true });
    };
    this.brainSocket.onclose = () => {
      document.getElementById('brain-link').textContent = '재연결 중';
      setTimeout(() => this._connectBrain(), 1500);
    };
    this.brainSocket.onmessage = (event) => {
      const msg = JSON.parse(event.data);
      if (msg.type !== 'state') return;
      this.brainState = msg;
      this.brain = msg.arena?.command || this.brain;
      this._updateBrainHud(msg);
      if (window.parent !== window) window.parent.postMessage({ type:'nmf-brain-state', state:msg }, location.origin);
    };
    addEventListener('beforeunload', () => this._brainSend({ type:'run', value:false }), { once:true });
    document.addEventListener('visibilitychange', () => {
      if (document.hidden) this._brainSend({ type:'run', value:false });
      else if (this.phase === 'running' && this.level === 'brain') this._brainSend({ type:'run', value:true });
    });
  }

  _brainSend(message) {
    if (this.brainSocket?.readyState === WebSocket.OPEN) this.brainSocket.send(JSON.stringify(message));
  }

  _announceRunState() {
    if (window.parent !== window) window.parent.postMessage({
      type:'nmf-run-state', phase:this.phase, running:this.phase === 'running',
    }, location.origin);
  }

  _handleParentCommand(message) {
    if (message?.type !== 'run') { this._brainSend(message); return; }
    if (message.value) {
      if (this.phase === 'ready' || this.phase === 'finished') this._startCountdown();
      else if (this.phase === 'paused') this._resumeRun();
      else if (this.phase === 'running' && this.level === 'brain') this._brainSend({type:'run',value:true});
    } else if (this.phase === 'running' || this.phase === 'countdown') {
      this._pauseRun();
    } else {
      this._brainSend({type:'run',value:false});
    }
  }

  _pauseRun() {
    this.phase = 'paused';
    this._brainSend({type:'run',value:false});
    overlayEl.classList.remove('hidden');
    overlayEl.innerHTML = '<h1>일시정지</h1><p>뇌 계산과 3D 물리가 함께 멈췄습니다.</p><button id="resume-run">계속 실행 ▶</button>';
    document.getElementById('resume-run').onclick = () => this._resumeRun();
    this._announceRunState();
  }

  _resumeRun() {
    this.phase = 'running';
    this._lastWall = undefined;
    overlayEl.classList.add('hidden');
    if (this.level === 'brain') this._brainSend({type:'run',value:true});
    this._announceRunState();
  }

  _updateBrainHud(msg) {
    const a = msg.arena;
    if (!a?.motors || !a?.command) return;
    const m = a.motors, c = a.command;
    document.getElementById('brain-active').textContent = Number(msg.active_neurons).toLocaleString();
    document.getElementById('brain-fb').textContent = `${m.DNp09_forward.toFixed(3)} / ${m.MDN_backward.toFixed(3)}`;
    document.getElementById('brain-lr').textContent = `${m.DNa_left.toFixed(3)} / ${m.DNa_right.toFixed(3)}`;
    document.getElementById('brain-memory').textContent = a.odor_memory.toFixed(3);
    const mean = (c.gain_left + c.gain_right) / 2, delta = c.gain_right - c.gain_left;
    const action = mean < -.15 ? 'MDN · 후진' : Math.abs(delta) > .18 ? (delta > 0 ? 'DNa · 우 조향' : 'DNa · 좌 조향') : mean > .08 ? 'DNp09 · 전진' : '정지';
    document.getElementById('brain-action').textContent = `${action}  L ${c.gain_left.toFixed(2)} · R ${c.gain_right.toFixed(2)}`;
    document.getElementById('brain-confidence').style.width = `${Math.min(100,c.confidence*100)}%`;
  }

  _updateTargetVisual() {
    const sensed = this.senses?.target;
    const target = sensed ? [sensed.position.x, sensed.position.y] :
      this.waypoints[Math.min(this.gateIndex, this.waypoints.length - 1)];
    if (!target || !this.targetBeacon) return;
    this.targetBeacon.position.set(target[0], target[1], sensed ? sensed.position.z + .65 : .55);
    this.targetHalo.position.set(target[0], target[1], .04);
    const color = sensed ? STIMULUS_DEFS[sensed.kind].color : 0x58f2b0;
    this.targetBeacon.material.color.setHex(color);
    this.targetHalo.material.color.setHex(color);
  }

  _sendPhysicsObservation(now) {
    if (this.phase !== 'running' || now - this._lastSenseWall < .12) return;
    this._lastSenseWall = now;
    const d=this.data,b=this.bodyId,x=d.xpos[3*b],y=d.xpos[3*b+1],z=d.xpos[3*b+2];
    const xm=d.xmat,heading=Math.atan2(xm[9*b+3],xm[9*b+0]);
    const gateTarget=this.waypoints[Math.min(this.gateIndex,this.waypoints.length-1)];
    const gateDx=gateTarget[0]-x,gateDy=gateTarget[1]-y,gateDist=Math.hypot(gateDx,gateDy);
    let gateReward=0;
    while(this.gateIndex<this.waypoints.length){const gate=this.waypoints[this.gateIndex];if(x>=gate[0]-.35&&x<=gate[0]+1.2&&Math.abs(y-gate[1])<2.25){this.gateIndex++;gateReward+=1;this._updateTargetVisual()}else break}
    const dist=this.senses.target ? this.senses.targetDistance : gateDist;
    const rawBearing=this.senses.target ? this.senses.targetBearing : Math.atan2(gateDy,gateDx)-heading;
    const bearing=Math.atan2(Math.sin(rawBearing),Math.cos(rawBearing));
    const speed=Math.hypot(d.qvel[0]||0,d.qvel[1]||0);
    const progress=this._lastTargetDistance===null?0:Math.max(-.25,Math.min(.25,this._lastTargetDistance-dist));
    this._lastTargetDistance=dist;
    const touch=Math.max(this.senses.touch,Math.max(0,Math.min(1,(d.ncon-6)/8)));
    if(touch>.45)this.collisionCount++;
    document.getElementById('physics-pose').textContent=`${x.toFixed(2)} · ${y.toFixed(2)} · ${z.toFixed(2)}`;
    document.getElementById('physics-contact').textContent=`${this.gateIndex} / ${d.ncon}`;
    window.__nmfDiagnostics={phase:this.phase,level:this.level,x,y,z,heading,speed,gate:this.gateIndex,
      contacts:d.ncon,targetDistance:dist,bearing,mobility:this.mobilityMode,flightThrottle:this.flightThrottle,
      senses:{...this.senses,target:this.senses.target?.kind||null},brain:{...this.brain}};
    if(this.level!=='brain')return;
    this._brainSend({
      type:'physics_observation',x,y,z,heading,speed,
      odor:this.senses.odor,visual:Math.max(this.brainThreat,.04,this.senses.visual),touch,taste:this.senses.taste,
      temperature:this.senses.temperature,humidity:this.senses.humidity,wind:this.senses.wind,
      target_distance:dist,target_bearing:bearing,reward:progress*.18+gateReward,
      gate:this.gateIndex,collisions:this.collisionCount,learning:this.brainLearning,
      threat:Math.max(this.brainThreat,this.senses.threat),stimulus_source:this.senses.target?.kind||'track',
    });
    this.brainThreat=Math.max(0,this.brainThreat-.04);
  }

  // --- scene ---------------------------------------------------------------
  _buildScene() {
    const stage = document.getElementById('stage');
    THREE.Object3D.DEFAULT_UP.set(0, 0, 1);            // MuJoCo is z-up
    this.renderer = new THREE.WebGLRenderer({ antialias: true });
    this.renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
    stage.appendChild(this.renderer.domElement);

    this.scene = new THREE.Scene();
    this.scene.fog = new THREE.Fog(0x14161b, 45, 140);
    this.camera = new THREE.PerspectiveCamera(50, 1, 0.05, 500);
    this.camera.up.set(0, 0, 1);

    this.scene.add(new THREE.AmbientLight(0xffffff, 0.85));
    const key = new THREE.DirectionalLight(0xffffff, 1.1); key.position.set(8, -10, 16);
    const fill = new THREE.DirectionalLight(0xffffff, 0.4); fill.position.set(-8, 6, 6);
    this.scene.add(key, fill);

    this._buildGround();
    this._buildFinish();
    this.meshGroup = buildMeshes(this.model, this.meta);
    this.scene.add(this.meshGroup);
    this.wingMeshes = (this.meshGroup.userData.items || [])
      .map((item) => item.mesh)
      .filter((mesh) => /(^|\/)[lr]_wing$/i.test(mesh.userData.name || '') || /[lr]_wing/i.test(mesh.userData.name || ''));
    this.haltereMeshes = (this.meshGroup.userData.items || [])
      .map((item) => item.mesh).filter((mesh) => /[lr]_haltere/i.test(mesh.userData.name || ''));
    this.targetBeacon = new THREE.Mesh(
      new THREE.SphereGeometry(.34, 24, 16),
      new THREE.MeshStandardMaterial({ color:0x58f2b0, emissive:0x1b704e, emissiveIntensity:1.4 }));
    this.targetHalo = new THREE.Mesh(
      new THREE.TorusGeometry(.62,.045,12,42),
      new THREE.MeshBasicMaterial({ color:0x58f2b0, transparent:true, opacity:.75 }));
    this.targetHalo.rotation.x = Math.PI / 2;
    this.scene.add(this.targetBeacon, this.targetHalo);
    this._buildStimulusWorld();
    this._updateTargetVisual();

    addEventListener('resize', () => this._resize());
    this._resize();
  }

  _buildGround() {
    // A big checkerboard plane (the model's own ground plane isn't rendered),
    // tinted per level: the greyscale checker texture is multiplied by the
    // material colour, so the squares come out as two shades of the level colour.
    const geo = new THREE.PlaneGeometry(400, 400);
    const checker = makeCheckerTexture();
    checker.repeat.set(12, 12);          // ~4 mm squares across the 400 mm plane
    this.groundMat = new THREE.MeshStandardMaterial({ map: checker, color: 0x8ace00, roughness: 0.96 });
    const ground = new THREE.Mesh(geo, this.groundMat);
    ground.position.z = -0.02;
    this.scene.add(ground);
    const railMat = new THREE.MeshBasicMaterial({color:0x58f2b0,transparent:true,opacity:.5});
    const {minX,maxX,minY,maxY}=this.arenaBounds;
    const longRail=new THREE.BoxGeometry(maxX-minX,.09,.08),sideRail=new THREE.BoxGeometry(.09,maxY-minY,.08);
    for(const yy of [minY,maxY]){const rail=new THREE.Mesh(longRail,railMat);rail.position.set((minX+maxX)/2,yy,.04);this.scene.add(rail)}
    for(const xx of [minX,maxX]){const rail=new THREE.Mesh(sideRail,railMat);rail.position.set(xx,(minY+maxY)/2,.04);this.scene.add(rail)}
    this._applyLevelColor();
  }

  _applyLevelColor() {
    const c = this.level === 'brain' ? [0.11,0.25,0.19,1] : (this.meta.arena.level_ground_colors[this.level] || [0.5, 0.5, 0.5, 1]);
    this.groundMat.color.setRGB(c[0], c[1], c[2], THREE.SRGBColorSpace);
  }

  _buildFinish() {
    // A translucent white banner spanning the finish line, plus a bright strip
    // on the ground, so the goal reads clearly from the chase camera. Built from
    // thin boxes (no fiddly plane rotations): the line runs along y at fixed x.
    const [[x, y1], [, y2]] = this.finish;
    const w = Math.abs(y2 - y1) + 0.2, yc = (y1 + y2) / 2, h = 5;
    const banner = new THREE.Mesh(
      new THREE.BoxGeometry(0.05, w, h),
      new THREE.MeshBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0.16 }));
    banner.position.set(x, yc, h / 2);
    const strip = new THREE.Mesh(
      new THREE.BoxGeometry(0.8, w, 0.02),
      new THREE.MeshBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0.65 }));
    strip.position.set(x, yc, 0.012);
    this.scene.add(banner, strip);
  }

  // --- placeable stimulus world -------------------------------------------
  _makeObjectLabel(text, color) {
    const canvas=document.createElement('canvas'); canvas.width=256; canvas.height=64;
    const ctx=canvas.getContext('2d');
    ctx.fillStyle='rgba(4,10,13,.82)'; ctx.beginPath(); ctx.roundRect(2,2,252,60,15); ctx.fill();
    ctx.strokeStyle=`#${color.toString(16).padStart(6,'0')}`; ctx.lineWidth=3; ctx.stroke();
    ctx.font='700 24px system-ui'; ctx.textAlign='center'; ctx.textBaseline='middle'; ctx.fillStyle='#fff'; ctx.fillText(text,128,33);
    const sprite=new THREE.Sprite(new THREE.SpriteMaterial({map:new THREE.CanvasTexture(canvas),transparent:true,depthTest:false}));
    sprite.scale.set(3.2,.8,1); sprite.renderOrder=12; return sprite;
  }

  _buildStimulusWorld() {
    this.stimulusGroup=new THREE.Group(); this.stimulusGroup.name='external-stimulus-world'; this.scene.add(this.stimulusGroup);
    this._raycaster=new THREE.Raycaster(); this._groundPlane=new THREE.Plane(new THREE.Vector3(0,0,1),0);
    this.renderer.domElement.addEventListener('click',(event)=>{
      if(!this.placementKind)return;
      const rect=this.renderer.domElement.getBoundingClientRect();
      const pointer=new THREE.Vector2((event.clientX-rect.left)/rect.width*2-1,-((event.clientY-rect.top)/rect.height)*2+1);
      this._raycaster.setFromCamera(pointer,this.camera);
      const point=new THREE.Vector3();
      if(this._raycaster.ray.intersectPlane(this._groundPlane,point)){
        point.x=Math.max(this.arenaBounds.minX+.6,Math.min(this.arenaBounds.maxX-.6,point.x));
        point.y=Math.max(this.arenaBounds.minY+.6,Math.min(this.arenaBounds.maxY-.6,point.y));
        this.addStimulus(this.placementKind,point);
        this.placementKind=null; this._renderPlacementState();
      }
    });
    this._populateDemoWorld();
  }

  _stimulusMesh(kind) {
    const def=STIMULUS_DEFS[kind], group=new THREE.Group(); let core;
    if(kind==='food'){
      core=new THREE.Mesh(new THREE.SphereGeometry(.42,22,14),new THREE.MeshStandardMaterial({color:def.color,roughness:.55,emissive:0x6a3f05,emissiveIntensity:.35}));
      core.scale.z=.72; core.position.z=.34;
      const stem=new THREE.Mesh(new THREE.CylinderGeometry(.035,.045,.35,8),new THREE.MeshStandardMaterial({color:0x568c48})); stem.position.z=.7; group.add(core,stem);
    }else if(kind==='light'){
      core=new THREE.Mesh(new THREE.SphereGeometry(.38,20,14),new THREE.MeshStandardMaterial({color:def.color,emissive:def.color,emissiveIntensity:2.6})); core.position.z=2.1;
      const ring=new THREE.Mesh(new THREE.TorusGeometry(.65,.035,10,36),new THREE.MeshBasicMaterial({color:def.color,transparent:true,opacity:.75})); ring.position.z=2.1; group.add(core,ring);
      const lamp=new THREE.PointLight(def.color,5,11,2); lamp.position.z=2.1; group.add(lamp);
    }else if(kind==='threat'){
      core=new THREE.Mesh(new THREE.SphereGeometry(.62,24,16),new THREE.MeshStandardMaterial({color:0x28131a,emissive:def.color,emissiveIntensity:1.45,roughness:.3})); core.position.z=2.0;
      const pupil=new THREE.Mesh(new THREE.SphereGeometry(.22,16,12),new THREE.MeshBasicMaterial({color:0xfff2e9})); pupil.position.set(-.56,0,2.05);
      const iris=new THREE.Mesh(new THREE.SphereGeometry(.105,14,10),new THREE.MeshBasicMaterial({color:0x080304})); iris.position.set(-.75,0,2.05); group.add(core,pupil,iris);
    }else if(kind==='wind'){
      core=new THREE.Mesh(new THREE.TorusGeometry(.58,.12,12,28),new THREE.MeshStandardMaterial({color:def.color,emissive:0x176e67,emissiveIntensity:.8})); core.rotation.y=Math.PI/2; core.position.z=1.2; group.add(core);
      for(let i=0;i<3;i++){
        const stream=new THREE.Mesh(new THREE.CylinderGeometry(.022,.055,2.4,7),new THREE.MeshBasicMaterial({color:def.color,transparent:true,opacity:.45}));
        stream.rotation.z=Math.PI/2; stream.position.set(1.5,(i-1)*.36,1.2); stream.userData.stream=true; group.add(stream);
      }
    }else if(kind==='heat'){
      core=new THREE.Mesh(new THREE.CylinderGeometry(1.25,.8,.12,36),new THREE.MeshBasicMaterial({color:def.color,transparent:true,opacity:.26,depthWrite:false})); core.position.z=.07; group.add(core);
      for(let i=0;i<3;i++){const ring=new THREE.Mesh(new THREE.TorusGeometry(.5+i*.28,.035,8,32),new THREE.MeshBasicMaterial({color:def.color,transparent:true,opacity:.55-i*.1}));ring.position.z=.18+i*.28;group.add(ring)}
    }else{
      core=new THREE.Mesh(new THREE.BoxGeometry(1.5,1.5,2.1),new THREE.MeshStandardMaterial({color:def.color,roughness:.72,metalness:.1})); core.position.z=1.05;
      const edge=new THREE.LineSegments(new THREE.EdgesGeometry(core.geometry),new THREE.LineBasicMaterial({color:0xffffff,transparent:true,opacity:.45})); edge.position.copy(core.position); group.add(core,edge);
    }
    const label=this._makeObjectLabel(def.label,def.color); label.position.set(0,0,kind==='heat'?1.65:3.0); group.add(label);
    return group;
  }

  addStimulus(kind,position) {
    if(!STIMULUS_DEFS[kind]||this.stimuli.length>=24)return;
    const group=this._stimulusMesh(kind),p=position.clone ? position.clone() : new THREE.Vector3(...position);
    group.position.set(p.x,p.y,0);
    const item={id:`${kind}-${Date.now()}-${Math.random()}`,kind,group,position:group.position,
      phase:Math.random()*TAU,windAngle:(Math.random()-.5)*.7};
    group.rotation.z=item.windAngle; group.userData.stimulus=item;
    this.stimuli.push(item); this.stimulusGroup.add(group); this._updateStimulusCount();
  }

  _populateDemoWorld() {
    this.clearStimuli();
    [['food',[8,2,0]],['light',[15,-3,0]],['obstacle',[11,.1,0]],['wind',[21,2.5,0]],['threat',[29,-5.2,0]],['heat',[38,3.5,0]]]
      .forEach(([kind,p])=>this.addStimulus(kind,new THREE.Vector3(...p)));
  }

  clearStimuli() {
    if(!this.stimulusGroup)return;
    for(const item of this.stimuli){
      item.group.traverse((o)=>{o.geometry?.dispose?.();if(o.material){o.material.map?.dispose?.();o.material.dispose?.()}});
      this.stimulusGroup.remove(item.group);
    }
    this.stimuli=[]; this._updateStimulusCount();
  }

  _shuffleStimuli() {
    for(const item of this.stimuli){item.group.position.x=5+Math.random()*43;item.group.position.y=-7+Math.random()*14;item.windAngle=(Math.random()-.5)*1.4;item.group.rotation.z=item.windAngle}
  }

  _updateStimulusCount(){const el=document.getElementById('stimulus-count');if(el)el.textContent=this.stimuli.length}
  _renderPlacementState(){
    document.querySelectorAll('[data-object]').forEach((button)=>button.classList.toggle('armed',button.dataset.object===this.placementKind));
    document.getElementById('place-hint').textContent=this.placementKind ?
      `${STIMULUS_DEFS[this.placementKind].label} 선택됨 — 3D 바닥을 클릭하세요.` : '오브젝트를 고른 뒤 3D 바닥을 클릭해 놓으세요.';
  }

  _updateStimulusWorld(now) {
    const d=this.data,b=this.bodyId,x=d.xpos[3*b],y=d.xpos[3*b+1],z=d.xpos[3*b+2];
    const heading=Math.atan2(d.xmat[9*b+3],d.xmat[9*b]);
    let odor=0,visual=0,touch=0,taste=0,temperature=0,threat=0,wind=0,target=null,targetScore=-1,targetDistance=0,targetBearing=0;
    let strongest=null,strongestValue=0;
    for(const item of this.stimuli){
      const def=STIMULUS_DEFS[item.kind],g=item.group;
      if(item.kind==='threat'){g.position.x+=Math.sin(now*.7+item.phase)*.002;g.position.y+=Math.cos(now*.55+item.phase)*.002;g.scale.setScalar(1+.12*Math.sin(now*4+item.phase))}
      if(item.kind==='light')g.rotation.z=now*.8;
      if(item.kind==='heat')g.children.filter((o)=>o.geometry?.type==='TorusGeometry').forEach((o,i)=>{o.position.z=.18+i*.28+.12*Math.sin(now*2+i);o.rotation.z=now*(.4+i*.1)});
      if(item.kind==='wind')g.children.filter((o)=>o.userData.stream).forEach((o,i)=>{o.material.opacity=.2+.35*(.5+.5*Math.sin(now*5+i*2))});
      const dx=g.position.x-x,dy=g.position.y-y,dist=Math.max(.05,Math.hypot(dx,dy));
      const bearing=Math.atan2(Math.sin(Math.atan2(dy,dx)-heading),Math.cos(Math.atan2(dy,dx)-heading));
      const proximity=Math.max(0,1-dist/def.range),front=.22+.78*Math.max(0,Math.cos(bearing));
      let value=0;
      if(item.kind==='food'){value=Math.exp(-dist/5.5);odor=Math.max(odor,value);if(dist<.72)taste=1;if(value>targetScore){target=item;targetScore=value;targetDistance=dist;targetBearing=bearing}}
      if(item.kind==='light'){value=proximity*front;visual=Math.max(visual,value);if(value*.72>targetScore){target=item;targetScore=value*.72;targetDistance=dist;targetBearing=bearing}}
      if(item.kind==='threat'){value=proximity*(.65+.35*Math.sin(now*4+item.phase));threat=Math.max(threat,value);visual=Math.max(visual,value*.9);if(value>.28&&value+1>targetScore){target=item;targetScore=value+1;targetDistance=dist;targetBearing=Math.atan2(Math.sin(bearing+Math.PI),Math.cos(bearing+Math.PI))}}
      if(item.kind==='wind'){value=proximity;wind=Math.max(wind,value);touch=Math.max(touch,value*.7)}
      if(item.kind==='heat'){value=proximity;temperature=Math.max(temperature,value);if(value>.38&&value+.8>targetScore){target=item;targetScore=value+.8;targetDistance=dist;targetBearing=Math.atan2(Math.sin(bearing+Math.PI),Math.cos(bearing+Math.PI))}}
      if(item.kind==='obstacle'){value=Math.max(0,1-Math.max(0,dist-def.radius)/2.2)*front;touch=Math.max(touch,Math.max(0,1-(dist-def.radius)/.45));visual=Math.max(visual,value*.55);if(value>.55&&value+.6>targetScore){target=item;targetScore=value+.6;targetDistance=dist;targetBearing=Math.atan2(Math.sin(bearing+Math.PI),Math.cos(bearing+Math.PI))}}
      if(value>strongestValue){strongestValue=value;strongest=item}
    }
    const bounds=this.arenaBounds,edgeDistance=Math.min(x-bounds.minX,bounds.maxX-x,y-bounds.minY,bounds.maxY-y);
    if(edgeDistance<2.2){const safeX=Math.max(bounds.minX+3,Math.min(bounds.maxX-3,x)),safeY=Math.max(bounds.minY+3,Math.min(bounds.maxY-3,y)),dx=safeX-x,dy=safeY-y,bearing=Math.atan2(Math.sin(Math.atan2(dy,dx)-heading),Math.cos(Math.atan2(dy,dx)-heading)),value=Math.max(.25,1-edgeDistance/2.2);const boundary={kind:'boundary',position:new THREE.Vector3(safeX,safeY,0)};target=boundary;targetScore=3;targetDistance=Math.max(.05,Math.hypot(dx,dy));targetBearing=bearing;touch=Math.max(touch,value*.75);strongest=boundary;strongestValue=Math.max(strongestValue,value)}
    const reaction=strongestValue<.05?'감지되는 자극이 거의 없습니다':
      strongest.kind==='boundary'?`경기장 경계 감지 → 안쪽으로 복귀`:
      strongest.kind==='threat'?`👁 포식자 감지 → ${this.mobilityMode==='walk'?'MDN 후진':'도피 비행'} 준비`:
      strongest.kind==='heat'?`🔥 열 감지 → 반대 방향으로 회피`:
      strongest.kind==='wind'?`🌬 바람 감지 → 몸이 밀리고 기계감각 활성`:
      strongest.kind==='obstacle'?`⬛ 장애물 접근 → 접촉 회피`:
      strongest.kind==='food'?`🍎 냄새를 따라 접근${taste?' → 먹이 접촉!':''}`:`💡 빛을 보고 접근`;
    this.senses={odor,visual,touch,taste,temperature,humidity:0,threat,wind,targetDistance,targetBearing,target,reaction,strongest:strongest?.kind||null};
    this._updateTargetVisual(); this._updateSenseHud(now);
  }

  _updateSenseHud(now){
    if(now-this._lastSenseHud<.08)return;this._lastSenseHud=now;
    document.getElementById('reaction').textContent=this.senses.reaction;
    for(const key of ['visual','odor','touch','taste','temperature']){
      const value=Math.max(0,Math.min(1,this.senses[key]||0));
      document.getElementById(`sense-${key}`).style.width=`${value*100}%`;
      document.getElementById(`sense-${key}-v`).textContent=value.toFixed(2);
    }
  }

  _resize() {
    const w = innerWidth, h = innerHeight;
    this.renderer.setSize(w, h, false);
    this.camera.aspect = w / Math.max(1, h);
    this.camera.updateProjectionMatrix();
  }

  // --- camera follow (ported from Game.update_camera_to_follow_fly) --------
  _updateCamera(init) {
    const { height, distance } = this.meta.camera;
    // The desktop applies its 0.995 yaw smoothing per *physics step* (thousands
    // per second); we update once per animation frame, so use a per-frame factor
    // that chases the fly's heading smoothly without lagging.
    const smoothing = 0.85;
    const d = this.data, b = this.bodyId;
    const fx = d.xpos[3 * b], fy = d.xpos[3 * b + 1], fz=d.xpos[3*b+2];
    const xm = d.xmat;                       // body x-axis (forward) in world
    const newYaw = Math.atan2(xm[9 * b + 3], xm[9 * b + 0]);
    if (init || this._yaw === undefined) this._yaw = newYaw;
    this._yaw = Math.atan2(
      smoothing * Math.sin(this._yaw) + (1 - smoothing) * Math.sin(newYaw),
      smoothing * Math.cos(this._yaw) + (1 - smoothing) * Math.cos(newYaw));
    const cy = Math.cos(this._yaw), sy = Math.sin(this._yaw);
    const cameraZ=Math.max(height,fz+height*.62);
    this.camera.position.set(fx - cy * distance, fy - sy * distance, cameraZ);
    this.camera.lookAt(fx + cy * 1.5, fy + sy * 1.5, Math.max(.4,fz));
  }

  _setMobilityMode(mode){
    if(!['walk','auto','fly'].includes(mode))return;
    this.mobilityMode=mode;
    document.querySelectorAll('#mobility button').forEach((button)=>button.classList.toggle('active',button.dataset.mode===mode));
  }

  _applyExternalForces() {
    const force=this.data.xfrc_applied,b=this.bodyId,base=6*b;
    for(let i=0;i<6;i++)force[base+i]=0;
    const d=this.data,x=d.xpos[3*b],y=d.xpos[3*b+1],z=d.xpos[3*b+2],weight=this.bodyMass*this.gravity;
    // Wind and the near-field of the visible obstacle are physical, not only pixels.
    for(const item of this.stimuli){
      const dx=x-item.group.position.x,dy=y-item.group.position.y,dist=Math.hypot(dx,dy);
      if(item.kind==='wind'){
        const strength=Math.max(0,1-dist/STIMULUS_DEFS.wind.range);
        force[base]+=Math.cos(item.windAngle)*weight*.14*strength;
        force[base+1]+=Math.sin(item.windAngle)*weight*.14*strength;
      }else if(item.kind==='obstacle'&&dist<STIMULUS_DEFS.obstacle.radius+.35){
        const repel=(STIMULUS_DEFS.obstacle.radius+.35-dist)*weight*.8;
        force[base]+=dx/Math.max(.05,dist)*repel; force[base+1]+=dy/Math.max(.05,dist)*repel;
      }
    }
    this._applyArenaBoundary(force,base,weight,x,y);
    this._applyAltitudeCeiling(force,base,weight,z);
    this._applyRightingForces(force,base,weight,z);
    const autoFlight=Math.max(this.senses.threat,this.brainThreat)>.18||this.senses.temperature>.62;
    const wantsFlight=this.mobilityMode==='fly'||(this.mobilityMode==='auto'&&autoFlight);
    const rate=this.dt*(wantsFlight?7:4);
    this.flightThrottle+=Math.sign((wantsFlight?1:0)-this.flightThrottle)*Math.min(rate,Math.abs((wantsFlight?1:0)-this.flightThrottle));
    this.flightReason=this.mobilityMode==='fly'?'강제 비행':Math.max(this.senses.threat,this.brainThreat)>.18?'포식자 회피':this.senses.temperature>.62?'열 회피':'';
    if(this.flightThrottle>.02){
      const vz=Number(d.qvel[2]||0),altError=this.flightTargetZ-z;
      // A gentle altitude controller. The previous proportional multiplier
      // effectively commanded fractions of 1 g per millimetre and launched the
      // tiny body violently from the contact surface.
      const verticalAccel=Math.max(-1700,Math.min(1300,82*altError-15*vz));
      const liftScale=Math.max(.82,Math.min(1.14,1+verticalAccel/this.gravity));
      force[base+2]+=weight*liftScale*this.flightThrottle;
      const motorLeft=this.level==='brain'?(this.brain.gain_left||0):(this.input.gainL||0);
      const motorRight=this.level==='brain'?(this.brain.gain_right||0):(this.input.gainR||0);
      const motorMean=Math.max(0,(motorLeft+motorRight)*.5);
      const steerRaw=Math.max(-1,Math.min(1,motorRight-motorLeft+Math.sin(this.senses.targetBearing||0)*.62));
      this._stabilizeFlightOrientation(steerRaw,motorMean);
      const heading=this._flightHeading??Math.atan2(d.xmat[9*b+3],d.xmat[9*b]);
      const thrust=weight*(.012+.016*motorMean)*this.flightThrottle;
      force[base]+=Math.cos(heading)*thrust-this.bodyMass*18*(d.qvel[0]||0);
      force[base+1]+=Math.sin(heading)*thrust-this.bodyMass*18*(d.qvel[1]||0);
      // Bound translational velocity as a numerical safety rail for sudden
      // contacts and overlapping wind fields.
      const planar=Math.hypot(d.qvel[this.rootDofAdr]||0,d.qvel[this.rootDofAdr+1]||0);
      if(planar>12){const s=12/planar;d.qvel[this.rootDofAdr]*=s;d.qvel[this.rootDofAdr+1]*=s}
      d.qvel[this.rootDofAdr+2]=Math.max(-7,Math.min(7,d.qvel[this.rootDofAdr+2]||0));
      for(const adhesionIndex of this.meta.adhesion) this.data.ctrl[adhesionIndex]=0;
    }else{this._flightHeading=null;this._flightSteer=0;this._flightBank=0}
  }

  _applyArenaBoundary(force,base,weight,x,y){
    const d=this.data,qa=this.rootQposAdr,da=this.rootDofAdr,b=this.arenaBounds,margin=2.0;
    if(x<b.minX+margin)force[base]+=weight*.22*(b.minX+margin-x);
    if(x>b.maxX-margin)force[base]-=weight*.22*(x-(b.maxX-margin));
    if(y<b.minY+margin)force[base+1]+=weight*.30*(b.minY+margin-y);
    if(y>b.maxY-margin)force[base+1]-=weight*.30*(y-(b.maxY-margin));
    if(x<b.minX-.5){d.qpos[qa]=b.minX-.5;d.qvel[da]=Math.abs(d.qvel[da]||0)*.2}
    if(x>b.maxX+.5){d.qpos[qa]=b.maxX+.5;d.qvel[da]=-Math.abs(d.qvel[da]||0)*.2}
    if(y<b.minY-.5){d.qpos[qa+1]=b.minY-.5;d.qvel[da+1]=Math.abs(d.qvel[da+1]||0)*.2}
    if(y>b.maxY+.5){d.qpos[qa+1]=b.maxY+.5;d.qvel[da+1]=-Math.abs(d.qvel[da+1]||0)*.2}
  }

  _applyAltitudeCeiling(force,base,weight,z){
    const d=this.data,qa=this.rootQposAdr,da=this.rootDofAdr;
    if(z>10)force[base+2]-=weight*.55*(z-10);
    if(z>14){d.qpos[qa+2]=14;d.qvel[da+2]=-Math.abs(d.qvel[da+2]||0)*.15}
  }

  _stabilizeFlightOrientation(steer,motorMean){
    const d=this.data,b=this.bodyId,qa=this.rootQposAdr,da=this.rootDofAdr;
    if(this._flightHeading===null)this._flightHeading=Math.atan2(d.xmat[9*b+3],d.xmat[9*b]);
    const follow=Math.min(1,this.dt*7);
    this._flightSteer+=(steer-this._flightSteer)*follow;
    this._flightHeading=Math.atan2(Math.sin(this._flightHeading+this._flightSteer*4.2*this.dt),Math.cos(this._flightHeading+this._flightSteer*4.2*this.dt));
    this._flightBank+=(-this._flightSteer*.16-this._flightBank)*Math.min(1,this.dt*9);
    const roll=this._flightBank,pitch=-Math.min(.075,.035+.025*motorMean),yaw=this._flightHeading;
    const cr=Math.cos(roll/2),sr=Math.sin(roll/2),cp=Math.cos(pitch/2),sp=Math.sin(pitch/2),cy=Math.cos(yaw/2),sy=Math.sin(yaw/2);
    d.qpos[qa+3]=cr*cp*cy+sr*sp*sy;
    d.qpos[qa+4]=sr*cp*cy-cr*sp*sy;
    d.qpos[qa+5]=cr*sp*cy+sr*cp*sy;
    d.qpos[qa+6]=cr*cp*sy-sr*sp*cy;
    // Translation remains fully dynamic; rotation is a bounded attitude servo
    // so contact impulses cannot accumulate into an uncontrolled tumble.
    d.qvel[da+3]=0;d.qvel[da+4]=0;d.qvel[da+5]=0;
  }

  _applyRightingForces(force,base,weight,z){
    const d=this.data,b=this.bodyId,xm=d.xmat;
    const upX=xm[9*b+2],upY=xm[9*b+5],upZ=xm[9*b+8];
    const onGround=z<3.6;
    if(onGround&&upZ<.55)this._invertedFor+=this.dt;
    else if(upZ>.78){this._invertedFor=0;this._righting=false}
    if(this._manualRighting>0)this._manualRighting=Math.max(0,this._manualRighting-this.dt);
    const active=onGround&&(this._invertedFor>.055||this._manualRighting>0)&&this.flightThrottle<.35;
    this._righting=active;
    if(!active)return;
    // First try a physical recovery: release sticky feet, kick all six legs,
    // lift the thorax slightly and roll the dorsal axis toward world-up.
    for(const adhesionIndex of this.meta.adhesion)d.ctrl[adhesionIndex]=0;
    for(let leg=0;leg<6;leg++)for(let dof=0;dof<3;dof++){
      const ci=this.meta.ctrl_index_by_leg_dof[leg][dof];
      d.ctrl[ci]+=Math.sin(this.simTime*95+leg*1.7+dof*.8)*(.11+dof*.025);
    }
    force[base+2]+=weight*.18;
    let axisX=upY,axisY=-upX;
    if(Math.hypot(axisX,axisY)<.08){axisX=(Math.sin(this.simTime*31)>=0?1:-1);axisY=.18}
    force[base+3]+=this.bodyMass*(780*axisX-38*(d.qvel[this.rootDofAdr+3]||0));
    force[base+4]+=this.bodyMass*(780*axisY-38*(d.qvel[this.rootDofAdr+4]||0));
    // If geometry has wedged the animal after a physical attempt, perform the
    // simulator's assisted-righting equivalent while preserving x/y and yaw.
    if(this._invertedFor>.12||this._manualRighting>.18)this._assistUpright();
  }

  _assistUpright(){
    const d=this.data,b=this.bodyId,qa=this.rootQposAdr,da=this.rootDofAdr;
    const yaw=Math.atan2(d.xmat[9*b+3],d.xmat[9*b]);
    d.qpos[qa+2]=Math.max(2.15,Math.min(2.4,d.qpos[qa+2]));
    d.qpos[qa+3]=Math.cos(yaw/2); d.qpos[qa+4]=0; d.qpos[qa+5]=0; d.qpos[qa+6]=Math.sin(yaw/2);
    for(let i=0;i<6;i++)d.qvel[da+i]=0;
    this._invertedFor=0;this._manualRighting=0;this._righting=false;
    this.mj.mj_forward(this.model,d);
  }

  _flipForTest(){
    if(this.phase!=='running')return;
    const d=this.data,b=this.bodyId,qa=this.rootQposAdr,da=this.rootDofAdr,yaw=Math.atan2(d.xmat[9*b+3],d.xmat[9*b]),cy=Math.cos(yaw/2),sy=Math.sin(yaw/2);
    this._setMobilityMode('walk');this.flightThrottle=0;
    d.qpos[qa+2]=Math.max(2.3,d.qpos[qa+2]);d.qpos[qa+3]=0;d.qpos[qa+4]=cy;d.qpos[qa+5]=sy;d.qpos[qa+6]=0;
    for(let i=0;i<6;i++)d.qvel[da+i]=0;
    this._invertedFor=.08;this._manualRighting=.32;this.mj.mj_forward(this.model,d);
  }

  _rotateMeshAtHinge(mesh,rootRotation,localRotation){
    const body=mesh.userData.bodyId,d=this.data;
    const pivot=new THREE.Vector3(d.xpos[3*body],d.xpos[3*body+1],d.xpos[3*body+2]);
    const worldRotation=rootRotation.clone().multiply(localRotation).multiply(rootRotation.clone().invert());
    const around=new THREE.Matrix4().makeTranslation(pivot.x,pivot.y,pivot.z)
      .multiply(worldRotation).multiply(new THREE.Matrix4().makeTranslation(-pivot.x,-pivot.y,-pivot.z));
    mesh.matrix.premultiply(around);
  }

  _animateFlight(now){
    // Display the biological ~200 Hz wingbeat in 0.1x visual slow motion. Each
    // wing rotates around its thorax attachment (body origin), not mesh centre.
    const phase=now*TAU*8.5,amp=this.flightThrottle*this.flightThrottle*(3-2*this.flightThrottle);
    const stroke=amp*(.92*Math.sin(phase)+.10*Math.sin(phase*3));
    const pitchFlip=.40*Math.tanh(3.2*Math.cos(phase));
    const elevation=.13+.09*Math.sin(phase*2-.35);
    const xm=this.data.xmat,b=this.bodyId;
    const rootRotation=new THREE.Matrix4().set(
      xm[9*b],xm[9*b+1],xm[9*b+2],0,
      xm[9*b+3],xm[9*b+4],xm[9*b+5],0,
      xm[9*b+6],xm[9*b+7],xm[9*b+8],0,
      0,0,0,1);
    for(const mesh of this.wingMeshes){
      const side=/l_wing/i.test(mesh.userData.name||'')?1:-1;
      const local=new THREE.Matrix4().makeRotationFromEuler(new THREE.Euler(
        side*elevation*amp,side*pitchFlip*amp,stroke,'XYZ'));
      this._rotateMeshAtHinge(mesh,rootRotation,local);
    }
    for(const mesh of this.haltereMeshes){
      const side=/l_haltere/i.test(mesh.userData.name||'')?1:-1;
      const local=new THREE.Matrix4().makeRotationX(side*-.42*Math.sin(phase+Math.PI)*amp);
      this._rotateMeshAtHinge(mesh,rootRotation,local);
    }
    const z=this.data.xpos[3*this.bodyId+2],airborne=z>1.8;
    const state=this._righting?'RECOVER · 다리로 몸 돌리는 중':this.flightThrottle>.12 ? `FLY ${Math.round(this.flightThrottle*100)}% · ${airborne?'air '+z.toFixed(1)+' mm':'takeoff'}` : airborne?'LAND · 착륙 중':`WALK · ground`;
    document.getElementById('flight-state').textContent=this.flightReason?`${state} · ${this.flightReason}`:state;
  }

  // --- game state ----------------------------------------------------------
  _resetSim() {
    this.mj.mj_resetDataKeyframe(this.model, this.data, 0);
    this.controller.reset();
    this.input.resetGains();
    this.mj.mj_forward(this.model, this.data);
    this.simTime = 0;
    this.gateIndex = 0;
    this.collisionCount = 0;
    this._lastTargetDistance = null;
    this.brain = { gain_left:0, gain_right:0, confidence:0 };
    this.flightThrottle=0;this.flightReason='';this._flightHeading=null;this._flightSteer=0;this._flightBank=0;
    this._invertedFor=0;this._righting=false;this._manualRighting=0;
    this._updateTargetVisual();
    document.getElementById('timer').textContent = '0.00';
    const b = this.bodyId;
    this.prevXY = [this.data.xpos[3 * b], this.data.xpos[3 * b + 1]];
    this._updateCamera(true);
  }

  setLevel(level) {
    if (!LEVELS[level]) return;
    this._brainSend({ type:'run', value:false });
    this.level = level;
    document.querySelectorAll('#levels button').forEach((bt) =>
      bt.classList.toggle('active', bt.dataset.level === level));
    document.getElementById('level-name').textContent = LEVELS[level].name;
    document.querySelector('#level .chip').style.background = LEVELS[level].chip;
    this._renderHelp();
    this._applyLevelColor();
    this._renderBest();
    this.restart();
  }

  restart() {
    this._brainSend({ type:'run', value:false });
    this._resetSim();
    this._showReady();
  }

  _showReady() {
    this.phase = 'ready';
    overlayEl.classList.remove('hidden');
    overlayEl.innerHTML =
      `<h1>${LEVELS[this.level].name}</h1>` +
      `<p>${this._levelBlurb()}</p>` +
      `<button id="go">Start ▶</button>` +
      `<p style="font-size:12px">or press any movement key</p>`;
    document.getElementById('go').onclick = () => this._startCountdown();
    this._announceRunState();
  }

  _levelBlurb() {
    return {
      brain: 'MaleCNS의 감각뉴런에 목표 방향·접촉·고유감각이 입력되고, 전체 166,700개 뉴런을 전파한 뒤 실제 DNp09·MDN·DNa 활동이 6개 다리 CPG를 조절합니다.',
      CPG: 'Steer with W/A/S/D. Central pattern generators coordinate all six legs for you — just point the fly through the gates to the white finish line.',
      tripod: 'Drive the two tripod groups: G/H step the left/right tripod forward, F/J backward. Alternate them to walk.',
      single: 'Drive each of the six legs individually (T G B / Z H N forward). Coordinating all six is hard — that\'s the point!',
    }[this.level];
  }

  _startCountdown() {
    this._resetSim();
    this.phase = 'countdown';
    this._announceRunState();
    let n = 3;
    const tick = () => {
      if (this.phase !== 'countdown') return;
      overlayEl.classList.remove('hidden');
      overlayEl.innerHTML = `<div class="big">${n > 0 ? n : 'GO!'}</div>`;
      if (n < 0) { overlayEl.classList.add('hidden'); this.phase = 'running'; this.simTime = 0; if(this.level==='brain')this._brainSend({type:'run',value:true}); this._announceRunState(); return; }
      n--; setTimeout(tick, n < 0 ? 350 : 700);
    };
    tick();
  }

  // The displayed "control time" is how long the player has been steering in
  // wall-clock terms: sim time runs at PLAYBACK_SPEED, so control = sim / speed.
  _controlTime() { return this.simTime / PLAYBACK_SPEED; }

  _finishRun() {
    this.phase = 'finished';
    this._brainSend({ type:'run', value:false });
    const t = this._controlTime();
    const { board, youIndex } = this._updateBoard(t);
    const rows = board.map((e, i) =>
      `<div class="row ${i === youIndex ? 'you' : ''}"><span>${i + 1}.</span>` +
      `<span>${e.t.toFixed(2)} s</span></div>`).join('');
    overlayEl.classList.remove('hidden');
    overlayEl.innerHTML =
      `<h1>Finished! 🏁</h1>` +
      `<p>${LEVELS[this.level].name} — your time</p>` +
      `<div class="big" style="font-size:54px">${t.toFixed(2)} s</div>` +
      `<div class="lb"><div class="row" style="color:var(--muted)"><span>Best times</span><span></span></div>${rows}</div>` +
      `<button id="again">Play again ▶</button>`;
    document.getElementById('again').onclick = () => this._startCountdown();
    this._renderBest();
    this._announceRunState();
  }

  // --- leaderboard (localStorage, per level, best 5) -----------------------
  _boardKey() { return `nmf-game-ctrltime-${this.level}`; }
  _getBoard() { try { return JSON.parse(localStorage.getItem(this._boardKey())) || []; } catch { return []; } }
  _updateBoard(t) {
    const board = this._getBoard();
    const entry = { t };
    board.push(entry);
    board.sort((a, b) => a.t - b.t);
    const top = board.slice(0, 5);
    try { localStorage.setItem(this._boardKey(), JSON.stringify(top.map((e) => ({ t: e.t })))); } catch { /* ignore */ }
    return { board: top, youIndex: top.indexOf(entry) };
  }
  _renderBest() {
    const board = this._getBoard();
    document.getElementById('best').textContent =
      board.length ? `Best: ${board[0].t.toFixed(2)} s` : 'Best: —';
  }

  // Help line for the current level, with the joystick hint appended whenever a
  // gamepad is connected (re-run on connect/disconnect via the Gamepad callback).
  _renderHelp() {
    const pad = this.pad && this.pad.connected ? `<div class="row2 pad">${HELP_PAD[this.level]}</div>` : '';
    document.getElementById('help').innerHTML = HELP[this.level] + pad;
  }

  _wireUi() {
    document.querySelectorAll('#levels button').forEach((bt) =>
      bt.addEventListener('click', () => this.setLevel(bt.dataset.level)));
    document.getElementById('brain-learning').onclick=()=>{this.brainLearning=!this.brainLearning;document.getElementById('brain-learning').textContent=this.brainLearning?'학습 ON':'학습 OFF'};
    document.getElementById('brain-reset').onclick=()=>{this._brainSend({type:'reset_brain'});this._brainSend({type:'arena_reset',keep_learning:false});this.restart()};
    document.getElementById('looming').onclick=()=>{this.brainThreat=1};
    document.getElementById('upright').onclick=()=>{this._manualRighting=.28;this._invertedFor=Math.max(this._invertedFor,.08)};
    document.getElementById('flip-test').onclick=()=>this._flipForTest();
    document.querySelectorAll('#mobility button').forEach((button)=>button.onclick=()=>this._setMobilityMode(button.dataset.mode));
    document.querySelectorAll('[data-object]').forEach((button)=>button.onclick=()=>{
      this.placementKind=this.placementKind===button.dataset.object?null:button.dataset.object;this._renderPlacementState();
    });
    document.getElementById('demo-world').onclick=()=>{this._populateDemoWorld();this._renderPlacementState()};
    document.getElementById('shuffle-world').onclick=()=>this._shuffleStimuli();
    document.getElementById('clear-world').onclick=()=>{this.clearStimuli();this.placementKind=null;this._renderPlacementState()};
    this._setMobilityMode('auto'); this._renderPlacementState();
    this.setLevel('brain');
  }

  // --- finish detection: does the path prev->cur cross the finish segment? --
  _crossed(cx, cy) {
    const ccw = (ax, ay, bx, by, c0, c1) => (c1 - ay) * (bx - ax) > (by - ay) * (c0 - ax);
    const [a, b] = this.finish, [px, py] = this.prevXY;
    return ccw(px, py, a[0], a[1], b[0], b[1]) !== ccw(cx, cy, a[0], a[1], b[0], b[1]) &&
           ccw(px, py, cx, cy, a[0], a[1]) !== ccw(px, py, cx, cy, b[0], b[1]);
  }

  // --- one physics substep: write ctrl from the active controller, then step -
  _physicsStep() {
    const ctrl = this.data.ctrl;
    const pad = this._padState;                       // sampled once this frame
    if (this.level === 'brain') {
      this.controller.stepCPG(ctrl, this.brain.gain_left || 0, this.brain.gain_right || 0);
    } else if (this.level === 'CPG') {
      // Stick (when engaged) overrides the persistent keyboard gains.
      let gL = this.input.gainL, gR = this.input.gainR;
      if (pad && pad.active) { gL = pad.gainL; gR = pad.gainR; }
      this.controller.stepCPG(ctrl, gL, gR);
    } else if (this.level === 'tripod') {
      const a = this.input.tripodAction(this._tripodAct);
      if (pad) for (let g = 0; g < 2; g++) if (this.pad.tripod[g]) a[g] = this.pad.tripod[g];
      this.controller.stepTripod(ctrl, a);
    } else {
      const a = this.input.singleAction(this._singleAct);
      if (pad) for (let i = 0; i < 6; i++) if (this.pad.single[i]) a[i] = this.pad.single[i];
      this.controller.stepSingle(ctrl, a);
    }
    this._applyExternalForces();
    this.mj.mj_step(this.model, this.data);
    this.simTime += this.dt;
  }

  // --- main loop -----------------------------------------------------------
  _frame(nowMs) {
    requestAnimationFrame((t) => this._frame(t));
    const now = nowMs / 1000;
    const wallDt = this._lastWall === undefined ? 0 : Math.min(now - this._lastWall, 0.1);
    this._lastWall = now;

    // Poll the joystick once per frame; engaging it on the ready screen starts
    // the run, just like pressing a movement key.
    this._padState = this.pad.sample(this.level);
    if (this.phase === 'ready' && this._padState && this._padState.active) this._startCountdown();

    this._updateStimulusWorld(now);
    let nSteps = 0;
    if (this.phase === 'running') {
      const b = this.bodyId;
      // wallDt is scaled by PLAYBACK_SPEED so the sim advances in slow motion.
      nSteps = this._stepper.advance(wallDt * PLAYBACK_SPEED, () => {
        this._physicsStep();
        const cx = this.data.xpos[3 * b], cy = this.data.xpos[3 * b + 1];
        if (this._crossed(cx, cy)) { this.prevXY = [cx, cy]; this._finishRun(); return false; }
        this.prevXY = [cx, cy];
      });
    }

    syncMeshes(this.meshGroup, this.data);
    this._animateFlight(now);
    this._sendPhysicsObservation(now);
    this._updateCamera(false);
    this.renderer.render(this.scene, this.camera);

    if (this.phase === 'running') document.getElementById('timer').textContent = this._controlTime().toFixed(2);
    this._statsMeter(now, nSteps);
  }
}
