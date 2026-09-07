(function () {
  'use strict';

  const vscode = acquireVsCodeApi();
  const canvas = document.getElementById('gl');
  const statusEl = document.getElementById('status');
  const headEl = document.getElementById('head');
  const overlay = document.getElementById('overlay');
  const overlayText = document.getElementById('overlaytext');
  const modeSel = document.getElementById('mode');
  const sizeInput = document.getElementById('size');
  const sizeVal = document.getElementById('sizeval');
  const upSel = document.getElementById('up');
  const axesBox = document.getElementById('axes');
  const bgBox = document.getElementById('bg');
  const resetBtn = document.getElementById('reset');

  /* ------------------------------------------------------------ Matrizen */
  function mat4() { return new Float32Array(16); }

  function perspective(out, fovy, aspect, near, far) {
    const f = 1 / Math.tan(fovy / 2);
    out.fill(0);
    out[0] = f / aspect;
    out[5] = f;
    out[10] = (far + near) / (near - far);
    out[11] = -1;
    out[14] = (2 * far * near) / (near - far);
    return out;
  }

  function lookAt(out, eye, center, up) {
    let zx = eye[0] - center[0], zy = eye[1] - center[1], zz = eye[2] - center[2];
    let l = Math.hypot(zx, zy, zz) || 1;
    zx /= l; zy /= l; zz /= l;
    let xx = up[1] * zz - up[2] * zy;
    let xy = up[2] * zx - up[0] * zz;
    let xz = up[0] * zy - up[1] * zx;
    l = Math.hypot(xx, xy, xz);
    if (l < 1e-9) { xx = 1; xy = 0; xz = 0; } else { xx /= l; xy /= l; xz /= l; }
    const yx = zy * xz - zz * xy;
    const yy = zz * xx - zx * xz;
    const yz = zx * xy - zy * xx;
    out[0] = xx; out[1] = yx; out[2] = zx; out[3] = 0;
    out[4] = xy; out[5] = yy; out[6] = zy; out[7] = 0;
    out[8] = xz; out[9] = yz; out[10] = zz; out[11] = 0;
    out[12] = -(xx * eye[0] + xy * eye[1] + xz * eye[2]);
    out[13] = -(yx * eye[0] + yy * eye[1] + yz * eye[2]);
    out[14] = -(zx * eye[0] + zy * eye[1] + zz * eye[2]);
    out[15] = 1;
    return out;
  }

  function multiply(out, a, b) {
    for (let c = 0; c < 4; c++) {
      const b0 = b[c * 4], b1 = b[c * 4 + 1], b2 = b[c * 4 + 2], b3 = b[c * 4 + 3];
      out[c * 4] = a[0] * b0 + a[4] * b1 + a[8] * b2 + a[12] * b3;
      out[c * 4 + 1] = a[1] * b0 + a[5] * b1 + a[9] * b2 + a[13] * b3;
      out[c * 4 + 2] = a[2] * b0 + a[6] * b1 + a[10] * b2 + a[14] * b3;
      out[c * 4 + 3] = a[3] * b0 + a[7] * b1 + a[11] * b2 + a[15] * b3;
    }
    return out;
  }

  /* -------------------------------------------------------------- Shader */
  const VS = [
    'attribute vec3 aPos;',
    'attribute vec3 aRGB;',
    'attribute float aScalar;',
    'uniform mat4 uMVP;',
    'uniform float uPointSize;',
    'uniform int uMode;',
    'uniform int uUpZ;',
    'uniform float uLo;',
    'uniform float uHi;',
    'uniform vec3 uFlat;',
    'varying vec3 vColor;',
    'vec3 turbo(float t){',
    '  t = clamp(t, 0.0, 1.0);',
    '  vec4 v4 = vec4(1.0, t, t*t, t*t*t);',
    '  vec2 v2 = v4.zw * v4.z;',
    '  float r = dot(v4, vec4(0.13572138, 4.61539260, -42.66032258, 132.13108234))',
    '          + dot(v2, vec2(-152.94239396, 59.28637943));',
    '  float g = dot(v4, vec4(0.09140261, 2.19418839, 4.84296658, -14.18503333))',
    '          + dot(v2, vec2(4.27729857, 2.82956604));',
    '  float b = dot(v4, vec4(0.10667330, 12.64194608, -60.58204836, 110.36276771))',
    '          + dot(v2, vec2(-89.90310912, 27.34824973));',
    '  return clamp(vec3(r, g, b), 0.0, 1.0);',
    '}',
    'void main() {',
    '  gl_Position = uMVP * vec4(aPos, 1.0);',
    '  gl_PointSize = uPointSize;',
    '  if (uMode == 0) {',
    '    vColor = aRGB;',
    '  } else if (uMode == 1) {',
    '    float h = (uUpZ == 1) ? aPos.z : aPos.y;',
    '    vColor = turbo((h - uLo) / max(uHi - uLo, 1e-6));',
    '  } else if (uMode == 2) {',
    '    vColor = turbo((aScalar - uLo) / max(uHi - uLo, 1e-6));',
    '  } else {',
    '    vColor = uFlat;',
    '  }',
    '}'
  ].join('\n');

  const FS = [
    'precision mediump float;',
    'varying vec3 vColor;',
    'uniform int uRound;',
    'void main() {',
    '  if (uRound == 1) {',
    '    vec2 d = gl_PointCoord - vec2(0.5);',
    '    if (dot(d, d) > 0.25) discard;',
    '  }',
    '  gl_FragColor = vec4(vColor, 1.0);',
    '}'
  ].join('\n');

  /* ----------------------------------------------------------------- GL */
  const gl = canvas.getContext('webgl2', { antialias: true, depth: true })
    || canvas.getContext('webgl', { antialias: true, depth: true });
  if (!gl) {
    fail('Dieser Webview hat kein WebGL. In den Einstellungen "disable-hardware-acceleration" pruefen.');
    return;
  }

  function compile(type, src) {
    const sh = gl.createShader(type);
    gl.shaderSource(sh, src);
    gl.compileShader(sh);
    if (!gl.getShaderParameter(sh, gl.COMPILE_STATUS)) {
      throw new Error('Shader: ' + gl.getShaderInfoLog(sh));
    }
    return sh;
  }

  const prog = gl.createProgram();
  gl.attachShader(prog, compile(gl.VERTEX_SHADER, VS));
  gl.attachShader(prog, compile(gl.FRAGMENT_SHADER, FS));
  gl.linkProgram(prog);
  if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) {
    fail('Shader-Link: ' + gl.getProgramInfoLog(prog));
    return;
  }
  gl.useProgram(prog);

  const A = {
    pos: gl.getAttribLocation(prog, 'aPos'),
    rgb: gl.getAttribLocation(prog, 'aRGB'),
    scalar: gl.getAttribLocation(prog, 'aScalar')
  };
  const U = {};
  ['uMVP', 'uPointSize', 'uMode', 'uUpZ', 'uLo', 'uHi', 'uFlat', 'uRound'].forEach(function (n) {
    U[n] = gl.getUniformLocation(prog, n);
  });

  gl.enable(gl.DEPTH_TEST);

  /* --------------------------------------------------------------- State */
  const cloud = {
    count: 0,
    centroid: [0, 0, 0],
    min: [0, 0, 0],
    max: [0, 0, 0],
    radius: 1,
    hasRGB: false,
    scalarName: null,
    zLo: 0, zHi: 1,
    yLo: 0, yHi: 1,
    sLo: 0, sHi: 1
  };
  let posBuf = null, rgbBuf = null, scalarBuf = null;
  let axisBuf = null, axisColBuf = null, axisLen = 1;

  const cam = { yaw: -Math.PI / 4, pitch: 0.45, dist: 10, target: [0, 0, 0] };
  const view = mat4(), proj = mat4(), mvp = mat4();
  let upZ = upSel.value === 'z';
  let mode = 0;
  let pointSize = parseFloat(sizeInput.value) || 2;
  let showAxes = true;
  let light = bgBox.checked;
  let needsDraw = true;

  function fail(msg) {
    overlay.style.display = 'flex';
    overlayText.textContent = msg;
    overlayText.className = 'err';
    statusEl.textContent = msg;
    vscode.postMessage({ type: 'error', message: msg });
  }

  /* ------------------------------------------------------------- Kamera */
  function eyePos() {
    const cp = Math.cos(cam.pitch), sp = Math.sin(cam.pitch);
    const cy = Math.cos(cam.yaw), sy = Math.sin(cam.yaw);
    const d = upZ ? [cp * cy, cp * sy, sp] : [cp * sy, sp, cp * cy];
    return [
      cam.target[0] + cam.dist * d[0],
      cam.target[1] + cam.dist * d[1],
      cam.target[2] + cam.dist * d[2]
    ];
  }

  function resetView() {
    cam.target = [0, 0, 0];
    cam.dist = Math.max(cloud.radius * 2.4, 1e-3);
    cam.yaw = -Math.PI / 4;
    cam.pitch = 0.5;
    needsDraw = true;
  }

  /* ------------------------------------------------------------- Zeichnen */
  function resize() {
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const w = Math.max(1, Math.floor(canvas.clientWidth * dpr));
    const h = Math.max(1, Math.floor(canvas.clientHeight * dpr));
    if (canvas.width !== w || canvas.height !== h) {
      canvas.width = w;
      canvas.height = h;
      needsDraw = true;
    }
  }

  function bindAttr(loc, buf, size, type, normalized) {
    if (loc < 0) return;
    if (buf) {
      gl.bindBuffer(gl.ARRAY_BUFFER, buf);
      gl.enableVertexAttribArray(loc);
      gl.vertexAttribPointer(loc, size, type, !!normalized, 0, 0);
    } else {
      gl.disableVertexAttribArray(loc);
      if (size === 1) gl.vertexAttrib1f(loc, 0);
      else gl.vertexAttrib3f(loc, 0.85, 0.85, 0.85);
    }
  }

  function draw() {
    resize();
    gl.viewport(0, 0, canvas.width, canvas.height);
    if (light) gl.clearColor(0.96, 0.96, 0.96, 1);
    else gl.clearColor(0.09, 0.09, 0.11, 1);
    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
    if (!cloud.count) return;

    const aspect = canvas.width / Math.max(1, canvas.height);
    const near = Math.max(cam.dist * 0.001, 1e-4);
    const far = cam.dist * 4 + cloud.radius * 6;
    perspective(proj, 50 * Math.PI / 180, aspect, near, far);
    lookAt(view, eyePos(), cam.target, upZ ? [0, 0, 1] : [0, 1, 0]);
    multiply(mvp, proj, view);

    gl.useProgram(prog);
    gl.uniformMatrix4fv(U.uMVP, false, mvp);
    gl.uniform1f(U.uPointSize, pointSize);
    gl.uniform1i(U.uUpZ, upZ ? 1 : 0);
    gl.uniform3f(U.uFlat, light ? 0.2 : 0.85, light ? 0.25 : 0.85, light ? 0.3 : 0.9);
    gl.uniform1i(U.uRound, pointSize >= 3 ? 1 : 0);

    let lo = 0, hi = 1;
    if (mode === 1) {
      lo = upZ ? cloud.zLo : cloud.yLo;
      hi = upZ ? cloud.zHi : cloud.yHi;
    } else if (mode === 2) {
      lo = cloud.sLo;
      hi = cloud.sHi;
    }
    gl.uniform1f(U.uLo, lo);
    gl.uniform1f(U.uHi, hi);
    gl.uniform1i(U.uMode, mode);

    bindAttr(A.pos, posBuf, 3, gl.FLOAT, false);
    bindAttr(A.rgb, rgbBuf, 3, gl.UNSIGNED_BYTE, true);
    bindAttr(A.scalar, scalarBuf, 1, gl.FLOAT, false);
    gl.drawArrays(gl.POINTS, 0, cloud.count);

    if (showAxes && axisBuf) {
      gl.uniform1i(U.uMode, 0);
      gl.uniform1i(U.uRound, 0);
      bindAttr(A.pos, axisBuf, 3, gl.FLOAT, false);
      bindAttr(A.rgb, axisColBuf, 3, gl.UNSIGNED_BYTE, true);
      bindAttr(A.scalar, null, 1, gl.FLOAT, false);
      gl.drawArrays(gl.LINES, 0, 6);
    }
  }

  function loop() {
    if (needsDraw) {
      needsDraw = false;
      draw();
    }
    requestAnimationFrame(loop);
  }

  /* -------------------------------------------------------------- Maus */
  let dragging = 0;
  let lastX = 0, lastY = 0;

  canvas.addEventListener('contextmenu', function (e) { e.preventDefault(); });

  canvas.addEventListener('pointerdown', function (e) {
    canvas.setPointerCapture(e.pointerId);
    dragging = (e.button === 0 && !e.shiftKey && !e.ctrlKey) ? 1 : 2;
    lastX = e.clientX;
    lastY = e.clientY;
  });

  canvas.addEventListener('pointerup', function (e) {
    dragging = 0;
    try { canvas.releasePointerCapture(e.pointerId); } catch (err) { /* egal */ }
  });

  canvas.addEventListener('pointermove', function (e) {
    if (!dragging) return;
    const dx = e.clientX - lastX;
    const dy = e.clientY - lastY;
    lastX = e.clientX;
    lastY = e.clientY;
    if (dragging === 1) {
      cam.yaw -= dx * 0.006;
      cam.pitch += dy * 0.006;
      cam.pitch = Math.max(-1.553, Math.min(1.553, cam.pitch));
    } else {
      // Verschieben in der Bildebene, Schrittweite haengt am Abstand
      const k = cam.dist * Math.tan(25 * Math.PI / 180) * 2 / Math.max(1, canvas.clientHeight);
      const right = [view[0], view[4], view[8]];
      const up = [view[1], view[5], view[9]];
      for (let i = 0; i < 3; i++) {
        cam.target[i] += -right[i] * dx * k + up[i] * dy * k;
      }
    }
    needsDraw = true;
  });

  canvas.addEventListener('wheel', function (e) {
    e.preventDefault();
    cam.dist *= Math.exp(e.deltaY * 0.0012);
    cam.dist = Math.max(cloud.radius * 1e-4, Math.min(cloud.radius * 200, cam.dist));
    needsDraw = true;
  }, { passive: false });

  window.addEventListener('resize', function () { needsDraw = true; });

  window.addEventListener('keydown', function (e) {
    if (e.key === 'r' || e.key === 'R') { resetView(); }
  });

  /* ------------------------------------------------------------ Bedienung */
  modeSel.addEventListener('change', function () {
    mode = parseInt(modeSel.value, 10);
    needsDraw = true;
  });
  sizeInput.addEventListener('input', function () {
    pointSize = parseFloat(sizeInput.value);
    sizeVal.textContent = sizeInput.value;
    needsDraw = true;
  });
  upSel.addEventListener('change', function () {
    upZ = upSel.value === 'z';
    resetView();
  });
  axesBox.addEventListener('change', function () {
    showAxes = axesBox.checked;
    needsDraw = true;
  });
  bgBox.addEventListener('change', function () {
    light = bgBox.checked;
    document.body.classList.toggle('light', light);
    needsDraw = true;
  });
  resetBtn.addEventListener('click', resetView);

  /* ------------------------------------------------------------- Aufbau */
  function percentile(values, count, stride, lo, hi) {
    const step = Math.max(1, Math.floor(count / 300000));
    const sample = [];
    for (let i = 0; i < count; i += step) {
      const v = values[i * stride];
      if (isFinite(v)) sample.push(v);
    }
    if (!sample.length) return [0, 1];
    sample.sort(function (a, b) { return a - b; });
    const a = sample[Math.floor((sample.length - 1) * lo)];
    const b = sample[Math.floor((sample.length - 1) * hi)];
    return [a, b === a ? a + 1e-6 : b];
  }

  function build(data) {
    const n = data.count;
    if (!n) throw new Error('Die Datei enthaelt keine Punkte.');
    const pos = data.positions;

    let minX = Infinity, minY = Infinity, minZ = Infinity;
    let maxX = -Infinity, maxY = -Infinity, maxZ = -Infinity;
    let sx = 0, sy = 0, sz = 0;
    let valid = 0;
    for (let i = 0; i < n; i++) {
      const x = pos[3 * i], y = pos[3 * i + 1], z = pos[3 * i + 2];
      if (!isFinite(x) || !isFinite(y) || !isFinite(z)) continue;
      if (x < minX) minX = x; if (x > maxX) maxX = x;
      if (y < minY) minY = y; if (y > maxY) maxY = y;
      if (z < minZ) minZ = z; if (z > maxZ) maxZ = z;
      sx += x; sy += y; sz += z;
      valid++;
    }
    if (!valid) throw new Error('Alle Koordinaten sind ungueltig.');
    const c = [sx / valid, sy / valid, sz / valid];

    // Um den Schwerpunkt zentrieren, sonst leidet die float32-Genauigkeit
    // bei Karten mit grossem Offset.
    for (let i = 0; i < n; i++) {
      pos[3 * i] -= c[0];
      pos[3 * i + 1] -= c[1];
      pos[3 * i + 2] -= c[2];
    }

    cloud.count = n;
    cloud.centroid = c;
    cloud.min = [minX, minY, minZ];
    cloud.max = [maxX, maxY, maxZ];
    cloud.radius = 0.5 * Math.hypot(maxX - minX, maxY - minY, maxZ - minZ) || 1;
    cloud.hasRGB = !!data.colors;
    cloud.scalarName = data.scalarName;

    const zp = percentile(pos.subarray(2), n, 3, 0.02, 0.98);
    cloud.zLo = zp[0]; cloud.zHi = zp[1];
    const yp = percentile(pos.subarray(1), n, 3, 0.02, 0.98);
    cloud.yLo = yp[0]; cloud.yHi = yp[1];
    if (data.scalar) {
      const sp = percentile(data.scalar, n, 1, 0.02, 0.98);
      cloud.sLo = sp[0]; cloud.sHi = sp[1];
    }

    posBuf = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, posBuf);
    gl.bufferData(gl.ARRAY_BUFFER, pos, gl.STATIC_DRAW);

    if (data.colors) {
      rgbBuf = gl.createBuffer();
      gl.bindBuffer(gl.ARRAY_BUFFER, rgbBuf);
      gl.bufferData(gl.ARRAY_BUFFER, data.colors, gl.STATIC_DRAW);
    }
    if (data.scalar) {
      scalarBuf = gl.createBuffer();
      gl.bindBuffer(gl.ARRAY_BUFFER, scalarBuf);
      gl.bufferData(gl.ARRAY_BUFFER, data.scalar, gl.STATIC_DRAW);
    }

    // Achsenkreuz am Ursprung der Originaldaten
    axisLen = cloud.radius * 0.15;
    const o = [-c[0], -c[1], -c[2]];
    const av = new Float32Array([
      o[0], o[1], o[2], o[0] + axisLen, o[1], o[2],
      o[0], o[1], o[2], o[0], o[1] + axisLen, o[2],
      o[0], o[1], o[2], o[0], o[1], o[2] + axisLen
    ]);
    const ac = new Uint8Array([
      230, 60, 60, 230, 60, 60,
      60, 200, 80, 60, 200, 80,
      70, 120, 240, 70, 120, 240
    ]);
    axisBuf = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, axisBuf);
    gl.bufferData(gl.ARRAY_BUFFER, av, gl.STATIC_DRAW);
    axisColBuf = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, axisColBuf);
    gl.bufferData(gl.ARRAY_BUFFER, ac, gl.STATIC_DRAW);

    // Farbmodi anbieten, die es in dieser Datei wirklich gibt
    const opts = [];
    if (data.colors) opts.push([0, 'RGB']);
    opts.push([1, 'Höhe']);
    if (data.scalar) opts.push([2, data.scalarName || 'Skalar']);
    opts.push([3, 'Einfarbig']);
    modeSel.innerHTML = '';
    opts.forEach(function (o2) {
      const el = document.createElement('option');
      el.value = String(o2[0]);
      el.textContent = o2[1];
      modeSel.appendChild(el);
    });
    mode = opts[0][0];
    modeSel.value = String(mode);

    resetView();

    const f2 = function (v) { return v.toFixed(2); };
    headEl.textContent = window.__CLOUD__.name;
    statusEl.textContent =
      n.toLocaleString('de-DE') + ' Punkte · ' + data.format +
      ' · X [' + f2(minX) + ', ' + f2(maxX) + ']' +
      ' Y [' + f2(minY) + ', ' + f2(maxY) + ']' +
      ' Z [' + f2(minZ) + ', ' + f2(maxZ) + ']' +
      ' · Schwerpunkt [' + f2(c[0]) + ', ' + f2(c[1]) + ', ' + f2(c[2]) + ']' +
      (data.colors ? ' · mit RGB' : ' · ohne RGB');
    overlay.style.display = 'none';
  }

  /* -------------------------------------------------------------- Laden */
  function handle(buffer) {
    try {
      overlayText.textContent = 'zerlege ' + (buffer.byteLength / 1048576).toFixed(1) + ' MB …';
      const t0 = performance.now();
      const data = window.CloudParsers.parse(buffer, window.__CLOUD__.name);
      build(data);
      vscode.postMessage({
        type: 'info',
        message: data.count.toLocaleString('de-DE') + ' Punkte in ' +
          Math.round(performance.now() - t0) + ' ms geladen'
      });
    } catch (err) {
      fail(String(err && err.message || err));
    }
  }

  // Notfallweg, falls der direkte Zugriff auf die Datei nicht klappt
  let chunks = null;
  let chunkSize = 0;
  window.addEventListener('message', function (e) {
    const msg = e.data;
    if (!msg) return;
    if (msg.type === 'begin') {
      chunks = [];
      chunkSize = msg.size;
      overlayText.textContent = 'lädt ' + (msg.size / 1048576).toFixed(1) + ' MB …';
    } else if (msg.type === 'chunk' && chunks) {
      const bin = atob(msg.data);
      const arr = new Uint8Array(bin.length);
      for (let i = 0; i < bin.length; i++) arr[i] = bin.charCodeAt(i);
      chunks.push(arr);
    } else if (msg.type === 'end' && chunks) {
      const all = new Uint8Array(chunkSize);
      let off = 0;
      chunks.forEach(function (c) { all.set(c, off); off += c.length; });
      chunks = null;
      handle(all.buffer);
    } else if (msg.type === 'error') {
      fail(msg.message);
    }
  });

  document.body.classList.toggle('light', light);
  requestAnimationFrame(loop);

  fetch(window.__CLOUD__.uri)
    .then(function (r) {
      if (!r.ok) throw new Error('HTTP ' + r.status);
      return r.arrayBuffer();
    })
    .then(handle)
    .catch(function () {
      overlayText.textContent = 'lädt …';
      vscode.postMessage({ type: 'needBytes' });
    });
})();
