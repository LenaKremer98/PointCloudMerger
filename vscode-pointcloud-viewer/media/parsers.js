/* Parser fuer PCD und PLY. Liefert immer dasselbe Ergebnis:
   { count, positions: Float32Array(3n), colors: Uint8Array(3n)|null,
     scalar: Float32Array(n)|null, scalarName: string|null, format: string } */
(function (global) {
  'use strict';

  function ascii(u8, from, to) {
    let s = '';
    const end = Math.min(to, u8.length);
    // In Bloecken zusammensetzen, sonst sprengt der Argumentstapel bei grossen Headern.
    for (let i = from; i < end; i += 4096) {
      s += String.fromCharCode.apply(null, u8.subarray(i, Math.min(end, i + 4096)));
    }
    return s;
  }

  /* ---------------------------------------------------------------- LZF */
  // PCD "binary_compressed" nutzt libLZF. Der Dekompressor ist kurz genug,
  // um ihn hier direkt mitzuschreiben.
  function lzfDecompress(input, outLen) {
    const out = new Uint8Array(outLen);
    let ip = 0;
    let op = 0;
    while (ip < input.length && op < outLen) {
      const ctrl = input[ip++];
      if (ctrl < 32) {
        let len = ctrl + 1;
        while (len-- > 0) out[op++] = input[ip++];
      } else {
        let len = ctrl >> 5;
        let ref = op - ((ctrl & 0x1f) << 8) - 1;
        if (len === 7) len += input[ip++];
        ref -= input[ip++];
        if (ref < 0) throw new Error('LZF: Rueckverweis vor dem Puffer.');
        len += 2;
        while (len-- > 0) out[op++] = out[ref++];
      }
    }
    return out;
  }

  /* ----------------------------------------------------------------- PCD */
  function readValue(dv, off, type, size, little) {
    if (type === 'F') {
      return size === 8 ? dv.getFloat64(off, little) : dv.getFloat32(off, little);
    }
    if (type === 'U') {
      if (size === 1) return dv.getUint8(off);
      if (size === 2) return dv.getUint16(off, little);
      if (size === 8) return Number(dv.getBigUint64(off, little));
      return dv.getUint32(off, little);
    }
    if (size === 1) return dv.getInt8(off);
    if (size === 2) return dv.getInt16(off, little);
    if (size === 8) return Number(dv.getBigInt64(off, little));
    return dv.getInt32(off, little);
  }

  function pickField(names, wanted) {
    for (let i = 0; i < names.length; i++) {
      if (names[i].toLowerCase() === wanted) return i;
    }
    return -1;
  }

  function unpackRGB(value, type) {
    // PCD speichert RGB meist als 4 Byte, entweder als float uminterpretiert
    // oder direkt als uint32. In beiden Faellen liegen die Kanaele als
    // 0x00RRGGBB im Integer.
    let v;
    if (type === 'F') {
      const tmp = new Float32Array(1);
      tmp[0] = value;
      v = new Uint32Array(tmp.buffer)[0];
    } else {
      v = value >>> 0;
    }
    return [(v >> 16) & 0xff, (v >> 8) & 0xff, v & 0xff];
  }

  function parsePCD(buffer) {
    const u8 = new Uint8Array(buffer);
    const probe = ascii(u8, 0, Math.min(u8.length, 1 << 16));
    const m = /^[ \t]*DATA[ \t]+(\S+)[ \t]*\r?\n/im.exec(probe);
    if (!m) throw new Error('Kein DATA-Feld im PCD-Header gefunden.');
    const dataStart = m.index + m[0].length;
    const dataMode = m[1].toLowerCase();

    const meta = {};
    probe.slice(0, m.index).split('\n').forEach(function (raw) {
      const line = raw.trim();
      if (!line || line.charAt(0) === '#') return;
      const sp = line.search(/\s/);
      if (sp < 0) return;
      meta[line.slice(0, sp).toUpperCase()] = line.slice(sp + 1).trim();
    });

    const fields = (meta.FIELDS || '').split(/\s+/).filter(Boolean);
    if (!fields.length) throw new Error('PCD ohne FIELDS.');
    const sizes = (meta.SIZE || '').split(/\s+/).filter(Boolean).map(Number);
    const types = (meta.TYPE || '').split(/\s+/).filter(Boolean).map(function (t) {
      return t.toUpperCase();
    });
    const counts = meta.COUNT
      ? meta.COUNT.split(/\s+/).filter(Boolean).map(Number)
      : fields.map(function () { return 1; });

    const width = parseInt(meta.WIDTH || '0', 10) || 0;
    const height = parseInt(meta.HEIGHT || '1', 10) || 1;
    let count = meta.POINTS ? parseInt(meta.POINTS, 10) : width * height;
    if (!count || count < 0) throw new Error('PCD ohne brauchbare Punktzahl.');

    const ix = pickField(fields, 'x');
    const iy = pickField(fields, 'y');
    const iz = pickField(fields, 'z');
    if (ix < 0 || iy < 0 || iz < 0) throw new Error('PCD ohne x/y/z-Felder.');

    let irgb = pickField(fields, 'rgb');
    if (irgb < 0) irgb = pickField(fields, 'rgba');
    const ir = pickField(fields, 'red');
    const ig = pickField(fields, 'green');
    const ib = pickField(fields, 'blue');
    let iscal = pickField(fields, 'intensity');
    if (iscal < 0) iscal = pickField(fields, 'reflectivity');

    const hasSplitRGB = ir >= 0 && ig >= 0 && ib >= 0;
    const positions = new Float32Array(count * 3);
    const colors = (irgb >= 0 || hasSplitRGB) ? new Uint8Array(count * 3) : null;
    const scalar = iscal >= 0 ? new Float32Array(count) : null;

    if (dataMode === 'ascii') {
      const text = ascii(u8, dataStart, u8.length);
      const lines = text.split('\n');
      let p = 0;
      for (let i = 0; i < lines.length && p < count; i++) {
        const line = lines[i].trim();
        if (!line) continue;
        const t = line.split(/\s+/);
        if (t.length < fields.length) continue;
        positions[3 * p] = parseFloat(t[ix]);
        positions[3 * p + 1] = parseFloat(t[iy]);
        positions[3 * p + 2] = parseFloat(t[iz]);
        if (colors) {
          if (hasSplitRGB) {
            colors[3 * p] = parseFloat(t[ir]) | 0;
            colors[3 * p + 1] = parseFloat(t[ig]) | 0;
            colors[3 * p + 2] = parseFloat(t[ib]) | 0;
          } else {
            const c = unpackRGB(parseFloat(t[irgb]), types[irgb]);
            colors[3 * p] = c[0];
            colors[3 * p + 1] = c[1];
            colors[3 * p + 2] = c[2];
          }
        }
        if (scalar) scalar[p] = parseFloat(t[iscal]);
        p++;
      }
      count = p;
    } else if (dataMode === 'binary') {
      const offs = [];
      let stride = 0;
      for (let i = 0; i < fields.length; i++) {
        offs.push(stride);
        stride += sizes[i] * counts[i];
      }
      const avail = Math.floor((u8.length - dataStart) / stride);
      if (avail < count) count = avail;
      const dv = new DataView(buffer);
      for (let p = 0; p < count; p++) {
        const b = dataStart + p * stride;
        positions[3 * p] = readValue(dv, b + offs[ix], types[ix], sizes[ix], true);
        positions[3 * p + 1] = readValue(dv, b + offs[iy], types[iy], sizes[iy], true);
        positions[3 * p + 2] = readValue(dv, b + offs[iz], types[iz], sizes[iz], true);
        if (colors) {
          if (hasSplitRGB) {
            colors[3 * p] = readValue(dv, b + offs[ir], types[ir], sizes[ir], true);
            colors[3 * p + 1] = readValue(dv, b + offs[ig], types[ig], sizes[ig], true);
            colors[3 * p + 2] = readValue(dv, b + offs[ib], types[ib], sizes[ib], true);
          } else {
            const c = unpackRGB(
              readValue(dv, b + offs[irgb], types[irgb], sizes[irgb], true), types[irgb]);
            colors[3 * p] = c[0];
            colors[3 * p + 1] = c[1];
            colors[3 * p + 2] = c[2];
          }
        }
        if (scalar) scalar[p] = readValue(dv, b + offs[iscal], types[iscal], sizes[iscal], true);
      }
    } else if (dataMode === 'binary_compressed') {
      const head = new DataView(buffer, dataStart, 8);
      const compLen = head.getUint32(0, true);
      const rawLen = head.getUint32(4, true);
      const comp = new Uint8Array(buffer, dataStart + 8, compLen);
      const raw = lzfDecompress(comp, rawLen);
      const dv = new DataView(raw.buffer, raw.byteOffset, raw.byteLength);
      // Die entpackten Daten liegen spaltenweise: erst alle x, dann alle y usw.
      const colStart = [];
      let acc = 0;
      for (let i = 0; i < fields.length; i++) {
        colStart.push(acc);
        acc += count * sizes[i] * counts[i];
      }
      const at = function (fi, p) {
        return colStart[fi] + p * sizes[fi] * counts[fi];
      };
      for (let p = 0; p < count; p++) {
        positions[3 * p] = readValue(dv, at(ix, p), types[ix], sizes[ix], true);
        positions[3 * p + 1] = readValue(dv, at(iy, p), types[iy], sizes[iy], true);
        positions[3 * p + 2] = readValue(dv, at(iz, p), types[iz], sizes[iz], true);
        if (colors) {
          if (hasSplitRGB) {
            colors[3 * p] = readValue(dv, at(ir, p), types[ir], sizes[ir], true);
            colors[3 * p + 1] = readValue(dv, at(ig, p), types[ig], sizes[ig], true);
            colors[3 * p + 2] = readValue(dv, at(ib, p), types[ib], sizes[ib], true);
          } else {
            const c = unpackRGB(readValue(dv, at(irgb, p), types[irgb], sizes[irgb], true),
              types[irgb]);
            colors[3 * p] = c[0];
            colors[3 * p + 1] = c[1];
            colors[3 * p + 2] = c[2];
          }
        }
        if (scalar) scalar[p] = readValue(dv, at(iscal, p), types[iscal], sizes[iscal], true);
      }
    } else {
      throw new Error('PCD-Datenformat "' + dataMode + '" wird nicht unterstuetzt.');
    }

    return {
      count: count,
      positions: positions,
      colors: colors,
      scalar: scalar,
      scalarName: iscal >= 0 ? fields[iscal] : null,
      format: 'PCD ' + dataMode + ', Felder ' + fields.join(' ')
    };
  }

  /* ----------------------------------------------------------------- PLY */
  const PLY_SIZE = {
    char: 1, int8: 1, uchar: 1, uint8: 1,
    short: 2, int16: 2, ushort: 2, uint16: 2,
    int: 4, int32: 4, uint: 4, uint32: 4,
    float: 4, float32: 4, double: 8, float64: 8
  };

  function plyRead(dv, off, type, little) {
    switch (type) {
      case 'char': case 'int8': return dv.getInt8(off);
      case 'uchar': case 'uint8': return dv.getUint8(off);
      case 'short': case 'int16': return dv.getInt16(off, little);
      case 'ushort': case 'uint16': return dv.getUint16(off, little);
      case 'int': case 'int32': return dv.getInt32(off, little);
      case 'uint': case 'uint32': return dv.getUint32(off, little);
      case 'float': case 'float32': return dv.getFloat32(off, little);
      case 'double': case 'float64': return dv.getFloat64(off, little);
      default: throw new Error('PLY: unbekannter Typ ' + type);
    }
  }

  function isFloatType(t) {
    return t === 'float' || t === 'float32' || t === 'double' || t === 'float64';
  }

  function parsePLY(buffer) {
    const u8 = new Uint8Array(buffer);
    const probe = ascii(u8, 0, Math.min(u8.length, 1 << 20));
    const em = /end_header[ \t]*\r?\n/.exec(probe);
    if (!em) throw new Error('Kein end_header im PLY gefunden.');
    const dataStart = em.index + em[0].length;
    const headerText = probe.slice(0, em.index);

    let format = 'ascii';
    const elements = [];
    let current = null;
    headerText.split('\n').forEach(function (raw) {
      const line = raw.trim();
      if (!line) return;
      const t = line.split(/\s+/);
      if (t[0] === 'format') {
        format = t[1];
      } else if (t[0] === 'element') {
        current = { name: t[1], count: parseInt(t[2], 10), props: [] };
        elements.push(current);
      } else if (t[0] === 'property' && current) {
        if (t[1] === 'list') {
          current.props.push({ list: true, countType: t[2], itemType: t[3], name: t[4] });
        } else {
          current.props.push({ list: false, type: t[1], name: t[2] });
        }
      }
    });

    const vertex = elements.filter(function (e) { return e.name === 'vertex'; })[0];
    if (!vertex) throw new Error('PLY ohne vertex-Element.');

    const names = vertex.props.map(function (p) { return p.name.toLowerCase(); });
    const find = function () {
      for (let a = 0; a < arguments.length; a++) {
        const i = names.indexOf(arguments[a]);
        if (i >= 0) return i;
      }
      return -1;
    };
    const ix = find('x');
    const iy = find('y');
    const iz = find('z');
    if (ix < 0 || iy < 0 || iz < 0) throw new Error('PLY ohne x/y/z.');
    const ir = find('red', 'r', 'diffuse_red');
    const ig = find('green', 'g', 'diffuse_green');
    const ib = find('blue', 'b', 'diffuse_blue');
    const iscal = find('intensity', 'scalar_intensity', 'gray', 'scalar_field', 'quality');

    const hasRGB = ir >= 0 && ig >= 0 && ib >= 0;
    let count = vertex.count;
    const positions = new Float32Array(count * 3);
    const colors = hasRGB ? new Uint8Array(count * 3) : null;
    const scalar = iscal >= 0 ? new Float32Array(count) : null;
    const colScale = hasRGB && isFloatType(vertex.props[ir].type) ? 255 : 1;

    if (format === 'ascii') {
      const text = ascii(u8, dataStart, u8.length);
      const lines = text.split('\n');
      let li = 0;
      // Elemente vor vertex ueberspringen
      for (let e = 0; e < elements.length && elements[e] !== vertex; e++) {
        li += elements[e].count;
      }
      let p = 0;
      for (; li < lines.length && p < count; li++) {
        const line = lines[li].trim();
        if (!line) continue;
        const t = line.split(/\s+/);
        positions[3 * p] = parseFloat(t[ix]);
        positions[3 * p + 1] = parseFloat(t[iy]);
        positions[3 * p + 2] = parseFloat(t[iz]);
        if (colors) {
          colors[3 * p] = Math.round(parseFloat(t[ir]) * colScale);
          colors[3 * p + 1] = Math.round(parseFloat(t[ig]) * colScale);
          colors[3 * p + 2] = Math.round(parseFloat(t[ib]) * colScale);
        }
        if (scalar) scalar[p] = parseFloat(t[iscal]);
        p++;
      }
      count = p;
    } else {
      const little = format === 'binary_little_endian';
      const dv = new DataView(buffer);
      let off = dataStart;

      // Alles vor dem vertex-Element ueberspringen
      for (let e = 0; e < elements.length && elements[e] !== vertex; e++) {
        const el = elements[e];
        let fixed = 0;
        let hasList = false;
        el.props.forEach(function (pr) {
          if (pr.list) hasList = true; else fixed += PLY_SIZE[pr.type];
        });
        if (!hasList) {
          off += fixed * el.count;
        } else {
          for (let k = 0; k < el.count; k++) {
            el.props.forEach(function (pr) {
              if (!pr.list) {
                off += PLY_SIZE[pr.type];
              } else {
                const n = plyRead(dv, off, pr.countType, little);
                off += PLY_SIZE[pr.countType] + n * PLY_SIZE[pr.itemType];
              }
            });
          }
        }
      }

      const vfixed = vertex.props.every(function (pr) { return !pr.list; });
      if (vfixed) {
        const offs = [];
        let stride = 0;
        vertex.props.forEach(function (pr) {
          offs.push(stride);
          stride += PLY_SIZE[pr.type];
        });
        const avail = Math.floor((u8.length - off) / stride);
        if (avail < count) count = avail;
        for (let p = 0; p < count; p++) {
          const b = off + p * stride;
          positions[3 * p] = plyRead(dv, b + offs[ix], vertex.props[ix].type, little);
          positions[3 * p + 1] = plyRead(dv, b + offs[iy], vertex.props[iy].type, little);
          positions[3 * p + 2] = plyRead(dv, b + offs[iz], vertex.props[iz].type, little);
          if (colors) {
            colors[3 * p] = plyRead(dv, b + offs[ir], vertex.props[ir].type, little) * colScale;
            colors[3 * p + 1] = plyRead(dv, b + offs[ig], vertex.props[ig].type, little) * colScale;
            colors[3 * p + 2] = plyRead(dv, b + offs[ib], vertex.props[ib].type, little) * colScale;
          }
          if (scalar) scalar[p] = plyRead(dv, b + offs[iscal], vertex.props[iscal].type, little);
        }
      } else {
        // Listeneigenschaften im vertex-Element sind selten, aber moeglich.
        for (let p = 0; p < count; p++) {
          for (let q = 0; q < vertex.props.length; q++) {
            const pr = vertex.props[q];
            if (pr.list) {
              const n = plyRead(dv, off, pr.countType, little);
              off += PLY_SIZE[pr.countType] + n * PLY_SIZE[pr.itemType];
              continue;
            }
            const v = plyRead(dv, off, pr.type, little);
            if (q === ix) positions[3 * p] = v;
            else if (q === iy) positions[3 * p + 1] = v;
            else if (q === iz) positions[3 * p + 2] = v;
            else if (colors && q === ir) colors[3 * p] = v * colScale;
            else if (colors && q === ig) colors[3 * p + 1] = v * colScale;
            else if (colors && q === ib) colors[3 * p + 2] = v * colScale;
            else if (scalar && q === iscal) scalar[p] = v;
            off += PLY_SIZE[pr.type];
          }
        }
      }
    }

    return {
      count: count,
      positions: positions,
      colors: colors,
      scalar: scalar,
      scalarName: iscal >= 0 ? vertex.props[iscal].name : null,
      format: 'PLY ' + format + ', ' + vertex.props.length + ' Eigenschaften'
    };
  }

  function parse(buffer, name) {
    const u8 = new Uint8Array(buffer);
    const magic = ascii(u8, 0, Math.min(u8.length, 4)).toLowerCase();
    if (magic.slice(0, 3) === 'ply') return parsePLY(buffer);
    if (/\.ply$/i.test(name)) return parsePLY(buffer);
    return parsePCD(buffer);
  }

  global.CloudParsers = { parse: parse, parsePCD: parsePCD, parsePLY: parsePLY };
})(window);
