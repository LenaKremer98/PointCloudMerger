const vscode = require('vscode');

const VIEW_TYPE = 'pointcloudViewer.cloud';
const CHUNK = 1 << 20; // 1 MiB je Nachricht im Notfallweg

function nonce() {
  let s = '';
  const abc = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789';
  for (let i = 0; i < 32; i++) s += abc.charAt(Math.floor(Math.random() * abc.length));
  return s;
}

class PointCloudEditorProvider {
  constructor(context) {
    this.context = context;
  }

  async openCustomDocument(uri) {
    return { uri, dispose() {} };
  }

  async resolveCustomEditor(document, panel) {
    const folder = vscode.Uri.joinPath(document.uri, '..');
    panel.webview.options = {
      enableScripts: true,
      localResourceRoots: [this.context.extensionUri, folder]
    };

    panel.webview.html = this.buildHtml(panel.webview, document.uri);

    panel.webview.onDidReceiveMessage(async (msg) => {
      if (!msg) return;
      if (msg.type === 'needBytes') {
        // Der direkte Weg ueber asWebviewUri hat nicht geklappt, also
        // schicken wir die Datei in Haeppchen als base64 hinueber.
        try {
          const bytes = await vscode.workspace.fs.readFile(document.uri);
          const total = Math.max(1, Math.ceil(bytes.length / CHUNK));
          panel.webview.postMessage({ type: 'begin', size: bytes.length, total });
          for (let i = 0; i < total; i++) {
            const slice = bytes.subarray(i * CHUNK, Math.min(bytes.length, (i + 1) * CHUNK));
            panel.webview.postMessage({
              type: 'chunk',
              index: i,
              data: Buffer.from(slice).toString('base64')
            });
          }
          panel.webview.postMessage({ type: 'end' });
        } catch (err) {
          panel.webview.postMessage({ type: 'error', message: String(err && err.message || err) });
        }
      } else if (msg.type === 'error') {
        vscode.window.showErrorMessage('Punktwolken-Viewer: ' + msg.message);
      } else if (msg.type === 'info') {
        vscode.window.setStatusBarMessage('Punktwolken-Viewer: ' + msg.message, 5000);
      }
    });
  }

  buildHtml(webview, fileUri) {
    const media = (name) =>
      webview.asWebviewUri(vscode.Uri.joinPath(this.context.extensionUri, 'media', name));
    const cloudUri = webview.asWebviewUri(fileUri);
    const cfg = vscode.workspace.getConfiguration('pointcloudViewer');
    const settings = {
      pointSize: cfg.get('pointSize', 2),
      background: cfg.get('background', 'dunkel'),
      upAxis: cfg.get('upAxis', 'z')
    };
    const n = nonce();
    const csp = [
      "default-src 'none'",
      `img-src ${webview.cspSource} data: blob:`,
      `style-src ${webview.cspSource} 'unsafe-inline'`,
      `script-src 'nonce-${n}'`,
      `connect-src ${webview.cspSource} blob: data:`
    ].join('; ');

    const name = fileUri.path.split('/').pop() || 'Punktwolke';

    return `<!DOCTYPE html>
<html lang="de">
<head>
<meta charset="utf-8">
<meta http-equiv="Content-Security-Policy" content="${csp}">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<link rel="stylesheet" href="${media('viewer.css')}">
<title>${name}</title>
</head>
<body>
<div id="bar">
  <span class="grp">
    <label for="mode">Farbe</label>
    <select id="mode"></select>
  </span>
  <span class="grp">
    <label for="size">Punkte</label>
    <input id="size" type="range" min="1" max="10" step="1" value="${settings.pointSize}">
    <span id="sizeval">${settings.pointSize}</span>
  </span>
  <span class="grp">
    <label for="up">Oben</label>
    <select id="up">
      <option value="z"${settings.upAxis === 'z' ? ' selected' : ''}>Z</option>
      <option value="y"${settings.upAxis === 'y' ? ' selected' : ''}>Y</option>
    </select>
  </span>
  <span class="grp">
    <label><input id="axes" type="checkbox" checked> Achsen</label>
  </span>
  <span class="grp">
    <label><input id="bg" type="checkbox"${settings.background === 'hell' ? ' checked' : ''}> Hell</label>
  </span>
  <span class="grp"><button id="reset">Ansicht zurücksetzen</button></span>
  <span class="spacer"></span>
  <span id="head"></span>
</div>
<canvas id="gl"></canvas>
<div id="status">lädt …</div>
<div id="overlay"><div id="overlaytext">lädt …</div></div>
<script nonce="${n}">
  window.__CLOUD__ = { uri: ${JSON.stringify(cloudUri.toString())}, name: ${JSON.stringify(name)} };
</script>
<script nonce="${n}" src="${media('parsers.js')}"></script>
<script nonce="${n}" src="${media('viewer.js')}"></script>
</body>
</html>`;
  }
}

function activate(context) {
  context.subscriptions.push(
    vscode.window.registerCustomEditorProvider(
      VIEW_TYPE,
      new PointCloudEditorProvider(context),
      {
        webviewOptions: { retainContextWhenHidden: true },
        supportsMultipleEditorsPerDocument: false
      }
    )
  );
}

function deactivate() {}

module.exports = { activate, deactivate };
