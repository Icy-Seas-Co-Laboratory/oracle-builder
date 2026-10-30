# Documentation screenshots

The dashboard screenshots were captured on 2026-10-30 from an existing
deployed Oracle Builder dashboard. The selected indexed run was the completed
example classification run `resnet18 powerlaw 128 gradient cce` (catalog prefix
`feea446d`, 30 completed epochs). The view was read-only: no work was submitted
or changed.

The browser chrome was cropped and each image was resized to 1800 pixels wide
before JPEG export. The recorded values are illustrative UI output from this
run, not a general benchmark and not evidence that the model is suitable for
deployment.

To refresh the images, choose a completed permitted demo run, wait for the
dashboard telemetry and charts to finish rendering, and capture the top-level
dashboard plus the scrolled learning-curves view. Crop browser chrome, remove
any tokens, credentials, private host URLs, or machine paths, and preserve the
displayed values. Do not submit or modify work while capturing documentation.

## README logo wrapper

`readme-logo.svg` is a self-contained presentation wrapper around the original `webgui/static/brand/oracle-builder-logo.png`, with a white rounded-rectangle backing for dark-theme contrast. It embeds the original artwork without recoloring or redrawing it. Regenerate the wrapper if the source logo changes.

The committed `readme-logo.png` derivative is rendered from that wrapper. To regenerate it without adding repository dependencies:

Run this from the oracle-builder repository root:

```bash
npm install --prefix "${TMPDIR:-/tmp}/pelagia-doc-assets" --no-save --no-audit --no-fund @resvg/resvg-js
NODE_PATH="${TMPDIR:-/tmp}/pelagia-doc-assets/node_modules" node <<'JS'
const { Resvg } = require('@resvg/resvg-js');
const fs = require('fs');
const svg = 'docs/assets/readme-logo.svg';
fs.writeFileSync('docs/assets/readme-logo.png', new Resvg(fs.readFileSync(svg, 'utf8')).render().asPng());
JS
```

The wrapper embeds the original artwork as base64; edit or regenerate that wrapper from the original logo when the source logo changes, then render the PNG derivative.

