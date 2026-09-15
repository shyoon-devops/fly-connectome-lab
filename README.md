# Fly Connectome Lab

An interactive browser lab for the public **MaleCNS v1.0** fruit-fly
connectome: whole-brain activity, stimulus controls, a small reward-learning
demo, and a NeuroMechFly v2 / MuJoCo-WASM physical arena.

> This is a research visualization and explicit computational model — not a
> live fly, a biological brain replica, or a validated behavioural predictor.

## Measured data vs. explicit model assumptions

- **Measured:** neuron IDs and annotations, directed connectivity,
  synapse-derived weights, neurotransmitter predictions, and soma locations.
- **Modelled here:** rate dynamics, stimulus mapping, the
  descending-neuron-to-CPG decoder, reward rule, and flight controller.
- **Physical scene:** NeuroMechFly v2 geometry and MuJoCo browser physics. The
  flight force model is an approximation, not CFD or measured aerodynamics.

## Data and setup

Research downloads and generated graph artifacts are deliberately excluded from
Git. Download these CC BY 4.0 MaleCNS v1.0 files from the official
[Janelia download page](https://male-cns.janelia.org/download/) into `data/raw/`:

1. `body-annotations-male-cns-v1.0-minconf-0.5.feather`
2. `body-neurotransmitters-male-cns-v1.0.feather`
3. `connectome-weights-male-cns-v1.0-minconf-0.5.feather`

Then prepare and run the complete model:

```bash
python -m venv .venv
# PowerShell: .\.venv\Scripts\Activate.ps1
# bash: source .venv/bin/activate
pip install -r requirements-server.txt
python scripts/prepare_full_data.py
uvicorn webapp.server:app --host 127.0.0.1 --port 8501
```

Open `http://127.0.0.1:8501`. Full preparation reads a large official table and
needs substantial RAM and disk.

## Docker

After data preparation, run `docker compose up --build`. The default mapping is
loopback-only (`127.0.0.1:8510`); use an authenticated reverse proxy or
Cloudflare Tunnel for public access, never a directly exposed app port.

## Anonymous visit counter

The public UI can show active sessions (seen within the previous 90 seconds)
and cumulative browser visitors. It creates random browser-local visitor and
session IDs, and the server stores only their SHA-256 hashes—never IP addresses,
account details, or browser metadata. Session records older than 30 days are
removed automatically.

## Validation

```bash
python scripts/validate_full_stack.py
node --check webapp/static/app.js
node --check webapp/static/nmf/game/game.js
```

## Attribution and license

Original glue code is MIT licensed. Bundled FlyGym / NeuroMechFly v2, MuJoCo,
and Three.js components retain their own licenses and notices. See
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) and
`webapp/static/nmf/NOTICE.txt`.

MaleCNS data are not bundled; cite the dataset and comply with CC BY 4.0 when
using the downloaded data or derived results.
