# The browser UI (`lattix ui`)

`lattix ui` starts a small local server and opens a page that reads a deck, draws its beam line,
translates it to any writable format and draws the written deck read back through lattix, aligned with
the source element by element — the fidelity ledger painted on the beam line, a per-element diff of the
physical quantities in the inspector, and the engines a click away.  Everything runs on your machine:
the server binds `127.0.0.1` only, the page is one self-contained HTML file (no CDN, works offline) and
the URL carries a random access token.

```
$ lattix ui --root tests/data/public          # opens http://127.0.0.1:<port>/?token=…
$ lattix ui --deck path/to/linac.dat          # load a deck on start
$ lattix ui --no-browser --port 8765          # print the URL only
$ lattix ui --check --no-browser              # self-test the server and exit (CI)
```

Options: `--root DIR` (the directory the file picker may browse; default the current one), `--any-path`
(allow absolute paths outside the root), `--host` (loopback only), `--port` (0 = any free port),
`--no-browser`, `--deck PATH`, `--check`.  Stop it with Ctrl-C; its temporary directory (sessions,
written decks, side files) is removed.

## Load a deck

Give a path under the root (or browse with `Browse…`), pick one of the public sample decks from the
list, paste a path, or upload a folder with `Folder…` (a TraceWin deck with its field maps: the folder
structure is kept so `FIELD_MAP_PATH` resolves).  The format is detected from the suffix (and sniffed
for `.lat`/`.json`); override it in the `Format` list.  Readers that carry no beam take the species, the
kinetic energy and the RF frequency from the top bar (TraceWin, Elegant, MAD8 …); `Line` names the
sequence/line/root to use (mapped to each reader's own option); the reader's other options sit under
"Reader options" in the left panel.

## The synoptic

Elements are drawn along `s` (lengths to scale) with one glyph per kind and one colour per kind used by
every view (chips, inspector, floor plan):

| glyph | kind |
|---|---|
| thin spine on the baseline | Drift |
| bar above the baseline (focusing in x for the actual species, `k1 = G/Bρ_signed > 0`) or below (defocusing), height ∝ |k1|; diagonal hatch = skew; `+n` badge = higher orders | Quadrupole |
| smaller signed bars | Sextupole, Octupole |
| 6 px bar with the highest-order badge (`s` = skew) | Multipole |
| wedge (long edge on top for a positive angle, mirrored otherwise; parallelogram for a rectangular bend; `V` badge and vertical hatch for a vertical bend; `g` badge for a gradient bend) | Bend |
| full-height box with coil hatch | Solenoid |
| lozenge, height ∝ voltage (a thin gap: a 10 px lozenge; dotted outline when the phase is a driven phase; `→` for a travelling wave) | RFCavity, NCells (with cell ticks), RFQCell (dashed) |
| ellipse (RF map), box/bar with a dotted outline (static solenoid/quadrupole map), `?` badge when the map was not integrated | FieldMap |
| box or line with arrows for the kick direction (hollow circle: zero kick; `E`: electric) | Kicker |
| hollow box with `[ ]` brackets | Collimator |
| full-height line | Marker |
| tick with a family symbol (circle: BPM/monitor; square: size/profile/screen; diamond: current; triangle: phase/emittance) | Instrument |
| hatched bar | Foil |
| box with `M` (smaller when it is the RF focusing lens of a thin gap) | Taylor |
| broken-axis slashes | Patch |
| flag `E` / `f` | ReferenceChange, Freq |
| not drawn (listed in the chips, the inspector and the Fidelity tab) | Directive |

Hover for the name, kind, position and headline numbers; click to open the inspector; `←`/`→` walk
the lattice (Shift skips drifts), `Tab` jumps to the aligned element on the other row, `/` searches by
name, `f` fits, `+`/`−`/`0` zoom, `i` hides the inspector, `1`–`5` switch tabs, `Esc` clears.  Drag pans,
Ctrl/⌘+wheel zooms about the pointer (both rows stay aligned), double-click zooms in, Shift+drag zooms
into a box, Shift+click pins the `s` cursor.  The chips filter by kind (Shift-click to add); the minimap
under the axis scrolls the view.

## Translate

Pick the target format (grouped readable+writable / write-only / via Bmad), fill the writer options
generated from the writer's signature (common ones on top, the rest under "Advanced"; values are
remembered per format), tick `strict` to fail on the first LOSSY/DROPPED element, and press
`Translate`.  The deck is written to a session directory (side files such as `rfdataN`, `1TN.T7`,
`.fields.txt` or the OPAL maps next to it), read back through lattix, and drawn as the TARGET row.
Download the deck, the fidelity JSON, a zip of everything, or the source deck.

## Before and after

The target row's outlines carry the worst ledger class of the source element(s) each target element
came from — EQUIVALENT `≈` amber, LOSSY `!` red, DROPPED `×` grey (hatched on the *source* row, there
is no target element) — and a 4 px strip under the row shows where the losses are at any zoom.
Elements the writer added (padding drifts, the `_rfdefocus` lens of a thin gap) are dotted.  Selecting
an element highlights its counterparts and draws connectors between the rows.

The inspector's *Translation* section shows how the element was matched (by name — through the
writers' naming conventions for split, padded and folded elements — or by position), the target
elements of its cluster with their roles, and a table of every physical quantity lattix conserves
(`length`, bend angles, `BnL`/`BsL` up to decapole, `BsolL`, `hkick`/`vkick`, `gain`, `volt`, the exit
energy and the entrance position): **equal** within 1e-9, **explained** when a LOSSY/DROPPED entry
of the ledger names that quantity (`docs/crossval.md`: the `AFFECTS` table) or a thin element was
written over a short surrogate length, **suspended** when the target cannot name the species, or
**unexplained**.  An unexplained difference escalates the element to the synthetic class **DIFF** `⚠`
(magenta): a difference the ledger does not account for — a translation defect worth reporting.  The
summary chips count the classes; the header of the target row states the battery's own round-trip
verdict (`lattix crossval`'s check (a), computed by the same code).

## Floor plan

The survey top view: path `Z` to the right, horizontal `X` up (a positive bend turns clockwise, MAD-X's
convention), arcs for bends, strokes in the kind colours, source solid and target dashed; pan, zoom,
click and hover as on the synoptic.

## Deck text and Fidelity

The Deck text tab shows the source deck or the written deck with line numbers; the selected element's
statement is highlighted (found through lattix's `name=` tags, the reader's line numbers, or the
identifier) and clicking a line selects its element.  The Fidelity tab lists the ledger by code with the
elements concerned and the catalogue meaning of each code (`docs/fidelity.md`).

## Validate with engines

The Validation tab lists the engine adapters lattix found (`lattix oracles`), pre-selects the ones that
read the source and the written deck, and runs `lattix validate` **in a separate process** (an
in-process engine such as HELIX or ImpactX never shares the server): the job log streams in, the per-pair
summary rows, the battery's verdict (which map blocks the pair can be held to, the tier's tolerance, the
engine-precision floor, the measured caveats of `docs/oracles.md`) and the per-block error-versus-`s`
chart appear when it finishes.  The beam form is prefilled from the deck; the Twiss values only matter
for envelope engines.  Engines that need a local build (TraceWin, DYNAC, Synergia, HELIX) are marked.

## Security and limits

Loopback only; a random token in the URL (kept by the page, sent as a header) and a `Host`/`Origin`
check refuse other pages and DNS-rebinding; file access is confined to `--root` unless `--any-path`;
request bodies are limited to 64 MiB; sessions live in a temporary directory removed on exit (16 kept);
one engine job runs at a time with a 30 min limit.

## API

| route | purpose |
|---|---|
| `GET /api/ping`, `/api/formats`, `/api/samples`, `/api/catalog`, `/api/oracles`, `/api/browse?path=` | catalogues and the file picker |
| `POST /api/session`, `PUT /api/session/{sid}/upload?path=` | an empty session and raw-body uploads (folders keep their paths) |
| `POST /api/read` | `{source: {sample}|{path}|{content, filename}, format?, options}` → the source view model |
| `POST /api/translate` | `{session, format, options, strict, reread}` → deck text, side files, ledger, the target view model, the comparison |
| `GET /api/session/{sid}[/translation/{tid}]`, `…/download?what=source|deck|fidelity|zip|side&name=`, `…/source_text` | resume, files |
| `POST /api/validate`, `GET /api/jobs/{jid}`, `POST /api/jobs/{jid}/cancel` | the engine job |

The view model of a lattice (`lattix.ui.model.lattice_view`) is placed rows (`s`, energies, glyph hints,
contributions, derived numbers, survey) plus deduplicated definitions and the ledger; the comparison
(`lattix.ui.compare.compare_translation`) is the alignment, the per-element diffs and the summary.
