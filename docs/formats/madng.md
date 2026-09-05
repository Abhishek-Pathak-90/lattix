# MAD-NG (writer only)

`lattix/formats/madng/` writes a MAD-NG Lua lattice (`.madng`) by handing the xtrack `Line` lattix
builds to xtrack's own `mad_writer.to_madng_sequence` (xtrack 0.112.0).  There is no reader: MAD-NG
files are Lua programs, and reading them needs a Lua interpreter or MAD-NG itself (deferred; see
`PLAN_PHASE5.md` §5.1).

## Reading

Not supported.  `lattix.formats.base.FORMATS["madng"]` has `reader_attr=None`, so `lattix read`,
`lattix crossval` and the fixed-point tests skip the format; the CLI reports "no reader" for a
`.madng` input.

## Writing

```
$ lattix convert fodo.madx fodo.madng --to madng
```

The file is xtrack's output with a lattix header:

```lua
-- lattix 0.1.0 from madx (MAD-NG via xtrack 0.112.0)
-- lattix: energy_mode=delta
(function()	 -- Begin chunk
lquad = 0.3
kfocus = 0.6
…
seq_chunk_0 = bline 'seq_chunk_0' {
quadrupole 'qf' { l = 0.3, k1 = 0.6, …},
drift 'drift_0' { l = 0.7},
sbend 'b1' { l = 1.0, angle = 0.1, e1 = 0.05, e2 = 0.05, …},
…
fodo = sequence 'fodo' { refer='centre', seq_chunk_0 }
```

Every IR kind goes through the xtrack rules (`docs/formats/xtrack.md`), so the ledger carries the
xtrack rows (`CONST_P0_DELTA_RIGIDITY` for accelerating lattices, `FM_AS_CAVITY`, `NCELLS_TO_DRIFT`,
`FOIL_TO_MARKER`, …) plus one EQUIVALENT `VIA_XTRACK` row per element: the Lua text is xtrack's
rendering, not lattix's, and its element models (sector `sbend`, `rfcavity` with `lag` in turns,
`multipole`) are xtrack's MAD-X vocabulary.  Write options are the xtrack ones (`energy_mode`,
`install_apertures`, `name` for the sequence).  Variables become MAD-NG variables and deferred
expressions (`k2 =\ (k2bi4bsw1l11)`): the PS Booster's 128 knobs survive MAD-X → IR → MAD-NG.

## Units and conventions

MAD-NG uses MAD-X units (m, rad, MV, MHz, `lag` in turns) and xtrack's writer converts from the
`Line`; lattix adds nothing of its own.  The energy mode is the xtrack one (`delta` by default:
constant reference momentum, strengths normalized to the probe momentum, phase slips absorbed into
`lag`; `docs/conventions.md` §3).

## Known limits

* Writer only; no fixed point, so the format is absent from the cross-format battery (covered by
  `tests/formats/test_madng_writer.py` and the xtrack tests instead).
* Whatever xtrack's `mad_writer` cannot express (thick kickers become `multipole`s with
  `isthick`, `SecondOrderTaylorMap`s, apertures) is limited by xtrack, not by the IR; check
  the ledger's xtrack rows.
* `Line.to_madng` (running MAD-NG through `pymadng`) is not used: nothing here needs MAD-NG installed.

## Oracle

None: MAD-NG is not installed here and the format cannot be read back.  The physics of the text is
xtrack's, which is validated by `lattix/oracles/xtrack.py` on the same `Line` (PSB 0.0 vs
`from_madx_sequence`, `fodo.madx` vs cpymad 3.7e-9).
