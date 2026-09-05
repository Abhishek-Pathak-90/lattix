# SciBmad worker for lattix.oracles.scibmad: loads a Beamlines.jl lattice file, measures the
# per-element linear maps by central finite differences around the *tracked* reference orbit
# (SciBmad keeps one reference momentum per beamline, so a cavity's gain lives in the orbit's pz),
# and writes a JSON result.
#
#   julia --startup-file=no scibmad_worker.jl DECK --out result.json [--line NAME] [--fd 1e-6]
#
# Engine limits substituted here (each one is reported in "warnings"):
#   * a zero-length RFCavity cannot be tracked by BeamTracking 0.5 -> tracked as 1 µm
#   * edge1_int/edge2_int ("not yet handled for tracking") -> tracked as 0
using SciBmad

# minimal JSON writer (JSON.jl is not a direct dependency of a SciBmad environment)
jstr(s::AbstractString) = "\"" * replace(replace(replace(String(s), "\\" => "\\\\"), "\"" => "\\\""), "\n" => "\\n") * "\""
jnum(x::Real) = (isfinite(x) ? repr(Float64(x)) : "null")
jval(x::AbstractString) = jstr(x)
jval(x::Bool) = x ? "true" : "false"
jval(x::Integer) = string(x)
jval(x::Real) = jnum(x)
jval(x::Nothing) = "null"
jval(x::AbstractVector) = "[" * join((jval(v) for v in x), ",") * "]"
jval(x::AbstractDict) = "{" * join((jstr(String(k)) * ":" * jval(v) for (k, v) in x), ",") * "}"

function parse_args(args)
    opts = Dict{String,Any}("fd" => 1e-6, "line" => nothing, "out" => nothing, "deck" => nothing)
    i = 1
    while i <= length(args)
        a = args[i]
        if a == "--out"; opts["out"] = args[i+1]; i += 2
        elseif a == "--line"; opts["line"] = args[i+1]; i += 2
        elseif a == "--fd"; opts["fd"] = parse(Float64, args[i+1]); i += 2
        else; opts["deck"] = a; i += 1
        end
    end
    return opts
end

opts = parse_args(ARGS)
deck = opts["deck"]
text = read(deck, String)
# the file's own `using …` lines: Beamlines is already in scope through SciBmad (and a bare
# `using Beamlines` fails in an environment that only has the umbrella package)
code = join(filter(l -> !occursin(r"^\s*(using|import)\s", l), split(text, '\n')), '\n')
# the root beamline: --line, else the last `name = Beamline(` assignment in the file
root_name = opts["line"]
if root_name === nothing
    for m in eachmatch(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*Beamline\("m, text)
        global root_name = m.captures[1]
    end
end
root_name === nothing && error("no `name = Beamline(` in $deck")

mod = Module(:LattixDeck)
Core.eval(mod, :(using SciBmad))
Base.include_string(mod, code, deck)
bl = Core.eval(mod, Symbol(root_name))
bl isa Beamline || error("$root_name is not a Beamline")

warnings = String[]
sp = bl.species_ref
mass = massof(sp)
charge = chargeof(sp)
E_ref = bl.E_ref
Ek = E_ref - mass

eles = collect(bl.line)
n = length(eles)
names = String[]
kinds = String[]
lengths = Float64[]
s_out = Float64[]
for e in eles
    push!(names, String(e.name))
    push!(kinds, String(e.kind))
    push!(lengths, Float64(e.L))
    push!(s_out, Float64(e.s) + Float64(e.L))
    if e.kind == "RFCavity" && e.L == 0
        e.L = 1e-6
        push!(warnings, "$(e.name): zero-length RFCavity tracked as 1e-6 m (BeamTracking 0.5 has no thin RF kernel)")
    end
    if e.kind == "SBend"
        try
            if e.edge1_int != 0 || e.edge2_int != 0
                e.edge1_int = 0.0
                e.edge2_int = 0.0
                push!(warnings, "$(e.name): edge1_int/edge2_int set to 0 for tracking (not yet handled by BeamTracking 0.5)")
            end
        catch
        end
    end
end

# 1. the reference orbit: a zero particle tracked element by element
centre = zeros(n, 6)
ref = Bunch(zeros(1, 6); p_over_q_ref=bl.p_over_q_ref, species=sp)
lost = false
for (i, e) in enumerate(eles)
    centre[i, :] = ref.coords.v[1, :]
    lost && continue
    try
        track!(ref, e)
    catch err
        push!(warnings, "reference particle failed in $(e.name): $(sprint(showerror, err))")
        global lost = true
    end
    if any(ref.coords.state .!= 1)
        push!(warnings, "reference particle lost at $(e.name)")
        global lost = true
    end
end

# 2. per-element maps by central differences around that orbit
h = opts["fd"]
R = fill(NaN, n, 36)
for (i, e) in enumerate(eles)
    v = zeros(12, 6)
    for k in 1:6
        v[2k-1, :] = centre[i, :]; v[2k-1, k] += h
        v[2k, :] = centre[i, :];   v[2k, k] -= h
    end
    b = Bunch(v; p_over_q_ref=bl.p_over_q_ref, species=sp)
    try
        track!(b, e)
    catch err
        push!(warnings, "$(e.name): tracking failed: $(sprint(showerror, err))")
        continue
    end
    if any(b.coords.state .!= 1)
        push!(warnings, "$(e.name): finite-difference particles lost")
        continue
    end
    vo = b.coords.v
    M = zeros(6, 6)
    for k in 1:6
        M[:, k] = (vo[2k-1, :] .- vo[2k, :]) ./ (2h)
    end
    R[i, :] = vec(permutedims(M))       # row-major 6x6
end

result = Dict(
    "names" => names, "kinds" => kinds, "length" => lengths, "s_out" => s_out,
    "mat6" => [R[i, :] for i in 1:n],
    "e_tot_in" => fill(E_ref, n), "e_tot_out" => fill(E_ref, n),
    "mass_ev" => mass, "charge" => charge, "species" => String(nameof(sp)),
    "p_over_q_ref" => bl.p_over_q_ref, "root" => root_name,
    "warnings" => warnings,
    "scibmad_version" => string(pkgversion(SciBmad)),
    "julia_version" => string(VERSION),
    "p0_model" => "constant: one reference momentum per beamline; the orbit's pz carries the RF gain",
    "basis" => "bmad",
)
open(opts["out"], "w") do io
    write(io, jval(result))
end
