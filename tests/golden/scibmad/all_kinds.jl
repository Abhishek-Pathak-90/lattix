# lattix <version> from IR (SciBmad / Beamlines.jl)
# lattix: energy_mode=delta
# lattix: reference species="proton" mass_eV=938272088.16 charge=1 kinetic_energy_eV=100000000
using Beamlines

# lattix directive: ADJUST 1 2
function lattix_map_ty(v, q, p=nothing)
    v1 = 1*v[1] + 0.491708750345107*v[2]
    v2 = 1*v[2]
    v3 = 1*v[3]
    v4 = 1*v[4]
    v5 = 1*v[5] + 0.245854375172554*v[6]
    v6 = 1*v[6]
    return (v1, v2, v3, v4, v5, v6), q
end

@elements begin
  dr = Drift(L = 0.4)
  qp = Quadrupole(L = 0.3, Kn1 = 0.6, tilt1 = 0.02)
  sx = Sextupole(L = 0.2, Kn2 = 1.5)
  oc = Octupole(L = 0.2, Kn3 = 2.5)
  mp = Multipole(Kn0L = 0.01, Kn2L = 0.3, Ks1L = 0.03)
  # lattix: name="bd" hgap="0.03"
  bd = SBend(L = 1, g_ref = 0.1, Kn0 = 0.1, e1 = 0.02, e2 = 0.03, edge1_int = 0.0135, edge2_int = 0.015, tilt_ref = 0.1)
  sl = Solenoid(L = 0.4, Ksol = 0.3)
  # lattix: name="cv" n_cell="5"
  cv = RFCavity(L = 0.5, voltage = -2000000, phi0 = -0.523598775598299, rf_frequency = 325000000)
  # lattix: name="fm" L_active_m="0.6" dE_ref_eV="1500000"
  fm = RFCavity(L = 0.6, voltage = -1732050.80756888, phi0 = -0.488617086777848, rf_frequency = 325000000)
  nc = Drift(L = 0.7)
  rq = Drift(L = 0.05)
  kk = Kicker(L = 0.1, Kn0L = -0.00101686211532553, Ks0L = -0.00203372423065107)
  # lattix: name="cl" kind="Collimator"
  cl = Drift(L = 0.1, x1_limit = -0.02, x2_limit = 0.02, y1_limit = -0.03, y2_limit = 0.03, aperture_shape = ApertureShape.Elliptical)
  mk = Marker()
  # lattix: name="bp" kind="Instrument" family="BPM"
  bp = Marker()
  # lattix: name="fl" kind="Foil" material="C" thickness_kg_per_m2="0.237"
  fl = Marker()
  ty = LineElement(L = 0.2, transport_map = lattix_map_ty)
  pt = Patch(dx = 0.001, dy_rot = 0.002, dz_rot = 0.3)
  # lattix: name="rc" kind="ReferenceChange" energy_eV="120000000"
  rc = Marker()
  # lattix: name="fq" kind="Freq" frequency_Hz="650000000"
  fq = Marker()
  sup_child = Drift(L = 0.2)
end

allkinds = Beamline([dr, qp, sx, oc, mp, bd, sl, cv, fm, nc, rq, kk, cl, mk, bp, fl, ty, pt, rc, fq, sup_child];
    pc_ref = 444583420.329639, species_ref = Species("proton"))
