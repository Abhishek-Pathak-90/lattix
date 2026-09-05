# lattix <version> from madx (SciBmad / Beamlines.jl)
# lattix: energy_mode=delta
# lattix: reference species="proton" mass_eV=938272088.16 charge=1 kinetic_energy_eV=799999911.84
using Beamlines

@elements begin
  qf = Quadrupole(L = 0.3, Kn1 = 0.6)
  drift_0 = Drift(L = 0.7)
  b1 = SBend(L = 1, g_ref = 0.1, Kn0 = 0.1, e1 = 0.05, e2 = 0.05)
  drift_1 = Drift(L = 1)
  qd = Quadrupole(L = 0.3, Kn1 = -0.6)
  drift_2 = Drift(L = 1)
  drift_3 = Drift(L = 1.3)
  m1 = Marker()
end

fodo = Beamline([qf, drift_0, b1, drift_1, qd, drift_2, b1, drift_3, m1];
    pc_ref = 1463295949.06973, species_ref = Species("proton"))
