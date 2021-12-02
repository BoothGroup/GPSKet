import netket as nk
import qGPSKet.operator.hamiltonian.J1J2 as j1j2
import qGPSKet.models.qGPS as qGPS

ha = j1j2.get_J1_J2_Hamiltonian(20)
hi = ha.hilbert
g = ha.graph

sa = nk.sampler.MetropolisExchange(hi, graph=g, n_chains=1)

op = nk.optimizer.Sgd(learning_rate=0.02)


qgt = nk.optimizer.qgt.QGTJacobianDense(holomorphic=True)
sr = nk.optimizer.SR(qgt=qgt)

model = qGPS.qGPS(2, apply_symmetries=qGPS.get_sym_transformation(g))

vs = nk.vqs.MCState(sa, model, n_samples=1000)

gs = nk.VMC(ha, op, variational_state=vs, preconditioner=sr)

for it in gs.iter(1000,1):
    print(it,gs.energy, flush=True)

