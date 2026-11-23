import numpy as np, mujoco

ARM = ["arm_root", "arm_shoulder", "arm_elbow", "arm_wrist"]

def _quat2mat(q):
    m = np.zeros(9); mujoco.mju_quat2Mat(m, np.asarray(q, float)); return m.reshape(3, 3)

def _skew(a):
    return np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]], float)
    
def _cross(a, b):
    return np.array([a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]])

class AnalyticalArm:
    def __init__(self, model, ref_qpos=None, g=None):
        d = mujoco.MjData(model)
        d.qpos[:] = model.qpos0 if ref_qpos is None else ref_qpos
        mujoco.mj_forward(model, d)
        self.g = -model.opt.gravity[2]                      # gravity assumed along -z
        self.jid = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in ARM]
        self.dofs = [model.jnt_dofadr[j] for j in self.jid]
        self.chain = [model.jnt_bodyid[j] for j in self.jid]       # assumed serial
        self.axis = [np.array(model.jnt_axis[j], float) for j in self.jid]
        self.K = [_skew(a / np.linalg.norm(a)) for a in self.axis]
        self.jpos = [np.array(model.jnt_pos[j], float) for j in self.jid]
        self.boff = [(np.array(model.body_pos[b], float), _quat2mat(model.body_quat[b]))
                     for b in self.chain]
        self.damp = np.array(model.dof_damping[self.dofs], float)
        self.arma = np.array(model.dof_armature[self.dofs], float)
        self.bodies = []                                           # (owner_idx, m, com_rel, I_rel)
        for b in range(1, model.nbody):
            a = b
            while a not in self.chain and a != 0:
                a = model.body_parentid[a]
            if a == 0:
                continue                                           # ball, target, ...
            o = self.chain.index(a); Ro = d.xmat[a].reshape(3, 3)
            Ri = Ro.T @ d.ximat[b].reshape(3, 3)
            I = Ri @ np.diag(model.body_inertia[b]) @ Ri.T
            self.bodies.append((o, float(model.body_mass[b]),
                                Ro.T @ (d.xipos[b] - d.xpos[a]), I))
                 # lump rigid bodies per owner link (exact: fingers are locked)
        lumped = []
        for o in range(4):
            bs = [b for b in self.bodies if b[0] == o]
            mt = sum(b[1] for b in bs)
            com = sum(b[1] * b[2] for b in bs) / mt
            I = np.zeros((3, 3))
            for _, mi, ci, Ii in bs:
                dv = ci - com
                I += Ii + mi * ((dv @ dv) * np.eye(3) - np.outer(dv, dv))
            lumped.append((o, mt, com, I))
        self.bodies = lumped

    def _fk(self, q):
        dt = q.dtype
        R = np.eye(3, dtype=dt); p = np.zeros(3, dtype=dt)
        frames, axes, jpts = [], [], []
        for k in range(4):
            p = p + R @ self.boff[k][0]; R = R @ self.boff[k][1]
            axes.append(R @ self.axis[k]); jpts.append(p + R @ self.jpos[k])
            Rj = np.eye(3) + np.sin(q[k]) * self.K[k] + (1 - np.cos(q[k])) * (self.K[k] @ self.K[k])
            p = p + R @ (self.jpos[k] - Rj @ self.jpos[k]); R = R @ Rj
            frames.append((p, R))
        return frames, axes, jpts

    def _terms(self, q):
        frames, axes, jpts = self._fk(q)
        zero = np.zeros(3, dtype=q.dtype); out = []
        for (o, m, c, I) in self.bodies:
            p, R = frames[o]
            com = p + R @ c
            Jv = np.stack([np.cross(axes[j], com - jpts[j]) if j <= o else zero for j in range(4)], 1)
            Jw = np.stack([axes[j] if j <= o else zero for j in range(4)], 1)
            out.append((m, Jv, Jw, R @ I @ R.T))
        return out

    def _M(self, q):                      # works for real or complex q (no conjugates)
        M = np.diag(self.arma).astype(q.dtype)
        for m, Jv, Jw, Iw in self._terms(q):
            M = M + m * Jv.T @ Jv + Jw.T @ Iw @ Jw
        return M

    def M(self, q):
        return self._M(np.asarray(q, float))

    def G(self, q):
        q = np.asarray(q, float); G = np.zeros(4)
        for m, Jv, Jw, Iw in self._terms(q):
            G = G + m * self.g * Jv[2, :]                          # V = sum m g z
        return G

    def dM(self, q):                      # dM[k,j,i] = dM_kj/dq_i via complex step (exact)
        q = np.asarray(q, float); h = 1e-20; out = np.zeros((4, 4, 4))
        for i in range(4):
            qc = q.astype(complex); qc[i] += 1j * h
            out[:, :, i] = self._M(qc).imag / h
        return out

    def C(self, q, qd):
        dM = self.dM(q); qd = np.asarray(qd, float)
        c = 0.5 * (dM + dM.transpose(0, 2, 1) - dM.transpose(2, 0, 1))   # Christoffel c_kji
        return np.einsum('kji,i->kj', c, qd)

    def V(self, q):
        q = np.asarray(q, float); frames, _, _ = self._fk(q)
        return sum(m * self.g * (frames[o][0] + frames[o][1] @ c)[2]
                   for (o, m, c, I) in self.bodies)

    # --- interface for Members 3-5 (numpy in, numpy out) ---
    def inverse_dynamics(self, q, qd, qdd):
        q, qd, qdd = (np.asarray(x, float) for x in (q, qd, qdd))
        return self.M(q) @ qdd + self.C(q, qd) @ qd + self.G(q) + self.damp * qd

    def forward_dynamics(self, q, qd, tau):
        q, qd, tau = (np.asarray(x, float) for x in (q, qd, tau))
        rhs = tau - self.C(q, qd) @ qd - self.G(q) - self.damp * qd
        return np.linalg.solve(self.M(q), rhs)
