# Born Approximation Derivation

## Forward Problem

We start with the Helmholtz equation for the total field $u(x)$ in a medium with squared slowness $m(x)$:

$$
-\Delta u - \omega^2 m(x) u = 0
$$

Let the model be a perturbation of a background model $m_0(x) = 1$:

$$
m(x) = 1 + \eta(x)
$$

The total field $u$ can be decomposed into an incident field $u_{in}$ (satisfying the equation for $m_0=1$) and a scattered field $u_{sc}$:

$$
u = u_{in} + u_{sc}
$$

Substituting this into the wave equation:

$$
(-\Delta - \omega^2 (1 + \eta)) (u_{in} + u_{sc}) = 0
$$

Expanding:

$$
(-\Delta - \omega^2) u_{in} + (-\Delta - \omega^2) u_{sc} - \omega^2 \eta (u_{in} + u_{sc}) = 0
$$

Since $(-\Delta - \omega^2) u_{in} = 0$, we have:

$$
(-\Delta - \omega^2) u_{sc} = \omega^2 \eta (u_{in} + u_{sc})
$$

This is an implicit equation for $u_{sc}$ because it appears on the RHS.

## First-Order Born Approximation

For small perturbations $\eta \ll 1$, we assume the scattered field is small compared to the incident field ($|u_{sc}| \ll |u_{in}|$). Thus, we can approximate the total field in the source term as $u \approx u_{in}$.

$$
(-\Delta - \omega^2) u_{sc} \approx \omega^2 \eta u_{in}
$$

Let $G_0$ be the Green's function for the background operator $-\Delta - \omega^2$. The scattered field is then:

$$
u_{sc} \approx G_0 (\omega^2 \eta u_{in})
$$

This linear relation between the perturbation $\eta$ and the scattered field $u_{sc}$ is the **Born Approximation**.

## Derivation of the Adjoint (Gradient)

In our inverse problem, we define the forward operator $F(\eta)$ as the map from $\eta$ to the sampled scattered field at receiver locations $x_r$.

$$
F(\eta) = P u_{sc}(\eta)
$$

where $P$ is the projection/sampling operator.

We minimize the misfit functional:

$$
J(\eta) = \frac{1}{2} \| F(\eta) - d_{obs} \|^2
$$

The gradient of $J$ with respect to $\eta$ requires the adjoint of the Jacobian of $F$.
Let $r = F(\eta) - d_{obs}$ be the residual.
The gradient is $g = F'(\eta)^* r$.

From the state equation (without approximation for full FWI):
$A(m) u_{sc} = S(\eta) = \omega^2 \eta u_{in}$.
Linearizing w.r.t $\eta$ (denoted by $\delta \eta$):

$$
A \delta u_{sc} \approx \omega^2 \delta \eta u_{in}
$$
(assuming $u_{in}$ is fixed and ignoring the secondary scattering $\eta \delta u_{sc}$ term for the gradient of the standard Born/FWI misfit).

Thus $\delta u_{sc} = A^{-1} (\omega^2 u_{in} \delta \eta)$.
The Jacobian acting on $\delta \eta$ is:
$$
J \delta \eta = P A^{-1} (\omega^2 u_{in} \delta \eta)
$$

The adjoint $J^*$ acting on a residual vector $w$ (in data space) is defined by:
$$
\langle J \delta \eta, w \rangle_{data} = \langle \delta \eta, J^* w \rangle_{model}
$$

$$
\langle P A^{-1} (\omega^2 u_{in} \delta \eta), w \rangle = \langle A^{-1} (\omega^2 u_{in} \delta \eta), P^* w \rangle
$$
$$
= \langle \omega^2 u_{in} \delta \eta, (A^{-1})^* P^* w \rangle
$$
$$
= \langle \delta \eta, \omega^2 \overline{u_{in}} (A^*)^{-1} P^* w \rangle
$$

So the gradient/adjoint is:
$$
g = J^* w = \omega^2 \overline{u_{in}} (A^*)^{-1} P^* w
$$

Let $v = (A^*)^{-1} P^* w$ be the **adjoint field** (back-propagated residual).
Then:
$$
g = \omega^2 \overline{u_{in}} v
$$

Note on signs:
If the source term is defined as $S = \omega^2 \eta u_{in}$ (positive), the gradient is positive.
If defined as $S = -\omega^2 \eta u_{in}$ (standard physics), the gradient has a negative sign ($-\omega^2 \bar{u} v$).
Our code implementation uses the convention $(\Delta + \omega^2) u = S$, with $S = -\omega^2 \eta u_{in}$ (negative source).
Consequently, the adjoint/gradient is computed with a negative sign to ensure correct descent direction.
