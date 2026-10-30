<!-- SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers -->
<!-- SPDX-License-Identifier: CC-BY-4.0 -->

# UIPC Integration

`SolverUIPC` wraps `libuipc` (the `uipc` / `pyuipc` Python package) behind
Newton's standard solver interface and requires `pyuipc>=0.0.29`.
UIPC provides implicit IPC-style contact
resolution for rigid bodies, cloth, and deformable bodies. Newton converts a
`Model` into a UIPC `Scene` made of AffineBody rigid objects, shell cloth
objects, tetrahedral deformables, contact-tabular entries, and optional
subscenes for multi-world isolation.

Because UIPC has its own scene graph and constitutions, many Newton properties
are baked into UIPC objects during initialization. The sections below describe
what the solver supports, how state and controls are synchronized, where contact
configuration lives, and which runtime updates can be pushed into an already
initialized UIPC world. For the per-parameter mapping table, see the
[UIPC parameter guide](../guide/uipc_parameters.md).

> **Note**
> `SolverUIPC` uses a fixed time step. Construct the solver with the `dt` you
> intend to simulate, and pass the same value to `step(..., dt=dt)`.

## Live viewer statistics

Examples run with `newton.examples.run(...)` automatically show UIPC statistics
in the top-right performance overlay of `ViewerGL` and `ViewerRTX`:

- **Step time**: host wall time in milliseconds for the last `SolverUIPC.step`,
  including UIPC advance/retrieve and Newton state transfers. Deferred
  initialization is excluded; the measurement adds no device synchronization.
- **Newton iters**: actual nonlinear iterations in that UIPC step.
- **Max PCG**: maximum actual PCG iterations among that step's linear solves.
  This is neither the total PCG iterations nor the configured iteration limit.

With several substeps per rendered frame, these values describe the last
substep. Pausing keeps the last result visible; switching examples follows
the new example's `solver`. Counters do not require `require_profile=True`.
Older UIPC builds without `linear_solver_max_iterations` show `Max PCG: N/A`.

Custom applications can read the same snapshot with
{meth}`~newton.solvers.SolverUIPC.get_frame_stats` and render it through
`viewer.register_ui_callback(callback, position="stats")`.

## Basic workflow

`SolverUIPC` supports deferred initialization. This lets callers customize the
scene configuration, contact table, subscene table, and inertia bridge before
UIPC finalizes the world:

```python
import newton
from newton.solvers import SolverUIPC

builder = newton.ModelBuilder()
SolverUIPC.register_custom_attributes(builder)
# ...build or import a model...
model = builder.finalize()

dt = 1.0 / 60.0
solver = SolverUIPC(model, dt=dt, workspace="/tmp/newton_uipc")

# UIPC contact is disabled by default in Newton's scene config.
solver.set_contact(True, d_hat=0.001)

# Optional: tune UIPC's JSON-like scene config before initialize().
solver.configure_scene(
    {
        "newton": {"velocity_tol": 1.0e-3},
        "line_search": {"max_iter": 8},
    }
)


# Optional: customize contact pairs before initialize().
def setup_contacts(tabular, world_index, ground_elem, env_elem, robo_elem, actor_elem):
    gripper_elem = tabular.create(f"gripper_{world_index}")
    tabular.insert(gripper_elem, actor_elem, 0.8, 1.0e9, True)
    tabular.insert(gripper_elem, ground_elem, 0.8, 1.0e9, True)


solver.configure_contact_tabular(setup_contacts)
solver.initialize()

state_in = model.state()
state_out = model.state()
control = model.control()
# Allocate reporting buffers; UIPC performs collision detection internally.
contacts = newton.Contacts(solver.get_max_contact_count(), 0, device=model.device)
observables = solver.observables({solver.ObservableKind.CONTACT_F}, contacts=contacts)

for _ in range(100):
    solver.step(state_in, state_out, control, contacts, dt, observables=observables)
    state_in, state_out = state_out, state_in
```

Calling `step` before `initialize` implicitly initializes the solver, but
explicit initialization is recommended whenever you use `configure_scene`,
`configure_contact_tabular`, `configure_subscene_tabular`, or
`sync_uipc_inertia_with_model`.

## External forces

Each `step` consumes `state_in.body_f` and `state_in.particle_f` as external
force inputs. `body_f` contains a world-space force [N] in its first three
components and a world-space torque [N·m] about the body's center of mass in
its last three components. The force acts at the COM. `particle_f` contains
world-space forces [N] on cloth and tetrahedral deformable vertices.
These inputs use UIPC's `AffineBodyExternalBodyForce` and
`FiniteElementExternalForce` constitutions, respectively.

Use Newton body and particle indices, including in models with different
numbers of objects per world. Instanced bodies and geometry-free articulation
proxies receive their own wrenches. Loose particles outside UIPC FEM meshes
are ignored, and fixed bodies or vertices do not respond to applied forces.

```python
state_in.clear_forces()
# Accumulate or assign this step's world-space forces to body_f / particle_f.
solver.step(state_in, state_out, control, dt=dt)
state_in, state_out = state_out, state_in
```

The inputs are refreshed every step. Zero entries or an absent force array
disable the corresponding UIPC external force, so previous loads do not
persist. A separate output state's force arrays are cleared after stepping;
the input arrays are preserved. In-place stepping also consumes the inputs
before clearing the force arrays.

The solver registers the native external-force constitutions once during
initialization and updates their values and `external_force/is_constrained`
masks each step. These masks are independent of the shared `is_constrained`
mask used by soft position and transform constraints. Joint efforts in
`Control.joint_f` remain additive to the State force inputs. Contact-force
reports continue to use observables, as described below.

## Joint types

| Newton type | UIPC equivalent | Notes |
| --- | --- | --- |
| `REVOLUTE` | `AffineBodyRevoluteJoint` + driving / external-torque / limit constitutions | Active joint. `Control.joint_target` can drive `aim_angle`; UIPC's `angle` edge attribute is read back into Newton `joint_q`. |
| `PRISMATIC` | `AffineBodyPrismaticJoint` + driving / external-force / limit constitutions | Active joint. `Control.joint_target` can drive `aim_distance`; UIPC's `distance` edge attribute is read back into Newton `joint_q`. |
| `FIXED` | `AffineBodyFixedJoint`, or fixed child instance for world anchors | A fixed joint to world marks the child instance as fixed. Inter-body fixed joints are emitted as batched UIPC fixed-joint geometry. |
| `FREE` | AffineBody object with `SoftTransformConstraint` | The body is simulated as a free actor. Body pose / velocity are synchronized through Newton `body_q` / `body_qd`, not through an active generalized-coordinate joint. |
| `BALL` | `AffineBodySphericalJoint` | Constrains parent and child anchors, but is not an active driven/read-back joint in Newton's UIPC articulation state. |
| `DISTANCE`, `D6` | *unsupported* | Skipped with a warning during UIPC joint construction. |
| `ROD` | `HookeanSpring` + optional `KirchhoffRodBending` | Centerline approximation with stretch and isotropic bending; see [Rods](#rods). |

For active `REVOLUTE` and `PRISMATIC` joints, parent/child anchors are
validated in world space before UIPC geometry is created. Revolute anchors and
axis endpoints must coincide; prismatic axes must be parallel and their anchors
must be collinear. Invalid authored joint frames raise at initialization rather
than failing later inside UIPC.

### Mimic joint coupling

Mimic constraints (`Model.constraint_mimic_*`, imported from URDF `<mimic>`,
MJCF joint equalities, or USD `PhysxMimicJointAPI` / `NewtonMimicAPI`) enforce
`joint0 = coef0 + coef1 * joint1` (follower = offset + scale × leader). Each
step, before `world.advance()`, the follower's position target is set to
`coef0 + coef1 * q_leader` and the follower is forced into position-driving
mode, so the UIPC animator drives it toward the coupled target. The leader value
is its commanded target when the leader is itself position-driven (no lag),
otherwise its measured start-of-step position (one-step lag).

The coupling is **soft**: the follower tracks its target through the driving
joint's `drive_strength_ratio` and may lag under load, unlike a hard equality
constraint. Both follower and leader must be active
`REVOLUTE` / `PRISMATIC` joints; a constraint whose follower or leader is not an
active UIPC joint is skipped with a warning. Coefficients are baked at
initialization — editing them requires reconstructing the solver.

## Geometry types

Rigid bodies are represented as UIPC AffineBody geometries. Newton gathers each
body's colliding shapes, converts them into one closed triangle mesh, and
applies `AffineBodyConstitution`. Only shapes with the `COLLIDE_SHAPES` flag
contribute to UIPC collision geometry.

| Newton type | UIPC equivalent | Notes |
| --- | --- | --- |
| `BOX` | Closed triangle mesh | Generated from Newton primitive dimensions. |
| `SPHERE` | Closed triangle mesh | Generated from Newton primitive dimensions. |
| `CAPSULE` | Closed triangle mesh | Capsule axis follows `model.up_axis`. |
| `CYLINDER` | Closed triangle mesh | Cylinder axis follows `model.up_axis`. |
| `CONE` | Closed triangle mesh | Cone axis follows `model.up_axis`. |
| `MESH` / `CONVEX_MESH` | Triangle mesh from the shape source | The merged body mesh must be closed and have positive volume. Non-watertight meshes raise with a suggestion to call `ModelBuilder.approximate_meshes`. Near-zero-volume meshes fall back to an AABB box. |
| `PLANE` | `uipc.geometry.halfplane` for `shape_body == -1` | World-attached planes become infinite UIPC halfplanes. Body-attached planes do not contribute to AffineBody meshes. |
| `HFIELD`, `ELLIPSOID`, `GAUSSIAN`, `NONE` | *unsupported* | Ignored by the UIPC rigid-body mesh builder, with warnings for unsupported geometry where applicable. |

World-space non-plane shapes (`shape_body == -1`) default to fixed static
colliders with UIPC's `Empty` constitution. They may explicitly select `"abd"`
to use fixed AffineBody instances. Different constitutions are built as
separate static geometries. Ground planes share UIPC's default
ground contact element; other static colliders use the per-world environment
contact element.

Newton's SDF, hydroelastic, and other Newton-native collision pipelines are not
fed into UIPC. `SolverUIPC` computes contact internally through UIPC's contact
system.

## Rigid bodies and inertia

Dynamic rigid bodies use UIPC's AffineBody formulation. Newton forwards:

- the initial `body_q` / `body_qd` transforms and velocities,
- kinematic flags as UIPC `is_fixed` instance attributes,
- mesh-derived or authored mass density,
- the solver-level AffineBody stiffness `kappa`, with optional per-body
  overrides through the `uipc:abd_kappa` custom attribute.

The experimental `NewtonUIPCRigidBodyAPI` declares a `uipc:constitution`
token on a rigid-body prim or a world-static collider:

```python
rigid_prim.ApplyAPI("NewtonUIPCRigidBodyAPI")
rigid_prim.GetAttribute("uipc:constitution").Set("abd")

static_collider_prim.ApplyAPI("NewtonUIPCRigidBodyAPI")
static_collider_prim.GetAttribute("uipc:constitution").Set("empty")
```

Register UIPC custom attributes before `builder.add_usd(...)` to import the
selection. On a rigid-body prim the token populates `model.uipc.rigid_model`;
on a collider it populates `model.uipc.shape_constitution`. An unauthored
selection defaults to ABD for bodies and Empty for world-static colliders.
World-static colliders support both `"abd"` and `"empty"`, with the appropriate
instance or vertex fixed markers. A rigid body and its owned colliders must
use ABD, including kinematic or joint-connected bodies; selecting Empty raises
an error. Represent a fixed Empty collider without `PhysicsRigidBodyAPI`.
Planes use implicit halfplanes and reject explicit constitution selection.

By default, `initialize` syncs UIPC's final AffineBody mass, center of mass, and
inertia back into the Newton `Model` via `sync_model_inertia_from_uipc`. This
keeps host-side Newton utilities, such as mass-matrix evaluation for Stable PD,
consistent with the body properties UIPC actually uses.

If you need UIPC to preserve Newton-authored inertial values instead, call
`sync_uipc_inertia_with_model(body_indices)` before `initialize`. To keep
Newton-authored values untouched after initialization, construct the solver with
`auto_sync_inertia=False`.

### Joint armature

Positive `Model.joint_armature` on revolute and prismatic joints creates an
`ExternalArticulationConstraint` with a diagonal joint-space mass matrix.
Its kinetic potential penalizes changes from the previous joint increment,
so armature works in every target mode without changing the child body's
mass or inertia.

For a revolute joint in `EFFORT` mode, torque is included in this same
potential by shifting the predicted increment by `dt**2 * torque / armature`.
Completing the square gives the joint work term `-torque * delta_angle`.
The separate affine external-torque channel is disabled for this joint to
avoid double application. Using a common joint coordinate for torque and
reflected inertia avoids artificial parent rotation from a frozen affine
torque Jacobian during large angular increments.

Revolute joints without positive armature and prismatic efforts retain the
native external torque/force channel. Runtime armature edits on existing
constraint edges use `notify_model_changed(ModelFlags.JOINT_DOF_PROPERTIES)`;
setting a revolute armature to zero restores the native torque channel.

(uipc-custom-attributes)=
## UIPC-specific custom attributes

`SolverUIPC.register_custom_attributes` registers Newton's UIPC namespace on a
`ModelBuilder`. Call it before :meth:`~newton.ModelBuilder.finalize` when the
model contains cloth or soft bodies. Registering immediately after builder
construction is recommended:

```python
import newton
from newton.solvers import SolverUIPC

builder = newton.ModelBuilder()
SolverUIPC.register_custom_attributes(builder)
# ...add bodies...
```

Currently registered:

| Attribute | Frequency | Default | Meaning |
| --- | --- | --- | --- |
| `model.uipc.abd_kappa` | Body | `-1.0` | Per-body AffineBody stiffness override [Pa]. Negative values inherit the solver-level `kappa`. |
| `model.uipc.rigid_model` | Body | `"abd"` | Rigid-body constitution, imported from body-level `uipc:constitution`. Only ABD preserves rigid-body state and joints; Empty is rejected. |
| `model.uipc.shape_constitution` | Shape | `""` | Collider-level `uipc:constitution`. World-static colliders support `"abd"` or `"empty"`; an empty string inherits ABD from an owning body or selects Empty for a world-static collider. |
| `model.uipc.cloth_model` | `uipc:cloth` | `"strain_limiting_baraff_witkin"` | Per-authored-cloth membrane constitution, imported from simulation-mesh `uipc:constitution`. Supported values are `"strain_limiting_baraff_witkin"` and `"neo_hookean"`. |
| `model.uipc.rod_model` | Curve | `"hookean_spring"` | Primary centerline constitution, imported from simulation-curve `uipc:constitution`. Positive Newton bend stiffness adds `KirchhoffRodBending` separately. |
| `model.uipc.deformable_model` | `uipc:deformable_body` | `"stable_neo_hookean"` | Per-authored-soft-body tetrahedral constitution. Supported values are `"stable_neo_hookean"`, `"arap"`, and `"empty"`. |

The same custom-frequency rows carry group labels, owning worlds, inclusive
`first`/`last` indices for the particle/topology domains, and authored density:

- `model.uipc.cloth_{label,world,particle_*,triangle_*,edge_*,spring_*,surface_density}`
- `model.uipc.deformable_body_{label,world,particle_*,tetrahedron_*,triangle_*,edge_*,density}`

The row counts are resolved from the final builder group registries, so
registration may occur after authoring but must occur before `finalize()`.
Without registration, `SolverUIPC` retains the legacy topology selectors and
does not preserve multiple authored groups that share one Newton world.

## Affine body visual bindings

`SolverUIPC.get_body_affine_transforms()` exports full `wp.mat44d` transforms
and a `wp.bool` validity mask in Newton body order. After initializing the
solver, pass these stable buffers to `newton.geometry.DeformableVisuals.update`
to preserve ABD stretch and shear in BODY-bound visual meshes:

```python
body_affine, body_affine_valid = solver.get_body_affine_transforms()
visuals = newton.geometry.DeformableVisuals(model)
viewer.set_deformable_visuals(visuals)  # After viewer.set_model(model).

# After each step, before viewer.log_state(state):
visuals.update(state, body_affine=body_affine, body_affine_valid=body_affine_valid)
viewer.log_state(state)
```

Entries with a false validity marker use `state.body_q`, including the FEM
rod segment bodies. World-fixed visual vertices stay fixed. The matrix maps
the same body-local bind coordinates used by BODY visual bindings to world
coordinates; its upper 3x3 retains stretch and shear. Normals are recomputed
from the deformed triangles. Keep the solver-owned arrays alive during
evaluation and graph replay, and order the visual update after the physics
step. This does not change the simulation state or regular shape rendering.

## Rods

Newton capsule-segment rods, including imported linear `BasisCurves`, use UIPC
`HookeanSpring` (UID 12) and, when bend stiffness is positive,
`KirchhoffRodBending` (UID 15). Their segments are excluded from ABD conversion.
The adapter welds ROD joint endpoints into a nonbranching centerline and
reconstructs Newton body poses, COM velocities, and joint coordinates after
each solve. Segment roll is parallel transported; there is no axial spin.
Newton capsule shapes retain their rest lengths while the FEM centerline can
stretch, so reconstructed capsules approximate the solved geometry.

For each ROD joint, let `L_dual` be the mean adjacent capsule rest length.
The adapter recovers `EA = stretch_stiffness * L_dual` and
`EI = bend_stiffness * L_dual`, then divides by `pi*r^2` and `pi*r^4/4`
to obtain the UIPC axial and bending moduli. An edge uses the mean recovered
`EA` of its incident joints. An isolated segment without a ROD joint uses
`EA = 1e5 * rest_length`. Half of each Newton body mass is assigned to each
endpoint; shared nodes sum those masses. External body forces and transverse
torques are distributed to the endpoints. Torques along the centerline have
no effect in this model.

Open and closed chains require one capsule per segment, uniform positive
radius, coincident initial joint endpoints, and isotropic bending. World
`FREE`, `FIXED`, and endpoint `BALL` roots are supported. Kinematic or
zero-mass segments fix both endpoints; kinematic targets follow the input
body state each step. `reset()` supports body position/velocity flags and
world masks. Branching, backtracking bending hinges, disabled ROD joints, attachments to other bodies,
per-body contact-element overrides, and coupling proxy rods raise errors.

This approximation omits shear, twist, material damping, and intrinsic rest
curvature and emits a warning at initialization. A curved rod will tend
toward a straight stress-free centerline. Native contact with other UIPC
geometry is available. `CONTACT_F` observables are not supported for scenes
containing these rods because a shared centerline node has no unique Newton
body owner.

The experimental `NewtonUIPCRodAPI` declares the USD selector:

```python
curve_prim.ApplyAPI("NewtonUIPCRodAPI")
curve_prim.GetAttribute("uipc:constitution").Set("hookean_spring")
```

The `BasisCurves` must already carry `PhysicsCurvesDeformableSimAPI`.
Register `SolverUIPC` attributes before `builder.add_usd(...)`; the selector
is stored in `model.uipc.rod_model` at the corresponding curve row. A raw
custom token also works. Connected source curves must agree on authored
curve-frequency custom attributes. The selector defaults to `hookean_spring`
when omitted, and bending remains controlled by Newton's bend stiffness.

## Cloth

Newton cloth is built from particles, triangle connectivity, triangle material
properties, and bending-edge properties. UIPC cloth objects use an open
triangle mesh plus a membrane constitution and `DiscreteShellBending`.

| Newton source | UIPC target | Notes |
| --- | --- | --- |
| `particle_q` | Cloth mesh vertices | World-space particle positions [m]. |
| `tri_indices` / `model.uipc.cloth_*` | Cloth mesh faces and grouping | Registered custom-frequency rows preserve each authored cloth; otherwise the legacy selector uses triangle connectivity not claimed by tetrahedra. |
| `particle_mass` / `tri_areas` | Membrane mass density | Estimated as total mass divided by total area and shell thickness, unless an authored cloth surface density is present. |
| `particle_radius` | Vertex shell thickness | Cloth collision thickness follows Newton particle radius. |
| `tri_materials[:, 0]` / `tri_materials[:, 1]` | Triangle `mu` / `lambda` attributes | Written directly to UIPC membrane attributes after applying the chosen membrane model. |
| `edge_bending_properties[:, 0]` | Edge `bending_stiffness` | Damping (`edge_kd`) is not forwarded. |
| `particle_mass <= 0.0` | UIPC `is_fixed` vertex marker | Marks kinematic cloth vertices. |

The default membrane model is `StrainLimitingBaraffWitkinShell`. Call
`SolverUIPC.register_custom_attributes(builder)` before `finalize()`, then set
`model.uipc.cloth_model[cloth_index]` before constructing `SolverUIPC` to choose
`NeoHookeanShell` for one authored `uipc:cloth` row. Closed /
watertight triangle meshes are rejected as cloth; use a deformable or rigid
representation for closed volumes.

USD cloth assets can author the membrane choice with the experimental
`NewtonUIPCClothAPI` on the simulation `Mesh`:

```python
cloth_prim.ApplyAPI("NewtonUIPCClothAPI")
cloth_prim.GetAttribute("uipc:constitution").Set("neo_hookean")
```

Use the updated `newton-usd-schemas` package and register its schemas before
creating or opening the USD stage. The supported tokens are
`"strain_limiting_baraff_witkin"` (default) and `"neo_hookean"`. Register UIPC
custom attributes before `builder.add_usd(...)`; the importer stores the token
in the corresponding `model.uipc.cloth_model` row. An explicitly authored
custom token named `uipc:constitution` also works without applying this API.
The mesh must already be marked as simulation cloth with
`PhysicsSurfaceDeformableSimAPI`. This choice selects only the membrane;
`DiscreteShellBending` remains a separate, automatically applied constitution.

When `enable_soft_position_constraint=True` (the default), the builder adds
dormant UIPC `SoftPositionConstraint` attributes. Use
`set_cloth_soft_position_constraints` to enable selected particle handles and
`clear_cloth_soft_position_constraints` to disable them.

## Deformable bodies

Newton deformables are built from particles and tetrahedra. UIPC deformables
use `tetmesh` geometry, surface labels for contact, and
`StableNeoHookean` elasticity by default. Call
`SolverUIPC.register_custom_attributes(builder)` before `finalize()`, then
set `model.uipc.deformable_model[soft_index]` before constructing `SolverUIPC`
to choose a different supported constitution for one authored
`uipc:deformable_body` row.

Set the value to `"empty"` to use UIPC `Empty` (UID 0). This retains the tet
geometry and mass while omitting elastic energy. Contact and soft-position
constraints remain available; `tet_materials` are ignored for this body.

USD assets can select the same constitution with the experimental
`NewtonUIPCDeformableAPI` from the vendored `newton-usd-schemas` package. Apply
it to the simulation `TetMesh`, then author its `uipc:constitution` token:

```python
import newton_usd_schemas  # Register schemas before creating a USD stage.

tet_prim.ApplyAPI("NewtonUIPCDeformableAPI")
tet_prim.GetAttribute("uipc:constitution").Set("empty")
```

Supported tokens are `"stable_neo_hookean"` (the schema default), `"arap"`,
and `"empty"`. Register UIPC custom attributes on the builder **before**
`builder.add_usd(...)`; the importer stores the authored token in the
corresponding `model.uipc.deformable_model` row. No separate
`model.uipc.constitution` attribute is created. An explicitly authored custom
token named `uipc:constitution` also works without applying the API schema.
This selection applies to tetrahedral soft bodies; it does not select cloth
or rigid-body constitutions.

| Newton source | UIPC target | Notes |
| --- | --- | --- |
| `particle_q` | Tet mesh vertices | World-space particle positions [m]. |
| `tet_indices` / `model.uipc.deformable_body_*` | Tet mesh topology and grouping | Registered custom-frequency rows keep multiple authored soft bodies separate. |
| `particle_mass` + tet volume | Mass density | Falls back to `default_mass_density` when the estimate is unavailable. |
| `tet_materials[:, 0]` / `tet_materials[:, 1]` | Deformable material attributes | Stable Neo-Hookean writes converted `mu` / `lambda`; ARAP writes `tet_materials[:, 0]` to UIPC `kappa`; Empty ignores these values. |
| `particle_mass <= 0.0` | UIPC `is_fixed` vertex marker | Marks kinematic deformable vertices. |

When soft-position constraints are enabled, use
`set_deformable_soft_position_constraints` and
`clear_deformable_soft_position_constraints` to drive selected deformable
particles.

## Actuators and controls

UIPC joint control is applied through UIPC `Animator` callbacks registered for
active revolute and prismatic joints. Before each `world.advance()`, Newton
copies the current `Control` buffers into CPU arrays consumed by those
callbacks.

| Newton target mode | UIPC behavior |
| --- | --- |
| `POSITION` | Enables UIPC driving and writes `aim_angle` / `aim_distance` from `Control.joint_target`. |
| `POSITION_VELOCITY` | Same as `POSITION`; the position target is forwarded. |
| `EFFORT` | Reads `Control.joint_f`; revolute joints with positive armature apply torque in the joint-space inertia potential, while other joints use native external torque / force. |
| `VELOCITY` | Passive; no UIPC velocity-only drive is written. |
| `NONE` | Passive. |

The aim-drive strength is a pure solver constraint-stiffness knob, deliberately
decoupled from `joint_target_ke` / `joint_target_kd` (which UIPC does not
consume — they remain portable metadata for other backends). Position-driven
joints get the solver-level `drive_strength_ratio` parameter: a global float
(default `100.0`, near-rigid tracking) or a per-joint mapping keyed by Newton
joint index, e.g. `SolverUIPC(model, drive_strength_ratio={3: 10.0})`; joints
missing from the mapping fall back to `100.0`. Non-position target modes get
no drive. The joint anchoring constraints themselves use the independent
solver-level `joint_strength_ratio` parameter (default `100.0`).

Alternatively, `SolverUIPC(implicit_pd=True)` opts position-driven joints into
implicit PD with physical gain semantics: `joint_target_ke` /
`joint_target_kd` [N·m/rad, N·m·s/rad] replace `drive_strength_ratio`. The PD
is expressed inside the incremental potential — the stiffness spring acts on
the new-state position error and the damping spring on the new-state velocity
error (an aim toward `q_prev + dt * dq_ref`, merged with the position spring
into the single drive channel) — so it is co-solved with contact,
unconditionally stable, and equivalent to `SolverKamino`'s implicit joint PD:
steady-state sag under load is `tau / ke`, and `kd` damps transients.
`POSITION_VELOCITY` mode feeds `joint_target_qd` as the damping reference;
plain `POSITION` damps toward rest; `VELOCITY` mode becomes a velocity servo
(the damping spring alone tracks `joint_target_qd`, `joint_target_ke` is
ignored). Gains are read at initialization; to change them at runtime, write
`joint_target_ke` / `joint_target_kd` and call
`notify_model_changed(ModelFlags.JOINT_DOF_PROPERTIES)` — the drive
strengths are re-derived and take effect on the next step (libuipc re-reads
them every frame). The same call re-applies `joint_armature` to the live
reflected-inertia constraints (for joints that carried armature at build
time); friction and joint-limit changes remain baked.

`ke`/`kd` sag under gravity or other sustained loads because implicit PD has
no gravity-compensation channel of its own. A joint is either position-driven
or force-controlled, never both, so compensate in the position domain rather
than adding a coexisting feedforward torque: the implicit-PD steady state
satisfies `ke * (q_ref - q) = tau_g` (sag `tau_g / ke`), so pre-adding that
offset to `control.joint_target_q` lands the joint on the true target.
Compute the bias with `newton.eval_inverse_dynamics_passive(model, state,
gravity_force=..., coriolis_force=...)` and add `(gravity_force +
coriolis_force) / joint_target_ke` to the target each step; see
`example_uipc_brick_stacking.py --implicit-pd`.
`joint_limit_lower` / `joint_limit_upper` create UIPC joint-limit
constitutions whose strength comes from the solver-level
`limit_strength_ratio` parameter (default `10.0`, same global-or-per-joint
form as `drive_strength_ratio`), decoupled from `joint_limit_ke`.

## Contact pipeline

`SolverUIPC` computes contact inside UIPC. When `CONTACT_F` is requested,
`step` writes UIPC contact records into the supplied `Contacts` and forces into
`observables.contact_f`. Allocate reporting storage directly with
`newton.Contacts` or with Newton's collision pipeline; do not call `collide()`
on it before reading the results.
Without an observable request, `step` skips contact readback.

Global contact is disabled in the default Newton UIPC scene config:

```python
solver = SolverUIPC(model)
solver.set_contact(True, d_hat=0.001)
```

`d_hat` is UIPC's IPC barrier distance [m]. Smaller values allow tighter
contacts but may require smaller time steps or more Newton iterations for
stability.

### Contact elements and default pairs

During initialization, Newton creates one shared ground contact element and
three contact elements per Newton world:

- `ground_elem` — ground planes.
- `env_elem` — environment / non-articulated rigid bodies and kinematic bodies.
- `robo_elem` — articulated robot links.
- `actor_elem` — free-joint actors, cloth, and deformables.

Default contact pairs use friction `0.5` and `kappa=-1.0`, which opts into
UIPC's scene-adaptive contact stiffness. UIPC chooses the effective kappa from
the scene mass, scale, `d_hat`, and time step:

| Pair | Enabled by default |
| --- | --- |
| `env` ↔ `env` | No |
| `env` ↔ `robot` | Yes |
| `env` ↔ `actor` | Yes |
| `ground` ↔ `env` | No |
| `ground` ↔ `robot` | Yes |
| `ground` ↔ `actor` | Yes |
| `robot` ↔ `robot` | No |
| `robot` ↔ `actor` | Yes |
| `actor` ↔ `actor` | Yes |

`SolverUIPC.ADAPTIVE_KAPPA_MIN` and `SolverUIPC.ADAPTIVE_KAPPA_MAX` expose the
measured UIPC 0.9.0 corridor for the brick-stacking scene
(`[9.538500988573645e9, 9.538500988573645e11]` Pa). The corridor is derived
from scene mass, scale, `d_hat`, and time step, so changing those inputs or the
UIPC build can change it. A callback can pass the lower value for a fixed
contact workload that should use the least stiff accepted resistance; this
also avoids registering the adaptive reporter for that scene. The
`uipc_brick_stacking` example uses this fixed-minimum policy because its
stud/tube interlock is a persistent, friction-dominated contact.

Use `configure_contact_tabular` before initialization to insert additional
elements, override friction / stiffness, or change which pairs are enabled. The
callback may also return a `{body_index: contact_element}` mapping to override
individual body assignment.

### Contact readback

`SolverUIPC` supports the experimental `ObservableKind.CONTACT_F` output.
Allocate it once, then pass it to each `step` that needs contact diagnostics:

```python
sensor = newton.sensors.SensorContact(model, sensing_shapes="box", request_contact_attributes=False)
contacts = newton.Contacts(solver.get_max_contact_count(), 0, device=model.device)
observables = solver.observables(sensor.solver_observable_kinds, contacts=contacts)

solver.step(state_in, state_out, control, contacts, dt, observables=observables)
sensor.update(state_out, contacts, observables=observables)
viewer.log_contacts(contacts, state_out, observables=observables)
```

Allocation uses the supplied `Contacts` capacities and does not freeze the
model's contact capacities. Allocate for the largest reporting buffer needed:
each step may use a different `Contacts` on the solver device if its combined
rigid and soft capacity fits `observables.contact_f`. Sensors and the viewer
must receive the buffer used by the step that produced the forces, available
as `observables.contacts` (or
`observables.contacts_for_kind(solver.ObservableKind.CONTACT_F)`).

The reporting capacity can be smaller than UIPC's native contact count;
excess entries are dropped. For multiple substeps, pass `observables` only
on the final substep when only frame-end
readings are needed. Omitted requests leave the previous report unchanged;
a requested readback with no contacts clears the report.

Each UIPC contact stencil (PT/EE/PE/PP/PH) anchored on a rigid body becomes
one rigid-contact entry. `observables.contact_f` contains the world-space
force [N] on the first shape, with zero torque components [N·m]. Pair arrays
contain the touching shapes and representative points
(`rigid_contact_point0/1`), so `SensorContact` reports per-counterpart forces
and `position_matrix` contact positions. FEM counterparts and the ground
are attributed to the ground-plane shape. Contact positions are vertex-level
representatives, so a face-face interface localizes to within the contact
face rather than its exact centroid. The soft-contact segment is unused and
zeroed; FEM-only contact stencils are not exported by `CONTACT_F`.

Contact diagnostics are available only through observables. Migrate
`update_contacts(contacts, state)` calls to the `step(..., observables=...)`
workflow above. UIPC no longer writes contact forces into `contacts.force`,
`state.body_f`, or `state.particle_f`. `BODY_QDD` and `BODY_PARENT_F` are not
supported by UIPC.

For CPU diagnostics, `_get_contact_forces` returns the lower-level per-body,
per-primitive UIPC contact-force buckets. It is intentionally an internal
debugging entry point rather than the normal public readback path.

## Multi-world support

For models assembled with `ModelBuilder.add_world` or `ModelBuilder.replicate`,
`SolverUIPC` builds all Newton worlds into one UIPC scene. Worlds may contain
different articulations and different numbers of bodies and joints. The solver
creates one UIPC subscene per Newton world and places each world's bodies,
cloth, and deformables into that subscene.

By default, cross-subscene contact is disabled, while each world can still
contact the default subscene used by ground planes and global objects. Use
`configure_subscene_tabular` before initialization to opt into cross-world
contact or other custom subscene rules.

### Sparse articulation views

Use :class:`~newton.selection.ArticulationView` with `homogeneous=False` to
control and read matching robots when some worlds contain other assets or
the robots occupy different offsets in each world's arrays. UIPC uses
absolute Newton joint, coordinate, DOF, and body indices for control and state
readback; no solver option is required.

For example, if `robot` occurs in model worlds 0, 2, and 4, the view has three
world rows and `world_ids` is `[0, 2, 4]`. Position targets use coordinate
indices, while velocities and efforts use DOF indices, including when a
floating root makes those indices differ:

```python
from newton.selection import ArticulationView

robots = ArticulationView(model, "robot", homogeneous=False)
drives = ArticulationView(
    model,
    "robot",
    homogeneous=False,
    include_joint_types=[newton.JointType.REVOLUTE, newton.JointType.PRISMATIC],
)
targets = drives.get_attribute("joint_target_q", control).numpy()
targets[..., 0] = 0.2
drives.set_attribute("joint_target_q", control, targets, mask=[True, False, True])

solver.step(state_in, state_out, control, dt=dt)
positions = drives.get_dof_positions(state_out)
velocities = drives.get_dof_velocities(state_out)
```

Getters can return gathered copies. Use the matching setter to publish edits.
View setters interpret masks on the view's compact world axis, so the mask
above selects model worlds 0 and 4. `SolverUIPC.reset` instead takes a mask on
the full model world axis. Map it through `world_ids` after editing root or
joint state through the view:

```python
import numpy as np
import warp as wp

view_mask = np.array([True, False, True])
model_mask = np.zeros(model.world_count, dtype=bool)
model_mask[robots.world_ids.numpy()[view_mask]] = True
solver.reset(
    state_in,
    world_mask=wp.array(model_mask, dtype=bool, device=model.device),
    flags=newton.StateFlags.JOINT_Q | newton.StateFlags.JOINT_QD,
)
```

Only frequencies with a common layout among the selected articulations are
available through a view. `homogeneous=False` does not make differing joint
counts into a rectangular tensor or extend UIPC's supported joint types.

## Runtime state synchronization

Each call to `SolverUIPC.step` uses the same three-phase cycle:

1. **Push control / snapshot joints.** Active joint controls are cached for
   UIPC animator callbacks. Current revolute `angle` / prismatic `distance`
   attributes are snapshotted so post-step joint velocities can be finite
   differenced.
2. **Advance UIPC.** `world.advance()` runs UIPC's nonlinear solve, contact
   handling, and animator callbacks. `world.retrieve()` makes the integrated
   state available for readback.
3. **Pull UIPC → Newton.** AffineBody transforms / velocities and FEM particle
   positions / velocities are copied into `state_out`; active joint positions
   and velocities are written into `state_out.joint_q` / `joint_qd`; force
   buffers are cleared. Requested contact observables are exported separately.

`dt` is fixed by the constructor. If the first `step` receives a different
`dt`, Newton warns; UIPC still uses the constructor-configured time step.

Per-world reset is supported at runtime via
:meth:`~newton.solvers.SolverUIPC.reset`, which re-pushes the selected worlds'
rigid-body (and FEM particle) state into the live IPC scene without rebuilding
the solver. After a body-state push, UIPC's internal revolute/prismatic
joint-angle tracker lags by one step; read ``Model.joint_q`` rather than
``State.joint_q`` for articulated bodies until the next ``step``.

## Runtime model updates

Most model properties are baked into UIPC geometry during initialization. After
the solver is initialized, `notify_model_changed` supports only:

| Flag | Runtime effect |
| --- | --- |
| `BODY_PROPERTIES` | Pushes updated `body_q` / `body_qd` and FEM particle state into UIPC. |
| `JOINT_PROPERTIES` | Runs Newton FK from `joint_q` / `joint_qd` / joint frames, then pushes the resulting body state into UIPC. |
| `MODEL_PROPERTIES` | Propagates `model.gravity` into the live UIPC `scene.config()`. |
| `JOINT_DOF_PROPERTIES` | Re-applies `joint_armature` to the live reflected-inertia constraints (joints that carried armature at build time only), and under `implicit_pd` re-derives the drive strengths from `joint_target_ke` / `joint_target_kd`. Friction and limit edits remain baked, as do gain edits without `implicit_pd`. |

Unsupported flags (`BODY_INERTIAL_PROPERTIES`, `SHAPE_PROPERTIES`,
`CONSTRAINT_PROPERTIES`, `TENDON_PROPERTIES`, and `ACTUATOR_PROPERTIES`)
produce one aggregated warning. Recreate the solver when those properties
change.

`set_contact` is safe before or after initialization. `set_animator_substep`
can be called after initialization to change how many times UIPC animator
callbacks fire within one `world.advance()`.

## Solver options

Important constructor arguments:

| Argument | Default | Notes |
| --- | --- | --- |
| `backend` | `"cuda"` | Passed to `uipc.Engine(backend_name=...)`. |
| `workspace` | `"/tmp/newton_uipc"` | UIPC engine workspace, surface-dump directory, and default performance-report root. |
| `dt` | `1.0 / 60.0` | Fixed UIPC scene time step [s]. |
| `scene_config` | `None` | Defaults to FusedPCG, full-GPU CUDA Graph mode 2, the MAS FEM preconditioner, and IPC contact, then applies Newton overrides for `dt`, gravity, contact, and Newton tolerances. Explicit scene configs retain their solver choices; an omitted FEM preconditioner defaults to MAS. |
| `kappa` | `1 GPa` | Solver-level AffineBody stiffness [Pa]. |
| `default_mass_density` | `1000.0` | Fallback density [kg/m³] for rigid/deformable construction. |
| `dump_enable` | `False` | Dumps UIPC surface OBJ snapshots before each physics advance. |
| `require_profile` | `False` | Collects UIPC timer data; use `save_performance_report`. |
| `auto_sync_inertia` | `True` | Sync final UIPC AffineBody inertia back into Newton after initialization. |
| `enable_soft_position_constraint` | `True` | Adds dormant cloth/deformable soft-position attributes. |

Use `configure_scene` for UIPC scene configuration not exposed as constructor
arguments. It deep-merges nested dictionaries before initialization.

FEM preconditioning defaults to `"mas"` when an explicit config omits
`linear_system` or `linear_system.fem_preconditioner`. A config created with
`uipc.Scene.default_config()` includes UIPC's native `"diag"` selection, which
Newton preserves. To enable MAS in that config, set it before constructing the
solver:

```python
import uipc
from newton.solvers import SolverUIPC

scene_config = uipc.Scene.default_config()
scene_config["linear_system"]["fem_preconditioner"] = "mas"
solver = SolverUIPC(model, scene_config=scene_config)
```

### FusedPCG CUDA Graph scope

The default scene config uses `linear_system.solver="fused_pcg"`,
`linear_system.use_cuda_graph=2`, `linear_system.fem_preconditioner="mas"`,
and `contact.constitution="ipc"`. UIPC 0.0.27 auto-partitions non-Empty FEM
geometries for MAS; this replaces its removed Python `mesh_partition()` API.
Mode 2 captures one FusedPCG solve, including initialization,
preconditioning, iterations, and device-side convergence checks, in one CUDA
Graph launch. It does not capture the whole frame: Newton iteration
orchestration, collision detection and dynamic contact generation,
gradient/Hessian assembly, CCD, CFL, line search, Newton convergence, and
scene retrieval remain outside the graph.

Mode 2 requires CUDA conditional graph-node support. UIPC falls back to mode 1
when that support is unavailable, and falls back to block replay or ordinary
kernel launches if capture fails. UIPC also disables graph replay for AL-IPC.
Mode 1 is generally faster; mode 2 is intended for workloads that benefit from
keeping the CPU out of the PCG loop. Call `configure_scene` before
`initialize()` to select a different mode.

## Unsupported Newton features and caveats

Smaller limitations are documented inline above. The most common ones are:

- **Global contact is off by default.** Call `set_contact(True, d_hat=...)` for
  contact-driven examples.
- **Rigid AffineBody meshes must be closed.** Non-watertight rigid meshes raise
  during initialization. Use `ModelBuilder.approximate_meshes` to generate a
  closed convex hull, COACD approximation, or bounding box before constructing
  the solver.
- **Only revolute and prismatic joints are active driven/read-back joints.**
  Ball joints constrain anchors but do not participate in the active joint
  control/readback arrays. Distance, D6, and cable joints are unsupported.
- **Mimic constraints are enforced softly via position driving.** The follower
  tracks `coef0 + coef1 * q_leader` through its driving-joint stiffness and may
  lag under load; this is not a hard equality constraint. Follower and leader
  must both be active revolute/prismatic joints, and coefficients are baked at
  initialization.
- **Velocity-only joint drives are not forwarded.** `VELOCITY` target mode is
  passive in UIPC; `POSITION_VELOCITY` forwards only the position target.
- **Most edits require solver reconstruction.** Shape changes, actuator changes,
  inertial changes, constraints, and tendons are not pushed into live UIPC
  objects. Joint-DOF properties are the exception:
  `notify_model_changed(JOINT_DOF_PROPERTIES)` re-applies `joint_armature`
  (for joints that carried armature at build time) and, under `implicit_pd`,
  `joint_target_ke` / `joint_target_kd` edits; friction and limit edits remain
  baked.
- **Newton-native contact features are separate.** SDF and hydroelastic contact
  pipelines are not fed into UIPC.

(uipc-code-pointers)=
## Code pointers

For readers navigating the source, the following symbols are the most useful
entry points. Symbols with a leading underscore are **internal entry points** —
stable enough to navigate to, but not part of the public API and subject to
change.

- `SolverUIPC.register_custom_attributes` — UIPC-specific custom attribute
  registration (`uipc:abd_kappa`).
- `SolverUIPC.__init__` — scene-config defaults and constructor options.
- `SolverUIPC.configure_scene`, `set_contact`, `configure_contact_tabular`, and
  `configure_subscene_tabular` — pre-initialization scene/contact setup.
- `SolverUIPC.initialize` — Newton `Model` → UIPC `Engine` / `World` / `Scene`
  construction and multi-world setup.
- `SolverUIPC.step` — per-step UIPC integration entry point.
- `SolverUIPC.observables` — allocate contact-force outputs for
  `step(..., observables=...)`, contact sensors, and the viewer.
- `SolverUIPC.sync_model_inertia_from_uipc` /
  `sync_uipc_inertia_with_model` — bidirectional AffineBody mass/inertia bridge.
- `SolverUIPC.notify_model_changed` — runtime updates supported by the UIPC
  backend.
- `SolverUIPC.set_cloth_soft_position_constraints` /
  `set_deformable_soft_position_constraints` — soft-position handles for FEM
  particles.
- `newton/_src/solvers/uipc/rigid_body.py` — AffineBody, ground-plane, and
  static-collider construction.
- `newton/_src/solvers/uipc/articulation_builder.py` and
  `articulation.py` — joint conversion, UIPC animator callbacks, and joint
  state readback.
- `newton/_src/solvers/uipc/cloth.py` — cloth shell and bending conversion.
- `newton/_src/solvers/uipc/deformable_body.py` — Stable Neo-Hookean tet
  conversion.
- `newton/_src/solvers/uipc/contact_forces.py` — UIPC contact-gradient readback
  and Warp scatter kernels.
- `newton/_src/solvers/uipc/converter.py` — shape-to-mesh conversion, state
  mapping, and UIPC backend-offset tables.
