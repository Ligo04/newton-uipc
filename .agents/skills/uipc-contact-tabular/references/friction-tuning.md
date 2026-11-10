# Friction smoothing and grasp slip

## Configuration and default ownership

`contact.eps_velocity` is the friction smoothing transition velocity in m/s.
Compare contact-relative tangential speeds with this threshold when diagnosing
slow sliding under sustained load. Pair-specific `mu` scales friction strength,
while `eps_velocity` controls the low-speed transition for all contact pairs.

Newton's `SolverUIPC` defaults to `0.001 m/s`, or `1 mm/s`, when `scene_config`
is omitted or its contact dictionary omits `eps_velocity`. Explicit values in
`scene_config` and pre-initialization `configure_scene()` overrides take
precedence. Check the current constructor when working in another revision.

Configure the absolute threshold before `solver.initialize()`.

```python
solver.configure_scene(
    {
        "contact": {
            "eps_velocity": 1.0e-3,
            "eps_velocity_relative": 0.0,
        }
    }
)
```

`eps_velocity_relative=0` selects the absolute threshold. A positive relative
value replaces it using the rest-scene diagonal. Inspect both fields in the
effective scene configuration.

The native `uipc.Scene.default_config()` default belongs to the loaded wheel.
The recorded `0.9.0+cu134.xsquare.g6e3386d1` wheel returns `0.01 m/s`; Newton
applies its own default when constructing `SolverUIPC`. A full native default
dictionary passed as `scene_config` supplies its `eps_velocity` explicitly.
Set the field in that dictionary or use `configure_scene()` to select another
value. Identify the requested default owner when changing framework defaults.

## Evaluate a tuning change

- Record the loaded wheel, source snapshot, `eps_velocity`, and
  `eps_velocity_relative`. Resolve the active contact elements and pair `mu`.
- Run a matched baseline and trial with the same initial geometry, mass,
  materials, motion schedule, time step, and substeps. Change the requested
  parameter and verify the actual configuration difference.
- Measure object motion relative to the fingers. A mass-weighted center of
  mass relative to the finger midpoint is a reproducible trajectory metric.
  It includes deformation and rotation. Contact-point tangential motion
  provides more direct evidence of surface slip.
- Report normal and tangential contact forces when assessing the `mu * N`
  friction limit. State the number of runs and the evaluated grasp criteria.

## Recorded Lulu comparison

On 2026-11-06, the `uipc_softbody_franka_lulu` trial used the xsquare wheel
above, `semi=0`, gripper `mu=0.9`, `use_cuda_graph=2`, MAS, 60 Hz, and one
substep. Material settings were `E=30 kPa`, `nu=0.3`, and density
`225 kg/m^3`. Source and wheel hashes, initial geometry, and mass matched
between configurations. The scene configurations differed only in
`contact.eps_velocity`, with `eps_velocity_relative=0` in both.

| Relative COM rearward displacement | 0.01 m/s | 0.001 m/s |
| --- | --- | --- |
| Closing, 4.5 to 7 s | 2.341 mm | 0.526 mm |
| Lifting and holding, 7 to 11 s | 10.265 mm | 0.215 mm |

Each configuration ran once for 1,000 frames and passed the example's grasp
checks. These measurements describe this scene and include deformation and
rotation. Evaluate other scenes with their own contact speeds and trajectories.

The [host-local comparison data](/home/ligo/.codex/visualizations/2026/11/06/01a1aefa-31a1-7323-9583-6b2b552986ae/franka-lulu-eps-velocity-001/comparison.json)
records the inputs and measurements. Configuration regression coverage is in
`newton/tests/test_uipc_scene_config.py`.
