Add a centerline FEM approximation of capsule-segment ROD chains in `SolverUIPC`,
using `HookeanSpring` and optional `KirchhoffRodBending`, with mass and material
mapping, fixed endpoints, body-force transfer, state readback, and masked resets.
Add `NewtonUIPCRodAPI` to select `hookean_spring` through USD `uipc:constitution`.
The approximation explicitly omits shear, twist, material damping, and intrinsic
rest curvature; contact wrench observables and attachments to other bodies are
not supported.
