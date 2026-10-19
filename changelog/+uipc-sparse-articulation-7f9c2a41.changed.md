Document and validate `SolverUIPC` control, state readback, and masked reset with
`ArticulationView(homogeneous=False)` across sparse and uneven worlds. Translate
view masks through `world_ids` before passing them to `SolverUIPC.reset`, whose
mask uses the full model world axis.
