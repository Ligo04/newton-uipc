Add optional affine body transforms and validity masks to
`DeformableVisuals.update()` so BODY-bound visual meshes can preserve UIPC ABD
stretch and shear while falling back to rigid poses for other bodies.
