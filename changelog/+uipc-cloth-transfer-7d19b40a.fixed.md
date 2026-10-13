Fix UIPC cloth mass conversion to use full shell thickness and area-weighted radii.
Convert Newton triangle Lamé inputs to Baraff-Witkin effective shear and stretch
coefficients, while retaining Lamé coefficients for Neo-Hookean shells. Existing
Baraff-Witkin scenes tuned against direct coefficient copies may need retuning;
triangle and edge damping remain unsupported by the UIPC cloth adapter.
