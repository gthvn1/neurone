{ pkgs ? import <nixpkgs> { } }:
with pkgs;
mkShell {
  buildInputs = [
    julia
    python313Packages.jupyterlab
    python313Packages.ipython
    python313Packages.ipympl
    python313Packages.ipywidgets
    python313Packages.matplotlib
    python313Packages.numpy
  ];

  shellHook = ''
  '';
}
