{ pkgs ? import <nixpkgs> { } }:
with pkgs;
mkShell {
  buildInputs = [
    julia
    jupyter-all
    python313Packages.ipython
    python313Packages.ipywidgets
    python313Packages.matplotlib
    python313Packages.numpy
  ];

  shellHook = ''
  '';
}
