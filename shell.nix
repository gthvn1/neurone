{ pkgs ? import <nixpkgs> { } }:
with pkgs;
mkShell {
  buildInputs = [
    julia
    jupyter
    python313Packages.ipython
    python313Packages.matplotlib
    python313Packages.numpy
  ];

  shellHook = ''
  '';
}
