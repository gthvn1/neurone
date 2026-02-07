{ pkgs ? import <nixpkgs> { } }:
with pkgs;
mkShell {
  buildInputs = [
    julia
  ];

  shellHook = ''
    # ...
  '';
}
