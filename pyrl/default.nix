{ pkgs ? import ./nix/nixpkgs.nix {} }:
let
  niv = (pkgs.callPackage pkgs.sources.niv {}).niv;
in
{
  pkg = pkgs.python3Packages.callPackage ./derivation.nix {};
  dev = pkgs.buildEnv {
    name = "pyrl-dev";
    paths = [
      niv
      pkgs.python-language-server
      (pkgs.python3.withPackages (ps: with ps; [
        pyls-black
        ipdb
        rope
        pyflakes
        pytest
        tensorflow-tensorboard_2
      ]))
    ];
  };
}
