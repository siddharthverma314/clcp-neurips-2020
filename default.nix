{ pkgs ? import ./pyrl/nix/nixpkgs.nix {} }:
let
  # nixpkgs
  pkgs_cpu = import ./pyrl/nix/nixpkgs.nix { cudaSupport = false; python = "python37"; };
  pkgs_gpu = import ./pyrl/nix/nixpkgs.nix { cudaSupport = true; python = "python37"; };

  # derivation
  pkg_cpu = pkgs_cpu.python3Packages.callPackage ./derivation.nix {};
  pkg_gpu = pkgs_gpu.python3Packages.callPackage ./derivation.nix {};

  # docker 
  docker = (pkgs: pkg: pkgs.dockerTools.buildImage {
    name = "siddharthverma/adversarial";
    tag = "latest";
    contents = [
      pkgs.coreutils
      pkgs.bashInteractive
      (pkgs.python3.withPackages (_: [ pkg ]))
    ];
    diskSize = 1024 * 32;
    #runAsRoot = '' 
    #  #!${pkgs.runtimeShell}
    #  mkdir -p /tmp
    #'';
    config = {
      Env = ["NVIDIA_DRIVER_CAPABILITIES=compute,utility" "NVIDIA_VISIBLE_DEVICES=all"];
    };
  });
in
{
  inherit pkg_cpu pkg_gpu;
  docker_cpu = docker pkgs_cpu pkg_cpu;
  #docker_gpu = docker pkgs_gpu pkg_gpu;
  pyrl_dev = (import ./pyrl {pkgs = pkgs_gpu;}).dev;
}
