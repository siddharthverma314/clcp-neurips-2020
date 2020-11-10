let
  pkgs = import <nixpkgs> {};
  pythonreqs = pkgs.runCommand "requirements" {
    buildInputs = [
      pkgs.yq
    ];
  } ''
    mkdir -p $out
    yq -r '.dependencies[-1].pip | join("\n")' ${./env.yml} > $out/requirements.txt
  '';
in
pkgs.mkShell {
  buildInputs = [
    pkgs.python36Full
  ];
  shellHook = ''
    python -m venv .venv
    source .venv/bin/activate
    export C_INCLUDE_PATH=/home/vsiddharth/.mujoco/mujoco200/include:/home/vsiddharth/research/dads/.venv/lib/python3.6/site-packages/numpy/core/include
    export LD_LIBRARY_PATH=/home/vsiddharth/.mujoco/mujoco200/bin:/usr/lib/nvidia-384
    pip install -r ${pythonreqs}/requirements.txt
  '';
}
