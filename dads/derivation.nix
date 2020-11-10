{ buildPythonPackage
#, tf-agents
, gym
, matplotlib
, mujoco-py
, click
, transforms3d
, callPackage
}:
let
  tensorflow = callPackage ./nix/tensorflow.nix {};
  tensorflow-probability = callPackage ./nix/tensorflow-probability.nix {};
  tf-agents = callPackage ./nix/tf-agents.nix {};
  robel = callPackage ./nix/robel.nix {};
in
buildPythonPackage {
  pname = "dads";
  version = "0.1.1";
  src = ./.;

  propagatedBuildInputs = [
    tensorflow
    tensorflow-probability
    tf-agents
    gym
    matplotlib
    mujoco-py
    click
    robel
    transforms3d
  ];
}
