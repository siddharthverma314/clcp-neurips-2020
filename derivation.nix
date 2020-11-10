{ buildPythonPackage
, mujoco-py
, gitignore
, scikit-video
, matplotlib
, callPackage
}:
let
  pyrl = callPackage ./pyrl/derivation.nix {};
  ant-hrl-maze = callPackage ./ant-hrl-maze/derivation.nix {};
  dads = callPackage ./dads/derivation.nix {};
in
buildPythonPackage {
  pname = "adversarial";
  version = "0.1.0";

  src = gitignore ./.;

  propagatedBuildInputs = [
    pyrl
    dads
    mujoco-py
    scikit-video
    ant-hrl-maze
    matplotlib
  ];
}
