{ buildPythonPackage
, nix-gitignore
, gym
, numpy
, mujoco-py
}:
buildPythonPackage {
  pname = "ant-hrl-maze";
  version = "0.1.0";

  src = nix-gitignore.gitignoreSource [] ./.;

  propagatedBuildInputs = [
    mujoco-py
    numpy
    gym
  ];

  doCheck = false;
}
