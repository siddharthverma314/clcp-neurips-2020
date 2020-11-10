{ buildPythonPackage
, pytorch
, gym
, cpprb
, flatten-dict
, termcolor
, pygments
, tabulate
, gitignore
}:
buildPythonPackage rec {
  pname = "pyrl";
  version = "0.1.0";

  src = gitignore ./.;

  propagatedBuildInputs = [
    pytorch
    gym
    cpprb
    flatten-dict
    termcolor
    pygments
    tabulate
  ];

  doCheck = false;
}
