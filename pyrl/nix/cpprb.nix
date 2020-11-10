{ callPackage
, buildPythonPackage
, fetchPypi
, numpy
, cython
, gym
, sources
}:
buildPythonPackage rec {
  pname = "cpprb";
  version = "9.2.0";
  src = sources.cpprb;
  propagatedBuildInputs = [ numpy cython gym ];
}
