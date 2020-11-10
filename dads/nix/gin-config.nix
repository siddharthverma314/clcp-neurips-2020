{ lib
, buildPythonPackage
, fetchPypi
, six
, enum34
}:

buildPythonPackage rec {
  pname = "gin-config";
  version = "0.1.3";

  src = fetchPypi {
    inherit pname version;
    sha256 = "18lbgqzc01i2ckk8vfpbqkvjzc3v4b0b7klgh17s9w8p57g3z107";
  };

  propagatedBuildInputs = [ six enum34 ];

  # PyPI archive does not ship with tests
  doCheck= false;
}
