{ stdenv, fetchPypi, buildPythonPackage
, numpy
, absl-py
, mock
}:

buildPythonPackage rec {
  pname = "tensorflow-estimator";
  version = "2.2.0";
  format = "wheel";

  src = fetchPypi {
    pname = "tensorflow_estimator";
    inherit version format;
    sha256 = "1hkx4k6927xn4qpwiba6wa56n0qqm7s23bymm377j9bz2bfsr7fh";
  };

  propagatedBuildInputs = [ mock numpy absl-py ];
}
