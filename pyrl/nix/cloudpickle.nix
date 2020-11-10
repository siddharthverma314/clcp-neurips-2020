{ stdenv, buildPythonPackage, fetchPypi, isPy27, pytest, mock }:

buildPythonPackage rec {
  pname = "cloudpickle";
  version = "1.3.0";
  disabled = isPy27; # abandoned upstream

  src = fetchPypi {
    inherit pname version;
    sha256 = "0lx7gy9clp427qwcm7b23zdsldpr03gy3vxxhyi8fpbhwz859brq";
  };

  buildInputs = [ pytest mock ];

  # See README for tests invocation
  checkPhase = ''
    PYTHONPATH=$PYTHONPATH:'.:tests' py.test
  '';

  # TypeError: cannot serialize '_io.FileIO' object
  doCheck = false;
}
