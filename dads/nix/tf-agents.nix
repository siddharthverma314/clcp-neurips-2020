{ buildPythonPackage
, fetchPypi
, absl-py
, cloudpickle
, numpy
, six
, protobuf
, wrapt
, pillow
, callPackage
}:

buildPythonPackage rec {
  pname = "tf_agents";
  version = "0.4.0";
  format = "wheel";

  src = fetchPypi {
    inherit pname version;
    python = "py3";
    format = "wheel";
    sha256 = "093irl49xglx9kz6ka5h1kx0fzdry56avczql79fnfa34s3wqn85";
  };

  propagatedBuildInputs = [
    absl-py
    cloudpickle
    (callPackage ./gin-config.nix {})
    (callPackage ./tensorflow-probability.nix {})
    numpy
    six
    protobuf
    wrapt
    pillow
  ];

  # TypeError: cannot serialize '_io.FileIO' object
  doCheck = false;
}
