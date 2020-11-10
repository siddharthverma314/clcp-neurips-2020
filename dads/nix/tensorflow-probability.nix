{ buildPythonPackage
, fetchPypi
, six
, numpy
, decorator
, gast
, hypothesis
, pytest
, scipy
, matplotlib
, mock
, cloudpickle
, fetchurl
, callPackage
}:

let
  pname = "tensorflow-probability";
  version = "0.10.0";
  dm-tree = callPackage ./dm-tree.nix {};
in
buildPythonPackage {
  inherit version pname;
  format = "wheel";

  src = fetchurl {
    url = "https://files.pythonhosted.org/packages/ec/61/800c19c1d586b1e06dc9d645f87c158aadf74233d9d03005d2fbef8a2c04/tensorflow_probability-0.10.1-py2.py3-none-any.whl";
    sha256 = "1l0lbwrpqpqnf8rik3canyk3q3fg3llr88xbfwyn44mfw39k1ris";
  };

  propagatedBuildInputs = [
    (callPackage ./tensorflow.nix {})
    six
    numpy
    decorator
    gast
    dm-tree
    cloudpickle
  ];

  # Listed here:
  # https://github.com/tensorflow/probability/blob/f01d27a6f256430f03b14beb14d37def726cb257/testing/run_tests.sh#L58
  checkInputs = [
    hypothesis
    pytest
    scipy
    matplotlib
    mock
  ];

  # actual checks currently fail because for some reason
  # tf.enable_eager_execution is called too late. Probably because upstream
  # intents these tests to be run by bazel, not plain pytest.
  # checkPhase = ''
  #   # tests need to import from other test files
  #   export PYTHONPATH="$PWD/tensorflow-probability:$PYTHONPATH"
  #   py.test
  # '';

  # sanity check
  checkPhase = ''
    python -c 'import tensorflow_probability'
  '';
}
