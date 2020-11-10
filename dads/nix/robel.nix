{ fetchPypi
, buildPythonPackage
, mujoco-py
, transforms3d
, gym
, numpy
, absl-py
}:

let
  pname = "robel";
  version = "0.1.2";
in
buildPythonPackage {
  inherit pname version;
  src = fetchPypi {
    inherit pname version;
    sha256 = "0pigphy5f4z47bhll6k285ws3q2k5ycsadwxds9sblp9xw7hcd7q";
  };

  preConfigure = ''
    cat << EOF > ./requirements.txt
      gym
      mujoco-py
      numpy
      transforms3d
    EOF
  '';

  propagatedBuildInputs = [
    mujoco-py
    transforms3d
    gym
    numpy
    absl-py
  ];

  doCheck = false;
}
