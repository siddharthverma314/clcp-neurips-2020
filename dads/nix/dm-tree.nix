{ six
, buildPythonPackage
, fetchurl
}:
let
  pname = "dm-tree";
  version = "0.1.5";
in
buildPythonPackage {
  inherit pname version;
  format = "wheel";

  src = fetchurl {
    url = "https://files.pythonhosted.org/packages/6b/d9/6d88e8d32bb454c4ef8f50c62714b0eb20170f4c1d2cd316e0d99755405e/dm_tree-0.1.5-cp37-cp37m-manylinux1_x86_64.whl";
    sha256 = "0s89bxvlxh7aw8ssi3s0dfdj30b5mrpdzfism41ks3wfxdrq2vmy";
  };

  propagatedBuildInputs = [ six ];
}
