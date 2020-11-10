{ buildPythonPackage
, six
, pathlib2
, sources
, pytest
}:
buildPythonPackage {
  pname = "flatten-dict";
  version = "0.3.0";

  src = sources.flatten-dict;
  postPatch = ''
    mv src/flatten_dict ./
    cat > setup.py << EOF
    from setuptools import setup, find_packages

    setup(
      name='flatten-dict',
      version='0.0.0',
      author='Siddharth Verma',
      packages=find_packages(),
    )
    EOF
  '';
  propagatedBuildInputs = [ six pathlib2 ];

  checkInputs = [ pytest ];
  checkPhase = "pytest ./flatten_dict";
}
