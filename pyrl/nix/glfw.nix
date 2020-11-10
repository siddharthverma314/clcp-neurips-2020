{ callPackage
, buildPythonPackage
, fetchPypi
, glfw3
, python3
, sources
}:
buildPythonPackage rec {
  pname = "glfw";
  version = "1.12.0";
  src = sources.pyGLFW;
  preFixup = ''
    cat <<EOF > $out/lib/${python3.libPrefix}/site-packages/glfw/library.py
    import ctypes
    glfw = ctypes.CDLL("${glfw3}/lib/libglfw.so")
    EOF
  '';
  buildInputs = [ glfw3 ];
  doCheck = false;
}
