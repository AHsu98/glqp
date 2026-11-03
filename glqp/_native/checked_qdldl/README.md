# Private checked QDLDL source

Selected source files come from the official qdldl-python 0.1.9.post1 source
archive identified in `upstream.json`. Original hashes and the QDLDL/Python/AMD
licenses are retained. Every C kernel/header remains unchanged.

Local changes are limited to C++ ownership/error handling and the private Python
binding: checked numeric-update status, invalid-state rejection, constructor
cleanup, validated direct matrix inputs and a value-only adapter update. The
binding has a distinct module and module-local class registration. The separate
CMake file and Python loader control an explicit, source-hashed private build.

See `docs/checked-qdldl.md` in the repository for build, freezing and failure
contracts. Do not replace the public `qdldl` dependency when comparing revisions.
