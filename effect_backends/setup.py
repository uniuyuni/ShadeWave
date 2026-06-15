from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext

try:
    import pybind11
except ImportError as exc:  # pragma: no cover - setup-time diagnostic.
    raise SystemExit("pybind11 is required to build effect_backends") from exc


class BuildExt(build_ext):
    def build_extensions(self):
        self.compiler.src_extensions.append(".mm")
        super().build_extensions()


setup(
    name="platypus-effect-backends",
    version="0.1.0",
    packages=["effect_backends"],
    package_dir={"effect_backends": "."},
    cmdclass={"build_ext": BuildExt},
    ext_modules=[
        Extension(
            "effect_backends._vignette_cpu",
            ["vignette_pybind.cpp", "vignette_cpu.c"],
            include_dirs=[pybind11.get_include()],
            language="c++",
            extra_compile_args=["-O3"],
        ),
        Extension(
            "effect_backends._cross_filter_cpu",
            ["cross_filter_pybind.cpp", "cross_filter_cpu.c"],
            include_dirs=[pybind11.get_include()],
            language="c++",
            extra_compile_args=["-O3"],
        ),
        Extension(
            "effect_backends._colour_functions_cpu",
            ["colour_functions_pybind.cpp", "colour_functions_cpu.c"],
            include_dirs=[pybind11.get_include()],
            language="c++",
            extra_compile_args=["-O3"],
        ),
        Extension(
            "effect_backends._tone_cpu",
            ["tone_pybind.cpp", "tone_cpu.c"],
            include_dirs=[pybind11.get_include()],
            language="c++",
            extra_compile_args=["-O3"],
        ),
        Extension(
            "effect_backends._cross_filter_metal",
            ["cross_filter_metal.mm"],
            include_dirs=[pybind11.get_include()],
            language="c++",
            extra_compile_args=["-O3", "-std=c++17", "-fobjc-arc"],
            extra_link_args=["-framework", "Metal", "-framework", "Foundation"],
        ),
        Extension(
            "effect_backends._image_transform_metal",
            ["image_transform_metal.mm"],
            include_dirs=[pybind11.get_include()],
            language="c++",
            extra_compile_args=["-O3", "-std=c++17", "-fobjc-arc"],
            extra_link_args=["-framework", "Metal", "-framework", "Foundation"],
        ),
    ],
)
