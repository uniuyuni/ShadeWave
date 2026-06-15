#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
現在アクティブな Python（pixi / venv を問わず `python` で解決される環境）を使い、
PyInstaller で macOS 用 .app をビルドする。

前提:
  pip install pyinstaller

使い方:
  cd /path/to/platypus
  pixi run python scripts/build_macos_app_pyinstaller.py
  # または
  python scripts/build_macos_app_pyinstaller.py

出力:
  dist/Shade Wave.app

注意:
  - これは「依存を可能な限り取り込む」第一歩です。libvips・libomp・Torch 周辺など、
    ネイティブライブラリは環境によって追加の --add-binary / フックが必要になることがあります。
  - 署名・公証は別作業です。
  - config.json などのユーザー編集設定は起動時に ~/Pictures/Shade Wave へコピーされます。
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import shutil
import subprocess
import sys
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _build_icns_with_pillow(png: Path, icns: Path) -> Path | None:
    try:
        from PIL import Image
    except Exception as e:
        print(f"警告: Pillow での .icns 生成を利用できません: {e}", file=sys.stderr)
        return None

    try:
        with Image.open(png) as image:
            image.save(
                icns,
                format="ICNS",
                sizes=[(16, 16), (32, 32), (64, 64), (128, 128), (256, 256), (512, 512), (1024, 1024)],
            )
    except Exception as e:
        print(f"警告: Pillow での .icns 生成に失敗: {e}", file=sys.stderr)
        return None

    print(f"アイコン生成(Pillow): {icns}", file=sys.stderr)
    return icns


def _build_icns_from_png(png: Path, out_dir: Path, base_name: str = "Shade Wave") -> Path | None:
    """指定された PNG から .icns を生成して返す。失敗時は None。"""
    if not png.is_file():
        print(f"警告: アイコン PNG が見つかりません: {png}", file=sys.stderr)
        return None
    out_dir.mkdir(parents=True, exist_ok=True)
    icns = out_dir / f"{base_name}.icns"
    if not shutil.which("sips") or not shutil.which("iconutil"):
        print("警告: sips/iconutil が見つからないため .icns を生成しません。", file=sys.stderr)
        return _build_icns_with_pillow(png, icns)

    iconset = out_dir / f"{base_name}.iconset"
    if iconset.exists():
        shutil.rmtree(iconset)
    iconset.mkdir()

    # (px, filename) — Apple 推奨 iconset の全エントリ
    entries = [
        (16, "icon_16x16.png"),
        (32, "icon_16x16@2x.png"),
        (32, "icon_32x32.png"),
        (64, "icon_32x32@2x.png"),
        (128, "icon_128x128.png"),
        (256, "icon_128x128@2x.png"),
        (256, "icon_256x256.png"),
        (512, "icon_256x256@2x.png"),
        (512, "icon_512x512.png"),
        (1024, "icon_512x512@2x.png"),
    ]
    for size, name in entries:
        dst = iconset / name
        try:
            subprocess.run(
                ["sips", "-z", str(size), str(size), str(png), "--out", str(dst)],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except subprocess.CalledProcessError as e:
            print(f"警告: sips でのリサイズに失敗 ({size}x{size}): {e}", file=sys.stderr)
            return None

    try:
        subprocess.run(
            ["iconutil", "-c", "icns", str(iconset), "-o", str(icns)],
            check=True,
        )
    except subprocess.CalledProcessError as e:
        print(f"警告: iconutil 失敗: {e}", file=sys.stderr)
        return _build_icns_with_pillow(png, icns)

    print(f"アイコン生成: {icns}", file=sys.stderr)
    return icns


def _strip_lensfun_duplicate_dylibs(app_path: Path) -> None:
    """
    lensfunpy が同梱する gettext/glib が opencv(gnutls) / libvips とバージョン衝突するため、
    conda lib/ に一本化するために重複ファイルを削除する。
    """
    dot = app_path / "Contents" / "Frameworks" / "lensfunpy" / "__dot__dylibs"
    if not dot.is_dir():
        return
    for name in (
        "libintl.8.dylib",
        "libglib-2.0.0.dylib",
        "libgobject-2.0.0.dylib",
        "libgio-2.0.0.dylib",
        "libgmodule-2.0.0.dylib",
    ):
        p = dot / name
        if p.is_file():
            p.unlink()
            print("lensfunpy 重複除去:", p, file=sys.stderr)


def _create_framework_lib_symlinks(app_path: Path) -> None:
    """
    Frameworks/lib/*.dylib への参照を期待する拡張向けに、Frameworks 直下へ symlink を作成する。
    """
    fw = app_path / "Contents" / "Frameworks"
    libdir = fw / "lib"
    if not libdir.is_dir():
        return
    for dylib in sorted(libdir.glob("*.dylib")):
        link = fw / dylib.name
        if not link.exists():
            try:
                link.symlink_to(Path("lib") / dylib.name)
            except OSError:
                pass


def _replace_pil_harfbuzz(app_path: Path) -> None:
    """PIL 同梱 harfbuzz を conda 版で置換し、pango/cairo とのシンボル不整合を回避する。"""
    src = Path(sys.prefix) / "lib" / "libharfbuzz.0.dylib"
    dst = app_path / "Contents" / "Frameworks" / "PIL" / "__dot__dylibs" / "libharfbuzz.0.dylib"
    if src.is_file() and dst.parent.is_dir():
        try:
            import shutil
            shutil.copy2(src, dst)
            print("PIL harfbuzz 置換:", dst, file=sys.stderr)
        except OSError as e:
            print(f"警告: PIL harfbuzz 置換失敗: {e}", file=sys.stderr)


def _patch_cv2_iconv_to_system(app_path: Path) -> None:
    """
    cv2/ffmpeg 系は _iconv を要求し、conda libiconv(gnu) だと失敗する場合がある。
    cv2 由来バイナリのみ /usr/lib/libiconv.2.dylib に書き換える。
    """
    fw = app_path / "Contents" / "Frameworks"
    rs = app_path / "Contents" / "Resources"

    targets: list[Path] = []
    if (fw / "cv2").is_dir():
        targets.extend((fw / "cv2").rglob("*.dylib"))
        targets.extend((fw / "cv2").rglob("*.so"))
    if (rs / "cv2").is_dir():
        targets.extend((rs / "cv2").rglob("*.dylib"))
        targets.extend((rs / "cv2").rglob("*.so"))

    # cv2 の ffmpeg 依存が root Resources に展開されるケースを拾う
    for pat in ("libav*.dylib", "libsw*.dylib", "libpostproc*.dylib"):
        targets.extend(rs.glob(pat))

    seen: set[Path] = set()
    for t in targets:
        if t in seen or not t.is_file():
            continue
        seen.add(t)
        try:
            out = subprocess.check_output(["otool", "-L", str(t)], text=True, stderr=subprocess.STDOUT)
        except Exception:
            continue
        if "@rpath/libiconv.2.dylib" in out:
            try:
                subprocess.run(
                    [
                        "install_name_tool",
                        "-change",
                        "@rpath/libiconv.2.dylib",
                        "/usr/lib/libiconv.2.dylib",
                        str(t),
                    ],
                    check=True,
                )
                print("cv2 iconv 参照書換:", t, file=sys.stderr)
            except Exception as e:
                print(f"警告: cv2 iconv 書換失敗 {t}: {e}", file=sys.stderr)


def _conda_lib_dylib_bundle_args() -> list[str]:
    """
    pixi/conda の lib/*.dylib をバンドル内 lib/ にコピー（pyvips / libvips 依存用）。
    全体で数百 MB になるが、@rpath 解決のためまとめて同梱する。
    """
    out: list[str] = []
    libdir = Path(sys.prefix) / "lib"
    if not libdir.is_dir():
        print("警告: sys.prefix/lib がありません:", libdir, file=sys.stderr)
        return out
    for p in sorted(libdir.glob("*.dylib")):
        out.extend(["--add-binary", _add_data_mac(p, "lib")])
    return out


def _ensure_pyinstaller() -> None:
    if importlib.util.find_spec("PyInstaller") is None:
        print(
            "PyInstaller が見つかりません。現在の環境で次を実行してください:\n"
            "  pip install pyinstaller",
            file=sys.stderr,
        )
        sys.exit(1)


def _add_data_mac(src: Path, dest_in_bundle: str) -> str:
    """macOS の --add-data は 'src:dest'（dest はバンドル内の相対パス）。"""
    return f"{src}:{dest_in_bundle}"


def _kivy_pyinstaller_flags() -> list[str]:
    """
    Kivy 公式 pyinstaller_hooks を使う（--collect-all kivy は kivy.garden で失敗するため使わない）。
    注: Kivy 既定フックは tkinter を除外する。処理中 HUD は macOS では PyObjC（processing_dialog → macos.MacOSProcessingOverlay）。frozen 時も assets/spinner.gif と pyobjc を同梱すれば利用可。
    """
    from kivy.tools.packaging import pyinstaller_hooks as kh

    extra: list[str] = []
    extra.extend(["--additional-hooks-dir", str(kh.hookspath()[0])])
    for rh in kh.runtime_hooks():
        extra.extend(["--runtime-hook", rh])
    # 動画/音声/GStreamer は不要なら除外してバンドル縮小
    deps = kh.get_deps_minimal(video=None, audio=None)
    for mod in deps["hiddenimports"]:
        extra.extend(["--hidden-import", mod])
    for ex in deps.get("excludes", []):
        extra.extend(["--exclude-module", ex])
    return extra


def _build_args(root: Path, name: str, bundle_id: str, icon: Path | None) -> list[str]:
    datas: list[str] = []

    # ルートの KV / JSON（実行時は main.py 側で sys._MEIPASS に chdir）
    for rel in (
        "main.kv",
        "film_presets.json",
        "export_presets.json",
        "config.json",
    ):
        p = root / rel
        if p.is_file():
            datas.append(_add_data_mac(p, "."))
        else:
            print(f"警告: 見つかりません（スキップ）: {p}", file=sys.stderr)

    presets_dir = root / "presets"
    if presets_dir.is_dir():
        for preset in sorted(presets_dir.rglob("*")):
            if preset.is_file():
                dest_dir = str(preset.parent.relative_to(root))
                datas.append(_add_data_mac(preset, dest_dir))

    widgets_dir = root / "widgets"
    if widgets_dir.is_dir():
        for kv in sorted(widgets_dir.rglob("*.kv")):
            # バンドル内はリポジトリと同じ相対パス（例: widgets/foo.kv, widgets/sub/bar.kv）
            dest_dir = str(kv.parent.relative_to(root))
            datas.append(_add_data_mac(kv, dest_dir))

    # libraw_enhanced の Metal シェーダー（GPUパス）
    metal_dir = root / "metal"
    if not metal_dir.is_dir():
        alt_metal = root / "libraw_enhanced" / "core" / "metal"
        if alt_metal.is_dir():
            metal_dir = alt_metal
    if metal_dir.is_dir():
        for f in sorted(metal_dir.rglob("*")):
            if f.is_file():
                rel = f.relative_to(metal_dir)
                dest = str(Path("metal") / rel.parent) if rel.parent != Path(".") else "metal"
                datas.append(_add_data_mac(f, dest))
    else:
        print("注意: metal/ が見つからないため GPU シェーダーを同梱しません。", file=sys.stderr)

    assets_dir = root / "assets"
    if assets_dir.is_dir():
        for f in sorted(assets_dir.rglob("*")):
            if f.is_file():
                rel = f.relative_to(assets_dir)
                dest = str(Path("assets") / rel.parent) if rel.parent != Path(".") else "assets"
                datas.append(_add_data_mac(f, dest))
    else:
        print(
            "注意: assets/ がありません。processing_dialog の GIF 等は同梱されません。",
            file=sys.stderr,
        )

    # 重複除去（同一ファイルの複数指定を防ぐ）
    seen: set[str] = set()
    uniq_datas: list[str] = []
    for d in datas:
        if d not in seen:
            seen.add(d)
            uniq_datas.append(d)

    hidden = [
        "kivymd",
        "PIL",
        "PIL._imagingtk",
    ]

    rth_libintl = root / "scripts" / "pyinstaller" / "rth_darwin_libintl.py"
    if not rth_libintl.is_file():
        print("警告: rth_darwin_libintl がありません:", rth_libintl, file=sys.stderr)

    args: list[str] = [
        str(root / "main.py"),
        "--name",
        name,
        "--windowed",
        "--onedir",
        "--noconfirm",
        "--clean",
        "--log-level=WARN",
        "--osx-bundle-identifier",
        bundle_id,
        f"--paths={root}",
        "--noupx",
    ]

    if icon is not None:
        args.extend(["--icon", str(icon)])

    # cv2 / lensfunpy の libintl 競合対策（main より前に実行）
    if rth_libintl.is_file():
        args.extend(["--runtime-hook", str(rth_libintl)])

    print("conda lib の .dylib を同梱しています（サイズ増のため数分かかることがあります）…", file=sys.stderr)
    args.extend(_conda_lib_dylib_bundle_args())

    # Kivy: 公式フック + hiddenimports（collect-all は使用しない）
    args.extend(_kivy_pyinstaller_flags())

    for h in hidden:
        args.extend(["--hidden-import", h])

    # KivyMD: サブモジュールとデータ（アセット）
    args.extend(["--collect-submodules", "kivymd"])
    args.extend(["--collect-data", "kivymd"])

    # pyvips（C 拡張 _libvips とバイナリ）
    args.extend(["--collect-all", "pyvips"])
    args.extend(["--hidden-import", "_libvips"])

    # OpenEXR: .EXR export uses the official Python bindings directly.
    args.extend(["--collect-all", "OpenEXR"])
    args.extend(["--hidden-import", "OpenEXR"])

    for d in uniq_datas:
        args.extend(["--add-data", d])

    return args


def main() -> None:
    parser = argparse.ArgumentParser(description="PyInstaller で Shade Wave.app をビルドする")
    parser.add_argument(
        "--name",
        default="Shade Wave",
        help=".app の製品名（dist/<name>.app）",
    )
    parser.add_argument(
        "--bundle-id",
        default="com.uniuyuni.platypus",
        help="CFBundleIdentifier に使う文字列（既定は内部互換のため platypus のまま）",
    )
    parser.add_argument(
        "--icon-png",
        type=Path,
        default=None,
        help="アプリアイコン元 PNG（既定: assets/Shade Wave icon.png）",
    )
    parser.add_argument(
        "--distpath",
        type=Path,
        default=None,
        help="出力 dist ディレクトリ（既定: リポジトリ直下の dist）",
    )
    parser.add_argument(
        "--workpath",
        type=Path,
        default=None,
        help="作業用 build ディレクトリ（既定: リポジトリ直下の build）",
    )
    args = parser.parse_args()

    root = _repo_root()
    os.chdir(root)
    kivy_home = root / "build" / "kivy_home"
    kivy_home.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("KIVY_HOME", str(kivy_home))

    _ensure_pyinstaller()

    # 前回ビルドの .spec が残っていると PyInstaller がそれを優先し、古い設定のままになる
    spec = root / f"{args.name}.spec"
    if spec.is_file():
        spec.unlink()
        print("既存の spec を削除:", spec)

    distpath = args.distpath or (root / "dist")
    workpath = args.workpath or (root / "build")

    icon_png = args.icon_png or (root / "assets" / "Shade Wave icon.png")
    if not icon_png.is_absolute():
        icon_png = root / icon_png
    if not icon_png.is_file():
        fallback_icon = root / "assets" / "icon.png"
        print(f"警告: 指定アイコンが見つからないため旧アイコンを使用します: {fallback_icon}", file=sys.stderr)
        icon_png = fallback_icon
    icon_path = _build_icns_from_png(icon_png, workpath, args.name)

    pyi_args = _build_args(root, args.name, args.bundle_id, icon_path)
    pyi_args.extend(
        [
            f"--distpath={distpath}",
            f"--workpath={workpath}",
        ]
    )

    print("使用 Python:", sys.executable)
    print("リポジトリ:", root)
    print("PyInstaller へ渡す引数（要約）: main.py --windowed --onedir ...")
    print()

    # python -m PyInstaller として実行（現在の環境の site-packages を確実に使う）
    cmd = [sys.executable, "-m", "PyInstaller", *pyi_args]
    result = subprocess.run(cmd, cwd=root)
    if result.returncode != 0:
        sys.exit(result.returncode)

    app = distpath / f"{args.name}.app"
    if app.is_dir():
        _strip_lensfun_duplicate_dylibs(app)
        _replace_pil_harfbuzz(app)
        _patch_cv2_iconv_to_system(app)
        _create_framework_lib_symlinks(app)
        print()
        print("ビルド完了:", app)
    else:
        print("警告: 想定パスに .app が見つかりません:", app, file=sys.stderr)


if __name__ == "__main__":
    main()
