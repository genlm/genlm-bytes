from pathlib import Path
import shutil
import mkdocs_gen_files

readme = Path("README.md")
index_md = Path("docs/index.md")
logo = Path("assets/logo.png")


def files_are_different(src: Path, dst: Path) -> bool:
    if not dst.exists():
        return True
    return src.read_bytes() != dst.read_bytes()


if readme.exists() and files_are_different(readme, index_md):
    shutil.copyfile(readme, index_md)

if logo.exists() and files_are_different(logo, Path("docs/assets/logo.png")):
    shutil.copyfile(logo, "docs/assets/logo.png")

nav = mkdocs_gen_files.Nav()

for path in sorted(Path("genlm/bytes").rglob("*.py")):
    if any(part.startswith(".") for part in path.parts):
        continue

    module_path = path.relative_to(".").with_suffix("")
    doc_path = path.relative_to(".").with_suffix(".md")
    full_doc_path = Path("reference", doc_path)

    parts = tuple(module_path.parts)

    if parts[-1] == "__init__":
        print(f"init, making parts {parts[:-1]}")
        parts = parts[:-1]
    elif parts[-1] == "__main__":
        continue

    nav[parts] = doc_path.as_posix()

    with mkdocs_gen_files.open(full_doc_path, "w") as fd:
        ident = ".".join(parts)
        fd.write(f"::: {ident}\n")
        fd.write("    options:\n")
        fd.write("      show_root_heading: true\n")
        fd.write("      show_source: true\n")
        fd.write("      heading_level: 2\n")

    mkdocs_gen_files.set_edit_path(full_doc_path, path)

with mkdocs_gen_files.open("reference/SUMMARY.md", "w") as nav_file:
    nav_file.writelines(nav.build_literate_nav())
