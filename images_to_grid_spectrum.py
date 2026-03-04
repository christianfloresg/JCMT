import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageOps


# ----------------------------
# 1) Read hand-made order file
# ----------------------------
def read_source_order(txt_path):
    """
    Read a text file with one source per line.
    Blank lines and lines starting with # are ignored.
    """
    order = []
    with open(txt_path, "r") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            order.append(s)
    return order


# -----------------------------------
# 2) Map "source name" -> "PNG path"
# -----------------------------------
def build_image_index_from_order(spectra_dir, which, order_txt, pattern="*.png"):
    """
    Build {source_name: image_path} by matching filenames against the
    exact source names in the order file. Works with underscores in names.
    """
    which = which.lower().strip()
    if which not in ("central", "fov"):
        raise ValueError("which must be 'central' or 'fov'")

    desired_sources = read_source_order(order_txt)

    # Sort longest-first so e.g. "GV_Tau" matches before "GV" (just in case)
    desired_sources = sorted(desired_sources, key=len, reverse=True)

    paths = glob.glob(os.path.join(spectra_dir, pattern))

    idx = {}
    for p in paths:
        base = os.path.basename(p)
        b = base.lower()

        # must be the right type
        if f"_{which}" not in b:
            continue
        if not base.startswith("spectrum_"):
            continue

        # try to match "spectrum_<SOURCE>_" exactly
        for src in desired_sources:
            prefix = f"spectrum_{src}_"
            if base.startswith(prefix):
                idx[src] = p
                break

    return idx

def order_image_paths(spectra_dir, which, order_txt, pattern="*.png", append_unlisted=False):
    idx = build_image_index_from_order(spectra_dir, which=which, order_txt=order_txt, pattern=pattern)
    desired = read_source_order(order_txt)

    ordered = []
    missing_sources = []
    for src in desired:
        if src in idx:
            ordered.append(idx[src])
        else:
            missing_sources.append(src)

    if missing_sources:
        print("\nSources listed in order file but no matching image found:")
        for m in missing_sources:
            print("  ", m)

    if append_unlisted:
        # if you really want this, we'd need a separate index of ALL images;
        # but for hand-ordered grids it's usually best to keep False.
        pass

    if len(ordered) == 0:
        raise FileNotFoundError(
            f"No matching images found using order file '{order_txt}' in '{spectra_dir}' for which='{which}'."
        )

    return ordered

# ----------------------------
# 3) Image padding helper
# ----------------------------
def pad_to_square(img, fill_color="white"):
    max_dim = max(img.size)
    padding = [(max_dim - s) // 2 for s in img.size]
    extra = [(max_dim - s) % 2 for s in img.size]
    return ImageOps.expand(
        img,
        (padding[0], padding[1], padding[0] + extra[0], padding[1] + extra[1]),
        fill=fill_color
    )


# -----------------------------------------
# 4) Plot grid FROM PATHS (the key change!)
# -----------------------------------------
def plot_spectra_grid_from_paths(
    image_paths,
    grid_shape=(7, 5),
    output_file=None,
    pad_and_resize=True,
    resize_to=(900, 900),
    show_titles=False,
    title_fontsize=7
):
    rows, cols = grid_shape
    capacity = rows * cols

    if len(image_paths) > capacity:
        print(f"Warning: {len(image_paths)} images but grid holds {capacity}. Plotting first {capacity}.")
        image_paths = image_paths[:capacity]

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.0, rows * 3.0))
    axes = np.array(axes).flatten()

    # Hide all axes first
    for ax in axes:
        ax.axis("off")

    def title_from_path(path):
        # show just the <source> part if possible
        base = os.path.basename(path)
        if base.startswith("spectrum_"):
            parts = base.split("_")
            if len(parts) >= 2:
                return parts[1]
        return os.path.splitext(base)[0]

    for ax, path in zip(axes, image_paths):
        img = Image.open(path).convert("RGB")
        if pad_and_resize:
            img = pad_to_square(img).resize(resize_to)

        ax.imshow(np.asarray(img))
        ax.set_aspect("equal")
        ax.axis("off")

        if show_titles:
            ax.set_title(title_from_path(path), fontsize=title_fontsize)

    plt.subplots_adjust(wspace=0.01, hspace=0.01)

    if output_file:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        print(f"Saved grid to: {output_file}")

    plt.show()


# ----------------------------
# 5) Main
# ----------------------------
if __name__ == "__main__":
    spectra_dir = "./Figures/Spectra/both_molec"   # change if needed
    order_txt   = "text_files/source_order.txt"              # your hand-made list
    which       = "central"                       # or "fov"
    grid_shape  = (7, 5)
    out_png     = f"./Figures/grid_plots/spectra_grid_{which}_ordered.png"

    paths = order_image_paths(
        spectra_dir=spectra_dir,
        which=which,
        order_txt=order_txt,
        pattern="*.png",
        append_unlisted=False   # set True if you want extras appended
    )

    plot_spectra_grid_from_paths(
        image_paths=paths,
        grid_shape=grid_shape,
        output_file=out_png,
        pad_and_resize=True,
        resize_to=(900, 900),
        show_titles=False
    )
