"""Find near-duplicate images in the dataset and copy them out for review.

Duplicates are detected per breed folder using a perceptual hash (the "diff
function") plus a Hamming-distance threshold, so resized / recompressed /
slightly edited copies are caught — not just byte-identical files.

Matching groups are copied (originals are left untouched) into:

    <output_dir>/<breed>/<NNNNN>_dup_<II>_<originalname>.<ext>

where NNNNN is the per-breed group number and II is the index within the group.
"""
import argparse
import shutil
from pathlib import Path

import numpy as np
from PIL import Image

from koi.config import load_config

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def dhash_bits(image: Image.Image, hash_size: int = 8) -> np.ndarray:
    """Difference hash: a (hash_size*hash_size,) bool array.

    Compares each pixel to its horizontal neighbour on a downscaled grayscale
    image, so it is robust to scaling and re-compression. This is the default
    "diff function"; pass a different callable to DuplicateFinder to change how
    similarity is defined.
    """
    img = image.convert("L").resize((hash_size + 1, hash_size))
    arr = np.asarray(img, dtype=np.int16)
    return (arr[:, 1:] > arr[:, :-1]).flatten()


class DuplicateFinder:
    # threshold is the max Hamming distance (out of hash_size**2 bits) at which
    # two images count as "the same". 0 = pixel-identical re-encodes only; ~2-3
    # catches resized/recompressed copies. Higher values over-group low-texture
    # breeds (e.g. solid-colour Muji), so review the output and tune per dataset.
    def __init__(self, data_dir, output_dir="duplicate", hash_size: int = 8,
                 threshold: int = 3, hash_fn=dhash_bits):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.hash_size = hash_size
        self.threshold = threshold
        self.hash_fn = hash_fn

    def _class_dirs(self):
        return sorted(p for p in self.data_dir.iterdir() if p.is_dir())

    def _images(self, class_dir):
        return sorted(p for p in class_dir.iterdir()
                      if p.suffix.lower() in IMG_EXTS)

    def _hashes(self, paths) -> np.ndarray:
        rows = []
        for p in paths:
            with Image.open(p) as im:
                rows.append(self.hash_fn(im, self.hash_size))
        return np.array(rows, dtype=bool)

    def _group(self, bits: np.ndarray):
        """Union-find over pairs within the Hamming threshold."""
        n = len(bits)
        parent = list(range(n))

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        for i in range(n - 1):
            dist = np.count_nonzero(bits[i] != bits[i + 1:], axis=1)
            for off in np.nonzero(dist <= self.threshold)[0]:
                a, b = find(i), find(i + 1 + int(off))
                if a != b:
                    parent[b] = a

        clusters: dict[int, list[int]] = {}
        for i in range(n):
            clusters.setdefault(find(i), []).append(i)
        return [sorted(idx) for idx in clusters.values() if len(idx) > 1]

    def find_duplicates(self) -> dict:
        """Return {breed: [[Path, ...] per duplicate group]} (groups of >= 2)."""
        result = {}
        for class_dir in self._class_dirs():
            paths = self._images(class_dir)
            if len(paths) < 2:
                continue
            bits = self._hashes(paths)
            groups = [[paths[i] for i in idx] for idx in self._group(bits)]
            if groups:
                result[class_dir.name] = groups
        return result

    def copy_duplicates(self, duplicates: dict) -> int:
        """Copy every member of every duplicate group into output_dir. Returns
        the number of files copied. Originals are not modified."""
        copied = 0
        for breed, groups in duplicates.items():
            out_class = self.output_dir / breed
            out_class.mkdir(parents=True, exist_ok=True)
            for gnum, group in enumerate(groups, start=1):
                for dnum, src in enumerate(group, start=1):
                    dst = out_class / f"{gnum:05d}_dup_{dnum:02d}_{src.stem}{src.suffix}"
                    shutil.copy2(src, dst)
                    copied += 1
        return copied

    def run(self) -> dict:
        duplicates = self.find_duplicates()
        copied = self.copy_duplicates(duplicates)
        return {
            "groups": sum(len(g) for g in duplicates.values()),
            "duplicate_images": sum(len(grp) for g in duplicates.values() for grp in g),
            "copied": copied,
            "by_breed": {b: len(g) for b, g in duplicates.items()},
            "output_dir": str(self.output_dir),
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--data-dir", default=None,
                        help="defaults to data_dir from --config")
    parser.add_argument("--out", default="duplicate")
    parser.add_argument("--threshold", type=int, default=3,
                        help="max Hamming distance to treat images as the same "
                             "(0=identical only; higher over-groups low-texture breeds)")
    parser.add_argument("--hash-size", type=int, default=8)
    args = parser.parse_args()

    data_dir = args.data_dir or load_config(args.config).data_dir
    finder = DuplicateFinder(data_dir, args.out, args.hash_size, args.threshold)
    summary = finder.run()

    print(f"data_dir:   {data_dir}")
    print(f"groups:     {summary['groups']}")
    print(f"duplicates: {summary['duplicate_images']} images copied to {summary['output_dir']}")
    for breed, n in sorted(summary["by_breed"].items(), key=lambda x: -x[1]):
        print(f"  {breed:20s} {n} group(s)")


if __name__ == "__main__":
    main()
