"""Find near-duplicate images in the dataset and copy them out for review.

Duplicates are detected per breed folder using a perceptual hash (the "diff
function") plus a Hamming-distance threshold, so resized / recompressed /
slightly edited copies are caught — not just byte-identical files.

Matching groups are copied (originals are left untouched) into:

    <output_dir>/<breed>/<NNNNN>_<L>_<originalname>.<ext>

where NNNNN is the per-breed group number and L is the member letter (A, B,
C, ...), so images in the same duplicate set share a number and differ by letter.
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


def _letter(n: int) -> str:
    """1->A, 2->B, ..., 26->Z, 27->AA, 28->AB, ... (Excel-style labels)."""
    label = ""
    while n > 0:
        n, rem = divmod(n - 1, 26)
        label = chr(ord("A") + rem) + label
    return label


class DuplicateFinder:
    # threshold is the max Hamming distance (out of hash_size**2 bits) at which
    # two images count as "the same". 0 = pixel-identical re-encodes only; ~2-3
    # catches resized/recompressed copies. Higher values over-group low-texture
    # breeds (e.g. solid-colour Muji), so review the output and tune per dataset.
    def __init__(self, data_dir, output_dir="duplicate", hash_size: int = 8,
                 threshold: int = 3, hash_fn=dhash_bits, verbose: bool = False):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.hash_size = hash_size
        self.threshold = threshold
        self.hash_fn = hash_fn
        self.verbose = verbose

    def _log(self, msg: str, end: str = "\n") -> None:
        if self.verbose:
            print(msg, end=end, flush=True)

    def _class_dirs(self):
        return sorted(p for p in self.data_dir.iterdir() if p.is_dir())

    def _images(self, class_dir):
        return sorted(p for p in class_dir.iterdir()
                      if p.suffix.lower() in IMG_EXTS)

    def _hashes(self, paths) -> np.ndarray:
        rows = []
        total = len(paths)
        for i, p in enumerate(paths, start=1):
            with Image.open(p) as im:
                rows.append(self.hash_fn(im, self.hash_size))
            if self.verbose and (i % 500 == 0 or i == total):
                self._log(f"\r    hashing {i}/{total}", end="")
        if self.verbose:
            self._log("")  # newline after the progress line
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
        class_dirs = self._class_dirs()
        for n, class_dir in enumerate(class_dirs, start=1):
            paths = self._images(class_dir)
            self._log(f"[{n}/{len(class_dirs)}] {class_dir.name}: {len(paths)} images")
            if len(paths) < 2:
                continue
            bits = self._hashes(paths)
            groups = [[paths[i] for i in idx] for idx in self._group(bits)]
            if groups:
                dup_imgs = sum(len(g) for g in groups)
                self._log(f"    -> {len(groups)} duplicate group(s), {dup_imgs} images")
                result[class_dir.name] = groups
        return result

    def copy_duplicates(self, duplicates: dict) -> int:
        """Copy every member of every duplicate group into output_dir. Returns
        the number of files copied. Originals are not modified.

        Files are named <NNNNN>_<L>_<originalname>.<ext>, where NNNNN is the
        per-breed group number and L is the member letter (A, B, C, ... and
        AA, AB, ... past 26), so one duplicate set shares a number and differs
        only by letter.
        """
        copied = 0
        for breed, groups in duplicates.items():
            out_class = self.output_dir / breed
            out_class.mkdir(parents=True, exist_ok=True)
            for gnum, group in enumerate(groups, start=1):
                for dnum, src in enumerate(group, start=1):
                    dst = out_class / f"{gnum:05d}_{_letter(dnum)}_{src.stem}{src.suffix}"
                    shutil.copy2(src, dst)
                    copied += 1
        return copied

    def remove_extras(self, duplicates: dict) -> int:
        """Delete all but the first image of each duplicate group from the
        dataset, keeping one representative. Returns the number deleted.

        Run copy_duplicates first so output_dir holds a backup of everything
        before anything is removed.
        """
        removed = 0
        for groups in duplicates.values():
            for group in groups:
                for src in group[1:]:
                    Path(src).unlink()
                    removed += 1
        return removed

    def run(self, apply: bool = False) -> dict:
        duplicates = self.find_duplicates()
        self._log(f"copying {sum(len(grp) for g in duplicates.values() for grp in g)} "
                  f"images to {self.output_dir} ...")
        copied = self.copy_duplicates(duplicates)
        if apply:
            self._log("removing extras (keeping 1 per group) ...")
        removed = self.remove_extras(duplicates) if apply else 0
        return {
            "groups": sum(len(g) for g in duplicates.values()),
            "duplicate_images": sum(len(grp) for g in duplicates.values() for grp in g),
            "copied": copied,
            "removed": removed,
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
    parser.add_argument("--apply", action="store_true",
                        help="delete all but one image per group from the dataset "
                             "(copies in --out remain as a backup); default is dry-run")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="print per-breed progress while scanning")
    args = parser.parse_args()

    data_dir = args.data_dir or load_config(args.config).data_dir
    finder = DuplicateFinder(data_dir, args.out, args.hash_size, args.threshold,
                             verbose=args.verbose)
    summary = finder.run(apply=args.apply)

    print(f"data_dir:   {data_dir}")
    print(f"groups:     {summary['groups']}")
    print(f"duplicates: {summary['duplicate_images']} images copied to {summary['output_dir']}")
    for breed, n in sorted(summary["by_breed"].items(), key=lambda x: -x[1]):
        print(f"  {breed:20s} {n} group(s)")
    if args.apply:
        print(f"removed:    {summary['removed']} duplicate images from {data_dir} "
              f"(kept 1 per group)")
    else:
        print("(dry run — pass --apply to remove extras from the dataset)")


if __name__ == "__main__":
    main()
