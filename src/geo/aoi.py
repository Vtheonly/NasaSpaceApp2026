"""Area-of-interest (AOI) validation and filtering (challenge requirement R-01).

Two explicit AOI types are supported, both validated on construction with
clear error messages (graceful failure, issue #13):

- ``BBoxAOI``  - an axis-aligned longitude/latitude box.
- ``PolygonAOI`` - a simple polygon shell using an even-odd ray-casting test
  (no external geometry dependency required by the core pipeline).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import pandas as pd

from src.data.schema import COL_LATITUDE, COL_LONGITUDE

LON_MIN, LON_MAX = -180.0, 180.0
LAT_MIN, LAT_MAX = -90.0, 90.0


class AOIValidationError(ValueError):
    """Raised when an AOI specification is invalid."""


@dataclass(frozen=True)
class BBoxAOI:
    min_lon: float
    min_lat: float
    max_lon: float
    max_lat: float

    def __post_init__(self) -> None:
        for name, value, lo, hi in (
            ("min_lon", self.min_lon, LON_MIN, LON_MAX),
            ("max_lon", self.max_lon, LON_MIN, LON_MAX),
            ("min_lat", self.min_lat, LAT_MIN, LAT_MAX),
            ("max_lat", self.max_lat, LAT_MIN, LAT_MAX),
        ):
            if not (math.isfinite(value) and lo <= value <= hi):
                raise AOIValidationError(
                    f"AOI {name}={value!r} is outside the valid range [{lo}, {hi}]"
                )
        if self.min_lon > self.max_lon or self.min_lat > self.max_lat:
            raise AOIValidationError(
                "AOI min coordinates must not exceed max coordinates "
                f"(got lon [{self.min_lon}, {self.max_lon}], lat [{self.min_lat}, {self.max_lat}])"
            )

    @property
    def aoi_id(self) -> str:
        return (
            f"bbox_{self.min_lon:.3f}_{self.min_lat:.3f}"
            f"_{self.max_lon:.3f}_{self.max_lat:.3f}"
        )

    def contains(self, lon: float, lat: float) -> bool:
        nan = lon != lon or lat != lat  # NaN-safe comparison
        if nan:
            return False
        return (
            self.min_lon <= lon <= self.max_lon and self.min_lat <= lat <= self.max_lat
        )


@dataclass(frozen=True)
class PolygonAOI:
    shell: tuple[tuple[float, float], ...]

    def __post_init__(self) -> None:
        shell = tuple(self.shell)
        if len(shell) < 3:
            raise AOIValidationError("A polygon AOI needs at least 3 vertices")
        for lon, lat in shell:
            if not (math.isfinite(lon) and math.isfinite(lat)):
                raise AOIValidationError("Polygon vertices must be finite numbers")
            if not (LON_MIN <= lon <= LON_MAX and LAT_MIN <= lat <= LAT_MAX):
                raise AOIValidationError(
                    f"Polygon vertex ({lon}, {lat}) is outside valid coordinate ranges"
                )
        object.__setattr__(self, "shell", shell)

    @property
    def aoi_id(self) -> str:
        digest_input = "|".join(f"{lon:.4f},{lat:.4f}" for lon, lat in self.shell)
        import hashlib

        return f"poly_{hashlib.sha1(digest_input.encode()).hexdigest()[:12]}"

    def contains(self, lon: float, lat: float) -> bool:
        """Even-odd ray-casting point-in-polygon test (boundary-exclusive by
        numerical necessity; boundary points within 1e-12 are treated inside)."""
        if lon != lon or lat != lat:
            return False
        inside = False
        n = len(self.shell)
        for i in range(n):
            x1, y1 = self.shell[i]
            x2, y2 = self.shell[(i + 1) % n]
            # On-edge tolerance check.
            cross = (x2 - x1) * (lat - y1) - (y2 - y1) * (lon - x1)
            if (
                abs(cross) < 1e-12
                and min(x1, x2) - 1e-12 <= lon <= max(x1, x2) + 1e-12
                and min(y1, y2) - 1e-12 <= lat <= max(y1, y2) + 1e-12
            ):
                return True
            if (y1 > lat) != (y2 > lat):
                x_int = x1 + (lat - y1) * (x2 - x1) / (y2 - y1)
                if lon < x_int:
                    inside = not inside
        return inside


def parse_aoi(spec: str) -> BBoxAOI | PolygonAOI:
    """Parse an AOI from a string specification.

    Formats:
    - ``bbox:lon1,lat1,lon2,lat2``       (min corner, max corner)
    - ``polygon:lon1,lat1,lon2,lat2,...`` (>= 3 vertices)
    """
    if not isinstance(spec, str) or ":" not in spec:
        raise AOIValidationError(
            "AOI spec must look like 'bbox:lon1,lat1,lon2,lat2' or "
            f"'polygon:lon1,lat1,lon2,lat2,...'; got {spec!r}"
        )
    kind, _, coords_text = spec.partition(":")
    kind = kind.strip().lower()
    try:
        coords = [float(part) for part in coords_text.replace(" ", "").split(",") if part != ""]
    except ValueError as exc:
        raise AOIValidationError(f"AOI coordinates must be numeric; got {coords_text!r}") from exc
    if kind == "bbox":
        if len(coords) != 4:
            raise AOIValidationError(
                f"bbox AOI needs exactly 4 numbers (min_lon,min_lat,max_lon,max_lat); got {len(coords)}"
            )
        return BBoxAOI(*coords)
    if kind == "polygon":
        if len(coords) < 6 or len(coords) % 2 != 0:
            raise AOIValidationError(
                f"polygon AOI needs >= 3 lon/lat pairs (6 numbers); got {len(coords)} numbers"
            )
        shell = tuple((coords[i], coords[i + 1]) for i in range(0, len(coords), 2))
        return PolygonAOI(shell)
    raise AOIValidationError(f"Unknown AOI type {kind!r}; expected 'bbox' or 'polygon'")


def filter_records(frame: pd.DataFrame, aoi: BBoxAOI | PolygonAOI) -> pd.DataFrame:
    """Return rows whose coordinates fall inside the AOI (input never mutated)."""
    if frame.empty:
        return frame.copy()
    lons = pd.to_numeric(frame[COL_LONGITUDE], errors="coerce")
    lats = pd.to_numeric(frame[COL_LATITUDE], errors="coerce")
    if isinstance(aoi, BBoxAOI):
        mask = (
            lons.between(aoi.min_lon, aoi.max_lon, inclusive="both")
            & lats.between(aoi.min_lat, aoi.max_lat, inclusive="both")
        )
    else:
        mask = pd.Series(
            [aoi.contains(lon, lat) for lon, lat in zip(lons, lats, strict=False)], index=frame.index
        )
    return frame[mask].reset_index(drop=True)


__all__ = [
    "AOIValidationError",
    "BBoxAOI",
    "PolygonAOI",
    "parse_aoi",
    "filter_records",
]
