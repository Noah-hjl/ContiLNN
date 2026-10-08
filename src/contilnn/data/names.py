#!/usr/bin/env python3
# Editor: Jialei.He
"""Pure filename grammars for CT, MRI, and PET slices."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class ParsedSliceName:
    case_id: str
    slice_index: int
    auxiliary_index: Optional[int]


@dataclass(frozen=True)
class ParsedPetName:
    patient: str
    z: int
    patch: Optional[int]


CT_PATTERN = re.compile(r"^(L\d+)_(\d+)(?:_(\d+))?$")
MRI_PATTERN = re.compile(r"^(IXI\d+-(?:Guys|HH|IOP)-\d+-T2)_(\d+)$")
PET_PATTERN = re.compile(r"^(Patient_\d+)_(\d+)(?:_(\d+))?$")


def parse_ct_stem(stem: str) -> ParsedSliceName:
    match = CT_PATTERN.fullmatch(stem)
    if match is None:
        raise ValueError(
            f"Unexpected CT filename {stem!r}; expected L<case>_<slice>[_<patch>]"
        )
    auxiliary = None if match.group(3) is None else int(match.group(3))
    return ParsedSliceName(match.group(1), int(match.group(2)), auxiliary)


def parse_mri_stem(stem: str) -> ParsedSliceName:
    match = MRI_PATTERN.fullmatch(stem)
    if match is None:
        raise ValueError(
            f"Unexpected MRI filename {stem!r}; "
            "expected IXI...-(Guys|HH|IOP)-...-T2_<slice>"
        )
    return ParsedSliceName(match.group(1), int(match.group(2)), None)


def parse_pet_stem(stem: str) -> ParsedPetName:
    match = PET_PATTERN.fullmatch(stem)
    if match is None:
        raise ValueError(
            f"Unexpected PET filename {stem!r}; expected Patient_<case>_<z>[_<patch>]"
        )
    patch = None if match.group(3) is None else int(match.group(3))
    return ParsedPetName(match.group(1), int(match.group(2)), patch)


__all__ = [
    "ParsedSliceName",
    "ParsedPetName",
    "parse_ct_stem",
    "parse_mri_stem",
    "parse_pet_stem",
]
