"""
Dataset metadata, relevant for loading correct data.
"""

# This file is part of Echofilter.
#
# Copyright (C) 2020-2022  Scott C. Lowe and Offshore Energy Research Association (OERA)
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as
# published by the Free Software Foundation, version 3.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import os

import numpy as np


def recall_passive_edges(sample_path, timestamps):
    """
    Defines passive data edges for samples within known datasets.

    Parameters
    ----------
    sample_path : str
        Path to sample.
    timestamps : array_like vector
        Vector of timestamps in sample.

    Returns
    -------
    passive_starts : numpy.ndarray or None
        Indices indicating the onset of passive data collection periods, or
        `None` if passive metadata is unavailable for this sample.
    passive_ends : numpy.ndarray or None
        Indices indicating the offset of passive data collection periods, or
        `None` if passive metadata is unavailable for this sample.
    finder_version : absent or str
        If `passive_starts` and `passive_ends`, this string may be present to
        indicate which passive finder algorithm works best for this dataset.
    """

    sample_path = sample_path.lower()
    sample_parts = os.path.normpath(sample_path).split(os.path.sep)

    nt = len(timestamps)

    if "minaspassage" in sample_parts:
        if "december2017" in sample_path:
            return np.array([]), np.array([])
        elif "march2018" in sample_path:
            # No clear best detector, but all samples well behaved
            passive_starts = np.arange(0, nt, 360)
            passive_ends = passive_starts + 60
            return passive_starts, passive_ends
        elif "september2018" in sample_path:
            # Some samples are not as expected
            for name in [
                "D20181021-T165220_D20181021-T222221",
                "D20181022-T105220_D20181022-T162217",
                "D20181022-T172213_D20181022-T232217",
                "D20181026-T082220_D20181026-T135213",
                "D20181026-T142217_D20181026-T195218",
            ]:
                # Detector v1 finds the correct passive boundaries, so we leave
                # it to that to work it out
                if name in sample_path:
                    return None, None, "v1"
            passive_starts = np.arange(300, nt, 360)
            passive_ends = passive_starts + 60
            return passive_starts, passive_ends

    elif "grandpassage" in sample_parts:
        like_phase1 = False
        if "phase2" in sample_parts:
            # Some samples have the same passive data as in phase1
            for name in [
                "WBAT_2B_20200125_UTC100017_floodhigh",
                "WBAT_2B_20200125_UTC160020_ebblow",
                "WBAT_2B_20200127_UTC000020_floodhigh",
                "WBAT_2B_20200127_UTC060020_ebblow",
                "WBAT_2B_20200127_UTC120021_floodhigh",
                "WBAT_2B_20200127_UTC180020_ebblow",
                "WBAT_2B_20200128_UTC000017_floodhigh",
                "WBAT_2B_20200128_UTC060017_ebblow",
                "WBAT_2B_20200128_UTC120023_floodhigh",
                "WBAT_2B_20200128_UTC180017_ebblow",
                "WBAT_2B_20200130_UTC080017_ebblow",
                "WBAT_2B_20200130_UTC140020_floodhigh",
                "WBAT_2B_20200130_UTC200017_ebblow",
                "WBAT_2B_20200131_UTC020020_floodhigh",
                "WBAT_2B_20200131_UTC080021_ebblow",
                "WBAT_2B_20200131_UTC140022_floodhigh",
                "WBAT_2B_20200202_UTC040019_floodhigh",
                "WBAT_2B_20200202_UTC100022_ebblow",
                # "WBAT_2B_20200130_UTC020017_floodhigh",  # broken csv encoding
            ]:
                if name in sample_path:
                    like_phase1 = True
            # There are 5 other passive timing schedules across the other
            # 14 samples. The passive data finder works well for all of these.
            if not like_phase1:
                return None, None, "v1"

        if like_phase1 or "phase1" in sample_parts:
            passive_starts = np.arange(-200, nt, 3420)
            passive_ends = passive_starts + 420
            passive_starts = np.maximum(passive_starts, 0)
            passive_ends = np.minimum(passive_ends, nt)

    # For the mobile dataset, v1 and v2 always agree and appear to be 100%
    # accurate.

    return None, None
