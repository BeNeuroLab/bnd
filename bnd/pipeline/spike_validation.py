"""
Module for validating spike times in NWB files to check alignment issues
"""

import os
from pathlib import Path
from typing import Tuple, Optional

import numpy as np
from pynwb import NWBHDF5IO
from rich import print

from ..logger import set_logging

logger = set_logging(__name__)


def check_negative_spike_times(nwb_path: Path) -> Tuple[bool, Optional[float]]:
    """
    Check if there are negative spike times in an NWB file.
    
    Parameters
    ----------
    nwb_path : Path
        Path to the NWB file to check
        
    Returns
    -------
    Tuple[bool, Optional[float]]
        (has_negative_spikes, min_spike_time)
        - has_negative_spikes: True if negative spike times exist
        - min_spike_time: The minimum spike time found (None if no spikes)
    """
    try:
        with NWBHDF5IO(nwb_path, mode="r") as io:
            nwbfile = io.read()
            
            # Check if ecephys processing module exists
            if not hasattr(nwbfile, "processing"):
                logger.warning(f"No processing module in {nwb_path.name}")
                return False, None
                
            if "ecephys" not in nwbfile.processing:
                logger.warning(f"No ecephys data in {nwb_path.name}")
                return False, None
            
            # Check all probe units for negative spike times
            ecephys = nwbfile.processing["ecephys"].data_interfaces
            min_spike_time = float('inf')
            has_spikes = False
            
            for probe_name, probe_units in ecephys.items():
                if hasattr(probe_units, 'spike_times'):
                    spike_times = probe_units.spike_times[:]
                    if len(spike_times) > 0:
                        has_spikes = True
                        probe_min = np.min(spike_times)
                        min_spike_time = min(min_spike_time, probe_min)
                        
                        if probe_min < 0:
                            logger.info(
                                f"Found negative spike times in {probe_name}: "
                                f"min = {probe_min:.3f}s"
                            )
            
            if not has_spikes:
                logger.warning(f"No spike times found in {nwb_path.name}")
                return False, None
            
            has_negative = min_spike_time < 0
            return has_negative, min_spike_time
            
    except Exception as e:
        logger.error(f"Error reading NWB file {nwb_path.name}: {str(e)}")
        raise


def validate_and_clean_session(
    session_path: Path,
    force_rerun: bool = False,
    delete_if_no_negative: bool = True
) -> bool:
    """
    Validate spike times in a session and optionally clean/re-run if needed.
    
    Parameters
    ----------
    session_path : Path
        Path to the session directory
    force_rerun : bool
        Force re-run of conversion even if negative spikes exist
    delete_if_no_negative : bool
        Delete NWB and pyaldata files if no negative spikes found
        
    Returns
    -------
    bool
        True if session needs re-conversion, False otherwise
    """
    # Check for NWB file
    nwb_path = session_path / f"{session_path.name}.nwb"
    if not nwb_path.exists():
        logger.warning(f"No NWB file found for session {session_path.name}")
        return True  # Needs conversion
    
    # Check for negative spike times
    has_negative, min_spike_time = check_negative_spike_times(nwb_path)
    
    if has_negative:
        logger.info(
            f"✓ Session {session_path.name} has negative spike times "
            f"(min = {min_spike_time:.3f}s). Alignment looks correct."
        )
        return False  # No re-conversion needed
    
    elif min_spike_time is None:
        logger.warning(
            f"Session {session_path.name} has no spike data to validate"
        )
        return False  # Can't validate, don't delete
    
    else:
        logger.warning(
            f"⚠ Session {session_path.name} has NO negative spike times "
            f"(min = {min_spike_time:.3f}s). This suggests alignment issues."
        )
        
        if delete_if_no_negative or force_rerun:
            print(f"\n[red]⚠ IMPORTANT: Re-conversion requires raw ephys data[/red]")
            print(f"The alignment issue was in NWB conversion, which needs access to")
            print(f"raw recording files (.meta files) to select the correct recording.")
            print(f"")
            print(f"[yellow]Before proceeding, ensure you have downloaded raw data:[/yellow]")
            print(f"  bnd dl {session_path.name}")
            print(f"")
            
            # Check if raw ephys data exists locally
            has_raw_ephys = any(session_path.rglob("*.meta")) or any(session_path.rglob("*_g?"))
            
            if not has_raw_ephys:
                print(f"[red]✗ Raw ephys data not found locally[/red]")
                print(f"Please download first: [bold]bnd dl {session_path.name}[/bold]")
                print(f"Then re-run: [bold]bnd to-pyal {session_path.name} -v -K[/bold]")
                return False
            else:
                print(f"[green]✓ Raw ephys data found locally[/green]")
            
            # Delete existing files
            files_to_delete = []
            
            # NWB file
            if nwb_path.exists():
                files_to_delete.append(nwb_path)
            
            # PyAlData files (could be partitioned)
            mat_files = list(session_path.glob("*_pyaldata*.mat"))
            files_to_delete.extend(mat_files)
            
            if files_to_delete:
                print(f"\n[yellow]Files to be deleted:[/yellow]")
                for file in files_to_delete:
                    print(f"  - {file.name}")
                
                # Confirm deletion
                response = input(
                    "\nDelete these files and re-run conversion? (y/n): "
                ).strip().lower()
                
                if "y" in response:
                    for file in files_to_delete:
                        os.remove(file)
                        logger.info(f"Deleted {file.name}")
                    print(f"\n[green]Files deleted. Re-conversion will proceed...[/green]")
                    return True  # Needs re-conversion
                else:
                    logger.info("Files not deleted. Keeping current data.")
                    print(f"\n[yellow]To manually re-convert later:[/yellow]")
                    print(f"  bnd to-pyal {session_path.name} -K")
                    return False
            
        return False


def batch_validate_sessions(
    sessions: list[Path],
    delete_if_no_negative: bool = True,
    summary_only: bool = False
) -> dict:
    """
    Validate multiple sessions and provide a summary.
    
    Parameters
    ----------
    sessions : list[Path]
        List of session paths to validate
    delete_if_no_negative : bool
        Delete files if no negative spikes found
    summary_only : bool
        Only show summary without making changes
        
    Returns
    -------
    dict
        Summary of validation results
    """
    results = {
        "valid": [],
        "invalid": [],
        "no_data": [],
        "missing_nwb": []
    }
    
    for session_path in sessions:
        nwb_path = session_path / f"{session_path.name}.nwb"
        
        if not nwb_path.exists():
            results["missing_nwb"].append(session_path.name)
            continue
        
        try:
            has_negative, min_spike_time = check_negative_spike_times(nwb_path)
            
            if min_spike_time is None:
                results["no_data"].append(session_path.name)
            elif has_negative:
                results["valid"].append((session_path.name, min_spike_time))
            else:
                results["invalid"].append((session_path.name, min_spike_time))
                
                if not summary_only and delete_if_no_negative:
                    validate_and_clean_session(
                        session_path,
                        delete_if_no_negative=True
                    )
                    
        except Exception as e:
            logger.error(f"Error validating {session_path.name}: {str(e)}")
    
    # Print summary
    print("\n[bold]Validation Summary:[/bold]")
    print(f"✓ Valid (with negative spikes): {len(results['valid'])}")
    print(f"✗ Invalid (no negative spikes): {len(results['invalid'])}")
    print(f"○ No spike data: {len(results['no_data'])}")
    print(f"□ Missing NWB: {len(results['missing_nwb'])}")
    
    if results["invalid"] and not summary_only:
        print(f"\n[yellow]Sessions needing re-conversion:[/yellow]")
        for session_name, min_time in results["invalid"]:
            print(f"  - {session_name} (min spike time: {min_time:.3f}s)")
    
    return results