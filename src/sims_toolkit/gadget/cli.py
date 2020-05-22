import os
import pathlib
import typing as t
from errno import ENOENT
from functools import reduce
from itertools import filterfalse

import attr
import click
import numpy as np
import pendulum
from colored import attr as c_attr, fg, stylize
from sims_toolkit.gadget.snapshot import Block, File, FileFormat, Header
from tabulate import tabulate

T_BlockDataAttrs = t.Dict[str, np.ndarray]

FG_CYAN = fg("cyan_2")
FG_DEEP_SKY_BLUE_1 = fg("deep_sky_blue_1")
FG_RED = fg("red")
FG_ORANGE = fg("orange_1")
BOLD_TXT = c_attr("bold")


def info_label(text: str):
    """Stylize an information label."""
    return stylize(text, BOLD_TXT)


def describe_snapshot(snap: File):
    """Show the basic info of a GADGET-2 snapshot.

    :param snap: The File instance that represents the snapshot.
    :return: The information as a string.
    """
    header = snap.header
    file_size = snap.size / 1024 ** 2
    if snap.format is FileFormat.ALT:
        file_format = "Enhanced (equivalent to SnapFormat=2)"
    else:
        file_format = "Default"
    num_par_spec_dict = attr.asdict(header.num_pars)
    mass_spec_dict = attr.asdict(header.par_masses)
    num_par_total_dict = attr.asdict(header.total_num_pars)
    par_types = [par_type.capitalize() for par_type in num_par_spec_dict]
    par_nums = [int(num_par) for num_par in num_par_spec_dict.values()]
    par_masses = [mass for mass in mass_spec_dict.values()]
    total_par_nums = [int(num_par) for num_par in num_par_total_dict.values()]
    table_headers = ["Particle Type", "Number", "Mass", "Total Number"]
    table_value_formats = ["s", "d", ".3G", "d"]
    table_values = list(zip(par_types, par_nums, par_masses, total_par_nums))
    par_spec_table = tabulate(table_values,
                              headers=table_headers,
                              floatfmt=table_value_formats)
    blocks = [block.id or "NOT IDENTIFIED" for block in snap.inspect()]
    snap_blocks = ", ".join(blocks)
    reachable_blocks = ", ".join(snap.keys())
    blocks_info_list = [describe_block(block) for block in snap.values()]
    blocks_info_str = "\n".join(blocks_info_list)
    stored_blocks_str = stylize(snap_blocks, FG_ORANGE + BOLD_TXT)
    reachable_blocks_str = stylize(reachable_blocks, FG_ORANGE + BOLD_TXT)
    # Here we construct out information string.
    snap_info_tpl = f"""
{stylize("==========================", FG_RED + BOLD_TXT)}
{stylize("   SNAPSHOT INFORMATION   ", FG_RED + BOLD_TXT)}
{stylize("==========================", FG_RED + BOLD_TXT)}

{stylize("File Information", FG_CYAN + BOLD_TXT)}
{stylize("++++++++++++++++", FG_CYAN + BOLD_TXT)}

Path:       {snap.name}
Format:     {file_format}
Size:       {file_size:.6F}MB

{stylize("Simulation Information", FG_CYAN + BOLD_TXT)}
{stylize("++++++++++++++++++++++", FG_CYAN + BOLD_TXT)}

{info_label("Time")}                {header.time:.5G}
{info_label("Redshift")}            {header.redshift:.5G}
{info_label("Flag Sfr")}            {header.flag_sfr}
{info_label("Flag Feedback")}       {header.flag_feedback}
{info_label("Flag Cooling")}        {header.flag_cooling}
{info_label("Number of Files")}     {header.num_files_snap}
{info_label("Box Size")}            {header.box_size:.5E}
{info_label("Omega0")}              {header.omega_zero:.5G}
{info_label("OmegaLambda")}         {header.omega_lambda:.5G}
{info_label("Hubble Param")}        {header.hubble_param:.5G}

{par_spec_table}

{info_label("Stored Snapshot Blocks")}: {stored_blocks_str}
{info_label("Reachable Snapshot Blocks (Exc. header)")}: {reachable_blocks_str}


{stylize("Blocks Information", FG_CYAN + BOLD_TXT)}
{stylize("++++++++++++++++++", FG_CYAN + BOLD_TXT)}
{blocks_info_str}
    """
    return snap_info_tpl


def describe_block(block: Block):
    """Give an overview of the contents of a snapshot data block.

    :param block: A Block instance.
    :return: The block contents as a string.
    """
    par_types = []
    par_data_str_list = []
    par_data_shape_list = []
    for par_type in block:
        data = block[par_type].data
        data_str = repr(data) if data is not None else "No data"
        data_shape_str = data.shape if data is not None else "No shape"
        par_types.append(par_type)
        par_data_str_list.append(data_str)
        par_data_shape_list.append(data_shape_str)
    table_headers = ["Particle Type", "Data", "Data Shape"]
    table_values = list(zip(par_types, par_data_str_list, par_data_shape_list))
    par_data_table = tabulate(table_values, headers=table_headers)
    block_info_str = f"""
---------------------------
    Block ID:   {block.id}
---------------------------

{par_data_table}
    """
    return block_info_str


@click.group()
def main():
    """Sims-Toolkit Command Line Interface."""
    pass


@main.group()
def gadget_snap():
    """Command group for GADGET-2 snapshot operations."""
    pass


@gadget_snap.command()
@click.argument("path", type=click.Path(exists=True))
def describe(path: str):
    """Describe the contents of a GADGET-2 snapshot."""
    snap = File(pathlib.Path(path))
    description = describe_snapshot(snap)
    click.echo_via_pager(description)


def merge_headers(header: Header, other_header: Header):
    """Combine the headers of two snapshots that will be merged.

    The snapshots should belong to a single simulation. Otherwise,
    the routine could break, or data consistency is not guaranteed.

    :param header: The first header to combine.
    :param other_header: The second header to combine
    :return: The combined header.
    """
    # Consistency check.
    data = header.as_data()
    data["Npart"] = data["Nall"]
    other_data = other_header.as_data()
    other_data["Npart"] = other_data["Nall"]
    assert Header.from_data(data) == Header.from_data(other_data)
    # Merge data of both snapshots.
    new_data = header.as_data()
    new_data["Npart"] += other_header.as_data()["Npart"]
    # Return updated header.
    return Header.from_data(new_data)


def merge_block_set(block_set: t.Iterable[Block], header: Header):
    """Merge a block set from a related collection of snapshots.

    The snapshots should belong to a single simulation. Otherwise,
    the routine could break, or data consistency is not guaranteed.

    :param block_set: The first block data to combine.
    :param header: The header of the final snapshot.
    :return: The combined block data.
    """

    def is_none(_obj: t.Any):
        """Return True if an object is None"""
        return _obj is None

    data_buffer = {par_type: None for par_type in attr.asdict(header.par_specs)}
    for par_type in data_buffer.keys():
        par_data_gen = (block[par_type].data for block in block_set)
        par_data_list = list(filterfalse(is_none, par_data_gen))
        par_data = np.concatenate(par_data_list) if par_data_list else None
        data_buffer[par_type] = par_data
    return data_buffer


file_format_type = click.Choice(["ALT", "DEFAULT"], case_sensitive=False)
blocks_help = "A (comma-separated) list of the data blocks ids that will be " \
              "merged. For instance, --blocks=POS,VEL,ID merges the " \
              "positions, velocities, and particles ids, respectively. By " \
              "default, only the positions get merged."
file_format_help = "The file format of the resulting snapshot. The " \
                   "enhanced format ALT is equivalent to SnapFormat=2."
dry_run_help = "Output the operations but do not execute anything"
merge_set_text = stylize("Executing snapshot merging tool",
                         FG_ORANGE + BOLD_TXT)


@gadget_snap.command()
@click.argument("base-path", type=click.Path(exists=True))
@click.option("--blocks", type=str, default="POS", help=blocks_help)
@click.option("-f", "--file-format", type=file_format_type, default="ALT",
              help=file_format_help)
@click.option("--dry-run", is_flag=True, default=False, help=dry_run_help)
def merge_set(base_path: str, blocks: str, file_format: str, dry_run: bool):
    """Merge a set of related GADGET-2 snapshots."""
    # Message.
    print(f"{merge_set_text}")
    base_path = pathlib.Path(base_path)
    with File(base_path) as base_snap:
        num_files_snap = base_snap.header.num_files_snap
    # Message.
    base_path_txt = stylize(base_path, FG_DEEP_SKY_BLUE_1 + BOLD_TXT)
    num_snap_txt = stylize(num_files_snap, FG_DEEP_SKY_BLUE_1 + BOLD_TXT)
    print(f"Base path: {base_path_txt}")
    print(f"Number of snapshots in collection: {num_snap_txt}")
    snap_set = []
    if not dry_run:
        for idx in range(num_files_snap):
            path = base_path.with_suffix(f".{idx}")
            if not path.exists():
                raise FileNotFoundError(ENOENT, os.strerror(ENOENT), path)
            snap_set.append(File(path))
    snap_format: FileFormat = FileFormat[file_format.upper()]
    format_suffix = f"fmt-{snap_format.name.upper()}"
    dt_suffix = pendulum.now().format("YYYYMMMDD_hhmmssA_zzZZ")
    merged_name = f"{base_path.stem}_merged_{format_suffix}_{dt_suffix}"
    merged_path = base_path.with_name(merged_name)
    # Message.
    merged_path_txt = stylize(merged_path, FG_DEEP_SKY_BLUE_1 + BOLD_TXT)
    print(f"Merged snapshot path: {merged_path_txt}")
    if snap_format is FileFormat.DEFAULT:
        snap_format_txt = "DEFAULT (SnapFormat=1)"
    else:
        snap_format_txt = "ALT (SnapFormat=2)"
    snap_format_txt = stylize(snap_format_txt, FG_DEEP_SKY_BLUE_1 + BOLD_TXT)
    print(f"Format of merged snapshot: {snap_format_txt}")
    # Create snapshot to store the merged data.
    print(f"Checking and combining information from snapshot headers")
    if not dry_run:
        new_snap = File(merged_path, "w", format=snap_format)
        headers = (snap.header for snap in snap_set)
        new_snap.header = reduce(merge_headers, headers)
        # Save to file.
        new_snap.flush()
    block_ids = blocks.split(",")
    for block_id in block_ids:
        block_id_txt = stylize(block_id, FG_DEEP_SKY_BLUE_1 + BOLD_TXT)
        print(f"Combining snapshots blocks with id {block_id_txt}")
        if not dry_run:
            block_set = [snap[block_id] for snap in snap_set]
            # noinspection PyUnboundLocalVariable
            merged_block_data = merge_block_set(block_set, new_snap.header)
            new_snap[block_id] = merged_block_data
            new_snap.flush()
    print("Completed")
    print("Closing merged snapshot and collection")
    if not dry_run:
        # Save block contents to file.
        new_snap.close()
        for snap in snap_set:
            snap.close()
    success_txt = stylize("Success", FG_ORANGE + BOLD_TXT)
    print(f"{success_txt}")


if __name__ == '__main__':
    main()
