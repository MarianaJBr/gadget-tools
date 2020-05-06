import pathlib
import typing as t

import attr
import click
import numpy as np
from colored import attr as c_attr, fg, stylize
from sims_toolkit.gadget.snapshot import Block, File, FileFormat
from tabulate import tabulate

T_BlockDataAttrs = t.Dict[str, np.ndarray]

FG_CYAN = fg("cyan_2")
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
    num_par_spec_dict = attr.asdict(header.num_par_spec)
    mass_spec_dict = attr.asdict(header.mass_spec)
    num_par_total_dict = attr.asdict(header.num_par_total)
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
    block_attrs: T_BlockDataAttrs = attr.asdict(block.data)
    par_types = [par_type.capitalize() for par_type in block_attrs.keys()]
    par_data_str_list = []
    par_data_shape_list = []
    for data in block_attrs.values():
        data_str = repr(data) if data is not None else "No data"
        data_shape_str = data.shape if data is not None else "No shape"
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


if __name__ == '__main__':
    main()
