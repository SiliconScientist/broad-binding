import polars as pl
from pymatgen.core.periodic_table import Element
from ase.atoms import Atoms


def get_unique_element_symbols(systems: list[Atoms]):
    atom_symbols = []
    for atoms in systems:
        symbols = atoms.get_chemical_symbols()
        for symbol in symbols:
            atom_symbols.append(symbol)
    return list(set(atom_symbols))


def get_element_properties(symbols: list[str], properties: list[str]) -> pl.DataFrame:
    element_properties = []
    for symbol in symbols:
        property_row = {}
        element = Element(symbol)
        for property_str in properties:
            element_property = getattr(element, property_str)
            property_row[property_str] = element_property
        element_properties.append(property_row)
    df = pl.DataFrame(element_properties).with_columns(symbol=pl.Series(symbols))
    return df


def process_system(atoms: Atoms, energy: float) -> pl.DataFrame:
    positions = atoms.get_positions()
    symbols = atoms.get_chemical_symbols()
    df = pl.DataFrame(positions, schema=["x", "y", "z"]).with_columns(
        electron_volts=pl.lit(energy), symbol=pl.Series(symbols)
    )
    return df


def filter_reactions(reactions: list[dict], bound_sites: list[str]) -> list[dict]:
    """
    Filter out any reaction not specified in the products lists
    """
    filtered_reactions = []
    for reaction in reactions:
        for key in reaction["reactionSystems"]:
            if key in bound_sites:
                reaction["reactionSystems"] = reaction["reactionSystems"][key]
                filtered_reactions.append(reaction)
    return filtered_reactions


def make_dataframe(
    reactions: list[dict], bound_sites: list[str], properties: list[str]
) -> pl.DataFrame:
    filtered_reactions = filter_reactions(reactions=reactions, bound_sites=bound_sites)
    systems = [reaction["reactionSystems"] for reaction in filtered_reactions]
    energies = [reaction["reactionEnergy"] for reaction in filtered_reactions]
    element_symbols = get_unique_element_symbols(systems=systems)
    element_properties = get_element_properties(
        symbols=element_symbols, properties=properties
    )
    dfs = []
    for i, (atoms, energy) in enumerate(zip(systems, energies)):
        df = (
            process_system(atoms=atoms, energy=energy)
            .join(other=element_properties, on="symbol", how="inner")
            .select(
                pl.col(["symbol", "electron_volts"]),
                pl.exclude(["symbol", "electron_volts"]),
            )
        ).with_columns(index=pl.lit(i))
        dfs.append(df)
    return pl.concat(dfs)
