import pickle
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
    print(df)
    return df


def filter_reactions(reactions: list[dict], products: list[str]) -> dict[dict]:
    """
    Filter out any reaction not specified in the products lists
    """
    filtered_reactions = {}
    for reaction in reactions:
        for key in reaction["reactionSystems"]:
            if key in products:
                filtered_reactions[key] = reaction
    return filtered_reactions


def main():
    products = [
        "Ostar",
        "Cstar",
        "Hstar",
        "CH3star",
        "Nstar",
        "CH2star",
        "CHstar",
        "NHstar",
        "OHstar",
        "H2Ostar",
        "SHstar",
    ]

    pymatgen_element_properties = [
        # "symbol",
        # "Z",
        # "atomic_radius_calculated",
        # "van_der_waals_radius",
        # "electrical_resistivity",
        # "velocity_of_sound",
        # "reflectivity",
        # "refractive_index",
        # "poissons_ratio",
        # "molar_volume",
        "thermal_conductivity",
        # "boiling_point",
        # "melting_point",
        # "critical_temperature",
        # "superconduction_temperature",
        # "bulk_modulus",
        # "youngs_modulus",
        # "brinell_hardness",
        # "rigidity_modulus",
        # "mineral_hardness",
        # "vickers_hardness",
        # "density_of_solid",
        # "coefficient_of_linear_thermal_expansion",
        # "ground_level",
        # "X"
        # "atomic_mass",
        # "atomic_radius",
        # "average_anionic_radius",
        "average_cationic_radius",
        # "average_ionic_radius",
        # "electron_affinity",
        "group",
        # "ionization_energy",
        # "is_actinoid",
        # "is_alkali",
        # "is_alkaline",
        # "is_chalcogen",
        # "is_halogen",
        # "is_lanthanoid",
        # "is_metal",
        # "is_metalloid",
        # "is_noble_gas",
        # "is_post_transition_metal",
        # "is_quadrupolar",
        # "is_rare_earth_metal",
        # "is_transition_metal",
        # "max_oxidation_state",
        # "metallic_radius",
        "min_oxidation_state",
        # "row",
    ]

    with open("data/temp.pickle", "rb") as f:
        smol_reactions = pickle.load(f)

    filtered_reactions = filter_reactions(reactions=smol_reactions, products=products)
    systems = [
        reaction["reactionSystems"][key] for key, reaction in filtered_reactions.items()
    ]
    energies = [
        reaction["reactionEnergy"] for _, reaction in filtered_reactions.items()
    ]
    element_symbols = get_unique_element_symbols(systems=systems)
    element_properties = get_element_properties(
        symbols=element_symbols, properties=pymatgen_element_properties
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
    pl.concat(dfs).write_parquet("data/dataset.parquet")

    dataset = pl.read_parquet(source="data/dataset.parquet")
    print(dataset)

    # y = data.drop_in_place("electron_volts")[0]
    # print(y)
    # break


if __name__ == "__main__":
    main()
