"""Generate SANTA-SIM configs from the template and run them."""
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET

ET.register_namespace("xsi", "http://www.w3.org/2001/XMLSchema-instance")

ROOT = Path(__file__).resolve().parents[2]
TEMPLATE = ROOT / "configs" / "template.xml"
JAR = ROOT / "santa-sim" / "santa.jar"


def make_config(out_path, prefix, mutation_rate, p_dual=None, p_rec=None,
                replicates=10, generations=2000, population=1000,
                sample_size=100):
    """Write one SANTA-SIM config. p_dual=None means no recombination."""
    tree = ET.parse(TEMPLATE)
    root = tree.getroot()
    sim = root.find("simulation")

    root.find("replicates").text = str(replicates)
    sim.find("population/populationSize").text = str(population)
    sim.find("mutator/nucleotideMutator/mutationRate").text = f"{mutation_rate:.4E}"
    sim.find("epoch/generationCount").text = str(generations)

    rep = sim.find("replicator")
    for child in list(rep):
        rep.remove(child)
    if p_dual is None:
        ET.SubElement(rep, "clonalReplicator")
    else:
        rr = ET.SubElement(rep, "recombinantReplicator")
        ET.SubElement(rr, "dualInfectionProbability").text = str(p_dual)
        ET.SubElement(rr, "recombinationProbability").text = str(p_rec)

    samplers = sim.findall("samplingSchedule/sampler")
    samplers[0].find("atGeneration").text = str(generations)
    samplers[0].find("fileName").text = f"{prefix}_%r.fasta"
    samplers[0].find("alignment/sampleSize").text = str(sample_size)
    samplers[1].find("fileName").text = f"{prefix}_%r_stats.csv"

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tree.write(out_path, encoding="utf-8", xml_declaration=True)
    return out_path


def run_santa(config_path, workdir):
    """Run SANTA-SIM. Output files land in workdir."""
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        ["java", "-jar", str(JAR), str(Path(config_path).resolve())],
        cwd=workdir, capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"SANTA-SIM failed:\n{result.stderr[-2000:]}")
    return result.stdout