Overview
========

Introduced in [*Atom level enzyme active site scaffolding using RFdiffusion2*](https://www.biorxiv.org/content/10.1101/2025.04.09.648075v1), RFdiffusion2 expands on the enzyme scaffolding capabilities of diffusion-based protein design by giving researchers finer control over enzyme active sites. 
The original [RFdiffusion](https://github.com/RosettaCommons/RFdiffusion) could generate enzyme scaffolds, but the geometry of the active site could only the specified at the residue level - no atomic or rotamer information could be directly provided. 
Although defining hotspot residues provided a way for protein designers to control scaffold-ligand interactions, they offered  limited flexibility for the placement of the catalytic residues in the final design. 

RFdiffusion2 addresses these limitations by: 
- Allowing active sites to be defined at the atomic level, avoiding the need for costly inverse rotamer sampling.
- Supporting scaffolding of disconnected groups of atoms, including residues with unknown sequence indices, to increase design flexibility. 
- Introducing the ORI (origin) token, which specifies the desired center of mass of the scaffold. This feature enables greater control over active-site placement and transition-state orientation relative to the protein core.

To learn how to run RFdiffusion2 using an [Apptainer](https://apptainer.org/) image, see the [READEME](readme_link.html). 

> **NOTE:** The current rendition of RFdiffusion2 makes it particularly useful for enzyme scaffolding, but for many other applications RFdiffusion (the original) will be easier to use and may provide comparable or better results. 
