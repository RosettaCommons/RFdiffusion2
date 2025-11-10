ORI Tokens
==========

ORI tokens define the center of mass for the designed portion of the enzyme scaffold. It allows for control over the active site and transition
state orientation relative to the protein core. In binder design, the ORI token defines the center of mass of the binder relative to the target protein.

.. warning:: 
   
    ORI tokens are only used in RFdiffusion2 and are not compatible with the original RFdiffusion.
    You must specify an ORI token when using RFdiffusion2

How to Add ORI Tokens to Your Input PDB
---------------------------------------
First add a ``HETATM`` to your PDB file:

.. code-block:: none

    HETATM 6700 ORI ORI z 1 9.450 95.149 43.231 0.00 0.00 z ORI

Let's break down what each of these fields mean:

* ``HETATM``: This indicates that the line describes a non-standard atom.
* ``6700``: This is the atom serial number. You can choose any unique number that does not conflict with existing atom numbers in your PDB file.
* ``ORI``: This is the atom name. It should be set to ``ORI`` to indicate that this atom represents the ORI token.
* ``ORI``: This is the residue name. It should also be set to ``ORI``.
* ``z``: This is the chain identifier. You can choose any single character (letter or number) that does not conflict with existing chain IDs in your PDB file.
* ``1``: This is the residue sequence number. You can choose any number that does not conflict with existing residue numbers in your PDB file.
* ``9.450 95.149 43.231``: These are the X, Y, and Z coordinates of the ORI token in Angstroms. Set these to the desired position for the ORI token.
* ``0.00 0.00``: These are the occupancy and temperature factor values. They can be set to ``0.00`` for the ORI token.
* ``z``: This is the element symbol. It can be set to any character, as it is not relevant for the ORI token.
* ``ORI``: This is the charge. It can be set to any value, as it is not relevant for the ORI token.

For an example of this you can take a look at ``rf_diffusion/benchmark/input/mcsa_41/M0584_1ldm.pdb``, which is used in the demo discussed in the README.
The last line of this file is the ORI token. 

.. note::
    You can also add the ORI token using PyMOL by running the following command in the PyMOL command line:

    .. code-block::

        cmd.delete("molecule1");cmd.pseudoatom(object="molecule1", pos=[-1,3,2], elem="ORI", name="ORI", vdw=1.5, hetatm=True, chain='z', segi='z', resn="ORI"); cmd.show("sphere", "molecule1");

Once you have added this line to your PDB file, you can move it using `PyMOL <https://www.pymol.org/>`_ or other protein visualization software.

For PyMOL, you can easily move the ORI token by selecting it (clicking on the sphere that represents the ORI token) and using the `"move" <https://pymolwiki.org/index.php/Move>`_ command or by dragging it with the mouse.
To drag the ``HETATM`` with the mouse, select the atom by clicking on it, then go to A (action) -> drag coordinates: 

.. figure:: /_static/images/drag_action.png
    :width: 90%
    :alt: Selecting the 'drag coordinates' option from the Action menu in PyMOL
   
    Selecting the menu option that will allow you to drag only the ORI token in PyMOL.

If you are using a three-button mouse, you can hold shift and right-click on the ``HETATM`` to drag it in and out of the screen. 
To translate the atom in and out of the screen, hold shift while center-clicking and dragging. 

.. figure:: /_static/images/moving_ori.gif
    :alt: Moving the ORI token in PyMOL by clicking and dragging it with the mouse

    Moving the ORI token in PyMOL by clicking and dragging it with the mouse.


Recommendations for ORI Token Placement
---------------------------------------
ORI tokens specify the center of mass of the designed region of the protein, but how do you determine where that should be? 

One method of doing this is to estimate where the center of mass of the designed region should be relative to your input structure and then generate several ORI tokens (20-30) in a sphere around that area. 
Whether you want to homogeneously or randomly sample different locations within that sphere and how far apart you want the different ORI tokens to be will depend on your specific design problem.
If you have the computational resources, you can generate several designs for each ORI token placement and then filter the results. 


 

