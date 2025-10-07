Usage Examples
==============

The usage examples linked below demonstrate how to use the various provided scripts to run inference with RFdiffusion2. 

Before using RFdiffusion2 for your own designs make sure you are able to run the demo included in the [README](../readme_link.html#inference).

How to run RFdiffusion2
-----------------------
For those of you who are familiar with running the `original RFdiffusion <https://github.com/RosettaCommons/RFdiffusion>`_, running RFdiffusion2 is very similar. 
The main differences are: 

* The inference script no longer chooses the best model weights for you to use, there is one recommended model weight file located at ``RFdiffusion2/rf_diffusion/model_weights/RFD_173.pt``. This is the set of weights used in the demo in the README.

* RFdiffusion2 can now take atomic inputs, not just backbone-level information

* An ORI token is expected in the input PDB, which specifies the center of mass for the design region. For more information on ORI tokens see 
  the documentation on :doc:`ORI tokens <ori_tokens>`.


More information
^^^^^^^^^^^^^^^^
* :doc:`ORI Tokens <ori_tokens>` - Explanation of ORI tokens and how to use them in your designs.
* :doc:`Configuration Options <configuration_options>` - Explanation of the various configuration options available for RFdiffusion2.

Examples
^^^^^^^^

- :doc:`Using the pipelines.py script <other_pipeline_example>`
- :doc:`Using the run_inference.py script <run_inference_example>`