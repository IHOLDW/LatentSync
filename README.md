

Welcome to the repository! This project provides a setup for running inference with support for super-resolution and device selection (CPU or GPU). Below are the instructions to set up and run the inference.

## Setup Instructions

1. **Set up the environment**:
   Run the following command to set up the environment:
   ```bash
   bash setup_env.sh
   
Inference Script Options

The inference.sh script supports the following parameters:

    Super-Resolution Option:

        Usage: Choose between "gfpgan" or "codeformer" for super-resolution.

        Default: None (no super-resolution applied).

    Device Option:

        Usage: Specify the device for inference: "cpu" or "gpu".

        Default: If not specified, the script will attempt to use the GPU if available.

Changelog

    Added Super-Resolution Support:

        Users can now choose between "gfpgan" or "codeformer" for super-resolution.

    Added CPU/GPU Support:

        Users can specify the device ("cpu" or "gpu") for inference.

    Note: CPU inference may take significantly longer compared to GPU inference. Testing on GPU was not possible due to hardware limitations.
