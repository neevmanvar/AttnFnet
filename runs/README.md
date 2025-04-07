# Runs Directory

This directory contains training logs generated during model training. The logs are stored in a format that can be visualized with TensorBoard, providing an easy way to monitor metrics like loss, accuracy, and other training statistics.

## Viewing Logs with TensorBoard

To launch TensorBoard and view these logs in your browser, follow these steps:

1. Open a terminal and navigate to the project's root directory.
2. Run the following command:

   ```bash
   tensorboard --logdir=runs
    ```
3. Once TensorBoard starts, open your browser and go to  `http://localhost:6006/` (or the URL specified in your terminal) to access the training logs.
