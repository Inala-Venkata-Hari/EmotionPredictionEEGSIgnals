# Last-block LaBraM adaptation: pilot in progress

The first eleven transformer blocks of the official pretrained encoder remain frozen. This batch adapts only the final block, normalization and a two-class head, using a fixed five-epoch training schedule. The initial learning rates are 1e-4 for the last block and 1e-3 for the head, with weight decay .01, dropout .25, gradient clipping 1 and seed 42. No epoch or learning-rate search is performed.

The same original-training-only five-outer/three-inner grouped protocol compares the fixed pilot with the majority control. One-second and 7.875-second inputs are evaluated separately. No original validation/test groups enter fitting or selection in this batch. The pretrained prefix is label-free, and only the current fitting fold supplies labels to adaptation.

Tests confirm that the prefix followed by the original last block reproduces the official encoder, and that task fitting cannot mutate the preserved pretrained encoder. Numerical source, checkpoint and runtime hashes are recorded in the frozen plan. No passable accuracy is claimed while the pilot is running.

Run `.venv/bin/python -m eegis.train_finetune`, then `.venv/bin/python -m eegis.verify_accuracy --artifacts artifacts_finetune`. The official checkpoint is downloaded with `python scripts/download_labram.py`. Binary checkpoints/caches remain local; measured results will be published after independent verification.
