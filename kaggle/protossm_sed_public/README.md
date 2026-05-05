# BirdCLEF 2026 0.943 Better Blend Kaggle Kernel

This folder is a mechanical script export of:

- `reference/birdclef-2026-0-943-better-blend.ipynb`

It is intended for Kaggle-side execution only. Do not run it locally under the current experiment policy.

## Push and run

From an environment with Kaggle CLI credentials:

```bash
kaggle kernels push -p kaggle/protossm_sed_public
```

After the Kaggle version finishes successfully, submit its `submission.csv`:

```bash
kaggle competitions submissions birdclef-2026
```

or use the Kaggle UI/code submission button for the generated private kernel version.

## Expected role

This is the current high-score anchor path. It should be prioritized before more local CNN/backbone experiments because the public notebook family reports around 0.943 public LB, while our current self-trained CNN path has reached 0.801.
