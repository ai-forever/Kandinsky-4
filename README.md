# Kandinsky-4: Text-to-Video diffusion model

There is one main pipeline now:
* Text-to-video: use **get_T2V_pipeline** function to get it

Examples of usage and more detailed parameters description are in the [examples.ipynb](examples.ipynb) notebook.

Make sure that you have weights folder with weights of all models.

We also add distributed inference opportunity: [run_inference_distil.py](run_inference_distil.py)

To run this examples:
```
python -m torch.distributed.launch --nnodes n --nproc-per-node m run_inference_distil.py
```
where n is a number of nodes you have and m is a number of gpus on these nodes. The code was tested with n=1 and m=8, so this is preferable parameters.

In distributed setting the DiT models are parallelized using tensor parallel on all gpus, which enables a significant speedup.

To run this examples from terminal without tensor parallel:
```
python run_inference_distil.py
```
