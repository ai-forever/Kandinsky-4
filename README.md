# Inference for Kandinsky 4

There are two main pipelines now:
* Text-to-video: use **get_T2V_pipeline** function to get it
* Text+Image-to-video: use **get_IT2V_pipeline** function to get it

Examples of usage and more detailed parameters description are in the [examples.ipynb](examples.ipynb) notebook.

Make sure that you have weights folder with weights of all models.

As the generation of 12s video in 1024 resolution takes 12 minutes on one gpu, we also add distributed inference opportunity. See examples in [run_inference.py](run_inference.py), [run_inference_it2v.py](run_inference_it2v.py) and [run_inference_flux.py](run_inference_flux.py).

To run this examples:
```
python -m torch.distributed.launch --nnodes n --nproc-per-node m run_inference.py
```
where n is a number of nodes you have and m is a number of gpus on these nodes. The code was tested with n=1 and m=8, so this is preferable parameters.

In distributed setting the DiT models are parallelized using tensor parallel on all gpus and generation of 12s video in 1024 resolution takes 4 minutes.

To run this examples from terminal without tensor parallel:
```
python run_inference.py
```
