# Py-FSMN
FSMN implementation with PyTorch

## Add FSMNKernelParallel version with group convolution to speed up fsmn computation.
```python
########################################
# diff: sum(fsmnp_out - fsmn_out) = 0.0
########################################
# parallel time used: 0.26619601249694824
# for-loop time used: 6.988285303115845
########################################
```
