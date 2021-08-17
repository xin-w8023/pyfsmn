# Py-FSMN
FSMN implementation with PyTorch

## Add FSMNKernelParallel version with group convolution to speed up fsmn computation.
```plain
################################################################################
maximum relative error: max(abs((fsmnp_out - fsmn_out)/ fsmnp_out)) = 0.00000056
################################################################################
parallel fsmn kernel time used: 0.66744542
for-loop fsmn kernel time used: 4.57218981
################################################################################
```
