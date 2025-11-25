Your kernel executed successfully and produced the correct output.
Here is your wall clock time: 27.5 milliseconds.

-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
                                                   Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg     Self CUDA   Self CUDA %    CUDA total  CUDA time avg    # of Calls  
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
              sigmoid_kernel(float const*, float*, int)         0.00%       0.000us         0.00%       0.000us       0.000us      18.662ms        67.43%      18.662ms      18.662ms             1  
                 zeros_kernel(float*, int)         0.00%       0.000us         0.00%       0.000us       0.000us      18.662ms        67.43%      18.662ms      18.662ms             1  
                                            aten::fill_         0.12%      33.910us         4.46%       1.293ms       1.293ms       9.016ms        32.57%      18.032ms      18.032ms             1  
                                Activity Buffer Request         4.15%       1.202ms         4.15%       1.202ms       1.202ms       9.016ms        32.57%       9.016ms       9.016ms             1  
void at::native::vectorized_elementwise_kernel<4, at...         0.00%       0.000us         0.00%       0.000us       0.000us       9.016ms        32.57%       9.016ms       2.254ms             4  
                                       aten::zeros_like         0.05%      13.720us         4.73%       1.371ms       1.371ms       0.000us         0.00%      18.032ms      18.032ms             1  
                                       aten::empty_like         0.05%      13.730us         0.19%      56.010us      56.010us       0.000us         0.00%       0.000us       0.000us             1  
                                    aten::empty_strided         0.15%      42.280us         0.15%      42.280us      42.280us       0.000us         0.00%       0.000us       0.000us             1  
                                            aten::zero_         0.03%       8.740us         4.49%       1.302ms       1.302ms       0.000us         0.00%      18.032ms      18.032ms             1  
                                       cudaLaunchKernel         0.21%      62.240us         0.21%      62.240us      12.448us       0.000us         0.00%       0.000us       0.000us             5  
                                  cudaDeviceSynchronize        95.25%      27.597ms        95.25%      27.597ms      13.799ms       0.000us         0.00%       0.000us       0.000us             2  
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
Self CPU time total: 28.974ms
Self CUDA time total: 27.678ms