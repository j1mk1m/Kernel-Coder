Your kernel executed successfully and produced the correct output.
Here is your wall clock time: 131.0 milliseconds.

-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
                                                   Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg     Self CUDA   Self CUDA %    CUDA total  CUDA time avg    # of Calls  
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
matrix_multiplication_kernel(float*, float*, float*,...         0.00%       0.000us         0.00%       0.000us       0.000us     127.888ms        99.79%     127.888ms     127.888ms             1  
                                            aten::fill_         0.02%      24.080us         0.81%       1.051ms       1.051ms     270.145us         0.21%     540.290us     540.290us             1  
                                Activity Buffer Request         0.75%     980.033us         0.75%     980.033us     980.033us     270.145us         0.21%     270.145us     270.145us             1  
void at::native::vectorized_elementwise_kernel<4, at...         0.00%       0.000us         0.00%       0.000us       0.000us     270.145us         0.21%     270.145us     270.145us             1  
                                            aten::zeros         0.35%     448.162us         1.31%       1.700ms       1.700ms       0.000us         0.00%     540.290us     540.290us             1  
                                            aten::empty         0.03%      42.560us         0.03%      42.560us      42.560us       0.000us         0.00%       0.000us       0.000us             1  
                                            aten::zero_         0.12%     157.650us         0.93%       1.209ms       1.209ms       0.000us         0.00%     540.290us     540.290us             1  
                                       cudaLaunchKernel         0.04%      55.181us         0.04%      55.181us      27.591us       0.000us         0.00%       0.000us       0.000us             2  
                                  cudaDeviceSynchronize        98.68%     128.103ms        98.68%     128.103ms      64.052ms       0.000us         0.00%       0.000us       0.000us             2  
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
Self CPU time total: 129.811ms
Self CUDA time total: 128.159ms