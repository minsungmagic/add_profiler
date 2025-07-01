import torch
from torch.profiler import profile, record_function, ProfilerActivity

with torch.profiler.profile(
    activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
    record_shapes=True,
    profile_memory=True,
) as prof:
    x = torch.randn(1000, 1000).cuda()
    y = torch.matmul(x, x)
    prof.step()

print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
