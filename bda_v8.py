import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import time

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

class BDAConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.theta = nn.Parameter(torch.full((out_channels,), -2.9))
        self.gamma = 0.5
        
        self.register_buffer('mask_cache', None)
        self.cache_hits = 0
        self.total_calls = 0
        self.register_buffer('total_forward', torch.tensor(0, dtype=torch.long))
        self.register_buffer('total_dormant', torch.tensor(0, dtype=torch.long))
    
    def get_threshold(self):
        return torch.sigmoid(self.theta) * self.gamma
    
    def forward(self, x):
        if self.training:
            act = F.relu(self.bn(self.conv(x)))
        else:
            if not hasattr(self, 'conv_fused'):
                mean = self.bn.running_mean
                var = self.bn.running_var
                gamma = self.bn.weight
                beta = self.bn.bias
                eps = self.bn.eps
                
                w = self.conv.weight
                scale = gamma / torch.sqrt(var + eps)
                w_fused = w * scale.view(-1, 1, 1, 1)
                b_fused = beta - gamma * mean / torch.sqrt(var + eps)
                
                self.conv_fused = nn.Conv2d(self.in_channels, self.out_channels, self.kernel_size, self.stride, self.padding, bias=True)
                self.conv_fused.weight.data = w_fused
                self.conv_fused.bias.data = b_fused
                self.conv_fused = self.conv_fused.to(self.conv.weight.device)
            
            act = F.relu(self.conv_fused(x))
        
        theta = self.get_threshold().view(1, -1, 1, 1)
        
        if self.training:
            mask = (act > theta).float()
            mask_ste = mask + (act/(act + 1e-8) - mask).detach()
            
            with torch.no_grad():
                self.total_forward += mask.numel()
                self.total_dormant += (mask == 0).sum().item()
            
            return act * mask_ste
        else:
            self.total_calls += 1
            if self.mask_cache is not None and self.mask_cache.shape == act.shape:
                self.cache_hits += 1
                return act * self.mask_cache
            
            mask = (act > theta).float()
            self.mask_cache = mask.detach().clone()
            
            with torch.no_grad():
                self.total_forward += mask.numel()
                self.total_dormant += (mask == 0).sum().item()
            
            return act * mask
    
    def get_dormancy(self):
        if self.total_forward.item() == 0:
            return 0.0
        return self.total_dormant.float().item() / self.total_forward.float().item()
    
    def get_cache_hit_rate(self):
        if self.total_calls == 0:
            return 0.0
        return self.cache_hits / self.total_calls

class SimpleResNet50(nn.Module):
    def __init__(self, use_bda=True):
        super().__init__()
        
        if use_bda:
            self.conv1 = BDAConv2d(3, 64, 3, 1, 1)
            self.conv2 = BDAConv2d(64, 128, 3, 2, 1)
            self.conv3 = BDAConv2d(128, 256, 3, 2, 1)
        else:
            self.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
            self.conv2 = nn.Conv2d(64, 128, 3, 2, 1, bias=False)
            self.conv3 = nn.Conv2d(128, 256, 3, 2, 1, bias=False)
            self.bn1 = nn.BatchNorm2d(64)
            self.bn2 = nn.BatchNorm2d(128)
            self.bn3 = nn.BatchNorm2d(256)
        
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(256, 100)
        self.use_bda = use_bda
        
        self.bda_layers = []
        if use_bda:
            for module in self.modules():
                if isinstance(module, BDAConv2d):
                    self.bda_layers.append(module)
    
    def forward(self, x):
        if self.use_bda:
            x = self.conv1(x)
            x = self.conv2(x)
            x = self.conv3(x)
        else:
            x = F.relu(self.bn1(self.conv1(x)))
            x = F.relu(self.bn2(self.conv2(x)))
            x = F.relu(self.bn3(self.conv3(x)))
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
    
    def get_stats(self):
        if not self.bda_layers:
            return {'dormancy': 0.0, 'cache_hit': 0.0, 'count': 0}
        dorm = [l.get_dormancy() * 100 for l in self.bda_layers]
        cache = [l.get_cache_hit_rate() * 100 for l in self.bda_layers]
        return {
            'dormancy_mean': float(np.mean(dorm)),
            'dormancy_std': float(np.std(dorm, ddof=1)) if len(dorm) > 1 else 0.0,
            'cache_hit_mean': float(np.mean(cache)),
            'cache_hit_std': float(np.std(cache, ddof=1)) if len(cache) > 1 else 0.0,
            'count': len(dorm)
        }

def measure_time(model, x, iterations=200):
    model.eval()
    for _ in range(50):
        _ = model(x)
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        _ = model(x)
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iterations

def run_benchmark():
    print("="*80)
    print("BDA v8.0 - Final Benchmark")
    print("="*80)
    
    batch_sizes = [1, 8, 32]
    results = {}
    
    for bs in batch_sizes:
        print(f"\nTesting batch_size = {bs}")
        x = torch.randn(bs, 3, 224, 224).cuda()
        x_half = x.half()
        
        std_model = SimpleResNet50(use_bda=False).cuda().eval()
        bda_model = SimpleResNet50(use_bda=True).cuda().eval()
        std_model_half = SimpleResNet50(use_bda=False).cuda().half().eval()
        bda_model_half = SimpleResNet50(use_bda=True).cuda().half().eval()
        
        std_time = measure_time(std_model, x, 200)
        bda_time = measure_time(bda_model, x, 200)
        std_half_time = measure_time(std_model_half, x_half, 200)
        bda_half_time = measure_time(bda_model_half, x_half, 200)
        
        stats = bda_model.get_stats()
        
        results[bs] = {
            'fp32': {
                'standard_ms': float(std_time),
                'bda_ms': float(bda_time),
                'overhead': float((bda_time - std_time) / std_time * 100)
            },
            'fp16': {
                'standard_ms': float(std_half_time),
                'bda_ms': float(bda_half_time),
                'overhead': float((bda_half_time - std_half_time) / std_half_time * 100)
            },
            'dormancy': float(stats['dormancy_mean']),
            'cache_hit': float(stats['cache_hit_mean']),
            'bda_layers': int(stats['count'])
        }
        
        print(f"  FP32 - Standard: {std_time:.3f}ms | BDA: {bda_time:.3f}ms | Δ: {results[bs]['fp32']['overhead']:+.1f}%")
        print(f"  FP16 - Standard: {std_half_time:.3f}ms | BDA: {bda_half_time:.3f}ms | Δ: {results[bs]['fp16']['overhead']:+.1f}%")
        print(f"  Dormancy: {stats['dormancy_mean']:.1f}% | Cache Hit: {stats['cache_hit_mean']:.1f}%")
        
        del std_model, bda_model, std_model_half, bda_model_half
        torch.cuda.empty_cache()
    
    print("\n" + "="*80)
    print("FINAL RESULTS")
    print("="*80)
    print("\nBatch | FP32 Std | FP32 BDA |  Δ%  | FP16 Std | FP16 BDA |  Δ%  | Dorm%")
    print("-"*80)
    
    for bs in batch_sizes:
        r = results[bs]
        print(f"{bs:5d} | {r['fp32']['standard_ms']:8.3f} | {r['fp32']['bda_ms']:8.3f} | {r['fp32']['overhead']:5.1f} | {r['fp16']['standard_ms']:8.3f} | {r['fp16']['bda_ms']:8.3f} | {r['fp16']['overhead']:5.1f} | {r['dormancy']:6.1f}")
    
    with open('bda_final_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n✅ Results saved to bda_final_results.json")
    return results

if __name__ == "__main__":
    print("""
    ╔══════════════════════════════════════════════════════════╗
    ║  BDA v8.0 - Final Version                               ║
    ╚══════════════════════════════════════════════════════════╝
    """)
    
    if torch.cuda.is_available():
        results = run_benchmark()
    else:
        print("CUDA not available")