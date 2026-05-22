#!/usr/bin/env python3
"""
实验数据补充代码模板
快速添加到现有代码中进行性能测试和消融实验
"""

import time
import numpy as np
import json
from collections import defaultdict

# ============================================
# 1. 性能监控模块（解决21 FPS问题）
# ============================================

class PerformanceMonitor:
    """监控各模块运行时间"""
    def __init__(self):
        self.timings = defaultdict(list)
        self.current_starts = {}
    
    def start(self, component_name):
        """开始计时"""
        self.current_starts[component_name] = time.time()
    
    def end(self, component_name):
        """结束计时"""
        if component_name in self.current_starts:
            elapsed = time.time() - self.current_starts[component_name]
            self.timings[component_name].append(elapsed * 1000)  # 转换为ms
    
    def get_stats(self):
        """获取统计信息"""
        stats = {}
        for name, times in self.timings.items():
            if times:
                stats[name] = {
                    'mean_ms': np.mean(times),
                    'std_ms': np.std(times),
                    'min_ms': np.min(times),
                    'max_ms': np.max(times),
                    'freq_hz': 1000.0 / np.mean(times) if np.mean(times) > 0 else 0
                }
        return stats
    
    def print_report(self):
        """打印性能报告"""
        stats = self.get_stats()
        print("\n" + "="*70)
        print("Performance Report")
        print("="*70)
        print(f"{'Component':<25} {'Mean(ms)':<12} {'Std(ms)':<12} {'Freq(Hz)':<12}")
        print("-"*70)
        
        total_time = 0
        for name, data in stats.items():
            print(f"{name:<25} {data['mean_ms']:>10.2f}  {data['std_ms']:>10.2f}  {data['freq_hz']:>10.2f}")
            if name != 'total':
                total_time += data['mean_ms']
        
        print("-"*70)
        if total_time > 0:
            fps = 1000.0 / total_time
            print(f"{'TOTAL PIPELINE':<25} {total_time:>10.2f}  {'---':>10}  {fps:>10.2f}")
        print("="*70 + "\n")
    
    def save_to_file(self, filename='performance_report.json'):
        """保存到文件"""
        stats = self.get_stats()
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2)
        print(f"Performance report saved to {filename}")

# 使用示例：
"""
# 在主代码中添加：
perf_monitor = PerformanceMonitor()

# 每个模块前后添加：
perf_monitor.start('visual_processing')
# ... your visual processing code ...
perf_monitor.end('visual_processing')

perf_monitor.start('imu_processing')
# ... your IMU processing code ...
perf_monitor.end('imu_processing')

# 运行结束后：
perf_monitor.print_report()
perf_monitor.save_to_file()
"""


# ============================================
# 2. 消融实验模式切换（解决纯Vision/IMU对比问题）
# ============================================

class AblationConfig:
    """消融实验配置"""
    PURE_VISUAL = 'pure_visual'
    PURE_IMU = 'pure_imu'
    KALMAN_FUSION = 'kalman'
    COMPLEMENTARY_FUSION = 'complementary'  # Ours
    
    def __init__(self, mode='complementary'):
        self.mode = mode
        print(f"Ablation Mode: {mode}")
    
    def should_use_visual(self):
        return self.mode in ['pure_visual', 'kalman', 'complementary']
    
    def should_use_imu(self):
        return self.mode in ['pure_imu', 'kalman', 'complementary']
    
    def get_fusion_type(self):
        if self.mode == 'kalman':
            return 'kalman'
        elif self.mode == 'complementary':
            return 'complementary'
        else:
            return 'none'

# 使用示例：
"""
# 在主代码开头：
ablation_config = AblationConfig(mode='complementary')  # 或 'pure_visual', 'pure_imu'

# 在融合代码中：
if ablation_config.should_use_visual():
    visual_velocity = extract_visual_velocity(image)
else:
    visual_velocity = np.zeros(3)

if ablation_config.should_use_imu():
    imu_velocity = integrate_imu(imu_data)
else:
    imu_velocity = np.zeros(3)

# 融合
if ablation_config.get_fusion_type() == 'complementary':
    fused_velocity = complementary_filter(visual_velocity, imu_velocity)
elif ablation_config.get_fusion_type() == 'kalman':
    fused_velocity = kalman_filter(visual_velocity, imu_velocity)
else:
    fused_velocity = visual_velocity + imu_velocity
"""


# ============================================
# 3. 模板统计收集器（解决64×验证问题）
# ============================================

class TemplateStatistics:
    """收集视觉模板统计数据"""
    def __init__(self):
        self.templates = []
        self.distances = []
        self.frame_count = 0
    
    def add_template(self, template_vector):
        """添加新模板"""
        self.templates.append(template_vector)
    
    def compute_distances(self):
        """计算所有模板对之间的距离（向量化加速）"""
        self.distances = []
        n = len(self.templates)
        if n < 2:
            return
        # 向量化：批量计算所有模板对之间的余弦距离
        mat = np.array(self.templates)
        norms = np.linalg.norm(mat, axis=1, keepdims=True) + 1e-12
        mat_norm = mat / norms
        # 余弦相似度矩阵，取上三角
        cos_sim = mat_norm @ mat_norm.T
        triu_idx = np.triu_indices(n, k=1)
        self.distances = (1.0 - cos_sim[triu_idx]).tolist()
    
    def get_statistics(self):
        """获取统计信息"""
        if not self.distances:
            self.compute_distances()
        
        return {
            'template_count': len(self.templates),
            'distance_mean': np.mean(self.distances) if self.distances else 0,
            'distance_std': np.std(self.distances) if self.distances else 0,
            'distance_min': np.min(self.distances) if self.distances else 0,
            'distance_max': np.max(self.distances) if self.distances else 0,
        }
    
    def print_report(self, dataset_name='Unknown'):
        """打印报告"""
        stats = self.get_statistics()
        print(f"\nTemplate Statistics for {dataset_name}:")
        print(f"  Template Count: {stats['template_count']}")
        print(f"  Distance Mean: {stats['distance_mean']:.4f}")
        print(f"  Distance Std: {stats['distance_std']:.4f}")
        print(f"  Distance Range: [{stats['distance_min']:.4f}, {stats['distance_max']:.4f}]")


# ============================================
# 4. 实验结果记录器
# ============================================

class ExperimentRecorder:
    """记录实验结果，便于生成LaTeX表格"""
    def __init__(self):
        self.results = []
    
    def add_result(self, dataset, method, metrics):
        """
        添加实验结果
        metrics: dict with keys like 'drift', 'vt_count', 'fps', etc.
        """
        result = {
            'dataset': dataset,
            'method': method,
            **metrics
        }
        self.results.append(result)
    
    def to_latex_table(self, caption='Experimental Results'):
        """生成LaTeX表格"""
        if not self.results:
            return "No results to display"
        
        # 获取所有metrics
        all_metrics = set()
        for r in self.results:
            all_metrics.update([k for k in r.keys() if k not in ['dataset', 'method']])
        all_metrics = sorted(all_metrics)
        
        # 生成表格
        latex = "\\begin{table}[h]\n"
        latex += "\\centering\n"
        latex += f"\\caption{{{caption}}}\n"
        latex += "\\begin{tabular}{ll" + "c" * len(all_metrics) + "}\n"
        latex += "\\toprule\n"
        latex += "Dataset & Method & " + " & ".join(all_metrics) + " \\\\\n"
        latex += "\\midrule\n"
        
        for result in self.results:
            row = f"{result['dataset']} & {result['method']}"
            for metric in all_metrics:
                value = result.get(metric, '--')
                if isinstance(value, float):
                    row += f" & {value:.2f}"
                else:
                    row += f" & {value}"
            row += " \\\\\n"
            latex += row
        
        latex += "\\bottomrule\n"
        latex += "\\end{tabular}\n"
        latex += "\\end{table}\n"
        
        return latex
    
    def save_results(self, filename='experiment_results.json'):
        """保存到JSON文件"""
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2)
        print(f"Results saved to {filename}")

# 使用示例：
"""
recorder = ExperimentRecorder()

# 运行实验并记录
recorder.add_result('Town01', 'Pure Visual', {
    'drift': 6.32,
    'vt_count': 321,
    'fps': 25,
    'rmse': 168.2
})

recorder.add_result('Town01', 'Ours', {
    'drift': 4.67,
    'vt_count': 321,
    'fps': 21,
    'rmse': 150.3
})

# 生成LaTeX表格
print(recorder.to_latex_table('Ablation Study Results'))
recorder.save_results()
"""


# ============================================
# 5. 快速集成示例
# ============================================

def example_integration():
    """展示如何快速集成到现有代码"""
    
    # 1. 初始化监控器
    perf = PerformanceMonitor()
    template_stats = TemplateStatistics()
    recorder = ExperimentRecorder()
    
    # 2. 模拟一个循环
    for frame_idx in range(100):
        
        # Visual processing
        perf.start('visual_processing')
        # Your visual processing code here
        time.sleep(0.025)  # 模拟25ms处理时间
        perf.end('visual_processing')
        
        # IMU processing
        perf.start('imu_processing')
        # Your IMU processing code here
        time.sleep(0.002)  # 模拟2ms处理时间
        perf.end('imu_processing')
        
        # Fusion
        perf.start('fusion')
        # Your fusion code here
        time.sleep(0.004)  # 模拟4ms处理时间
        perf.end('fusion')
        
        # Grid cell update
        perf.start('grid_cell_update')
        # Your grid cell code here
        time.sleep(0.009)  # 模拟9ms处理时间
        perf.end('grid_cell_update')
        
        # 收集模板
        if frame_idx % 10 == 0:  # 每10帧收集一次
            fake_template = np.random.randn(4096)
            template_stats.add_template(fake_template)
    
    # 3. 打印报告
    perf.print_report()
    template_stats.print_report('Town01')
    
    # 4. 记录结果
    recorder.add_result('Town01', 'Ours', {
        'drift': 4.67,
        'vt_count': template_stats.get_statistics()['template_count'],
        'fps': 20.0
    })
    
    print("\nLaTeX Table:")
    print(recorder.to_latex_table())


if __name__ == '__main__':
    print("Testing experiment templates...")
    example_integration()
