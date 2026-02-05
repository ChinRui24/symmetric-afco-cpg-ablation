"""
消融实验可视化 - SCI论文级别
Publication-Quality Figures for Ablation Study

生成适合SCI论文的高质量图表：
1. 性能雷达图（Radar Chart）
2. 消融对比柱状图（Bar Charts）
3. 时域演化曲线（Time-domain Evolution）
4. 扰动恢复对比（Disturbance Recovery）
5. 相位同步可视化（Phase Synchronization）
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
import pandas as pd
from pathlib import Path
from matplotlib import patches
from matplotlib.gridspec import GridSpec
import json

# 设置论文风格
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.titlesize': 13,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
})

sns.set_style("whitegrid")
sns.set_palette("Set2")


class AblationVisualizer:
    """消融实验可视化器"""
    
    def __init__(self, data_path: str):
        """
        初始化
        
        Args:
            data_path: 数据文件路径（CSV或JSON）
        """
        self.data_path = Path(data_path)
        self.output_dir = self.data_path.parent / "figures"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 加载数据
        if self.data_path.suffix == '.csv':
            self.df = pd.read_csv(self.data_path)
        elif self.data_path.suffix == '.json':
            with open(self.data_path) as f:
                data = json.load(f)
            self.df = pd.DataFrame(data)
        
        print(f"✅ 已加载数据: {len(self.df)} 条配置")
        print(f"   列名: {list(self.df.columns)}")
        
        # 配置顺序（用于图表）
        self.config_order = [
            "Full Model",
            "w/o Symmetric PRC",
            "w/o GRF Weighting",
            "w/o Adaptive Coupling",
            "w/o Frequency Adapt",
            "w/o Shock Suppress",
            "Minimal Model",
        ]
        
        # 颜色方案
        self.colors = {
            "Full Model": "#2ecc71",           # 绿色（最好）
            "w/o Symmetric PRC": "#e74c3c",    # 红色（关键创新）
            "w/o GRF Weighting": "#f39c12",    # 橙色
            "w/o Adaptive Coupling": "#3498db", # 蓝色
            "w/o Frequency Adapt": "#9b59b6",   # 紫色
            "w/o Shock Suppress": "#1abc9c",    # 青色
            "Minimal Model": "#95a5a6",         # 灰色（最差）
        }
    
    def plot_radar_chart(self):
        """
        图1: 性能雷达图
        
        展示各配置在多个维度上的综合性能
        """
        print("\n生成雷达图...")
        
        # 选择关键指标（归一化到[0,1]，值越大越好）
        metrics = {
            'Phase Sync': 'phase_synchronization',      # 反向（越小越好）
            'Convergence': 'phase_convergence_time',    # 反向
            'Stability': 'phase_stability',             # 正向
            'Regularity': 'gait_regularity',            # 正向
            'Recovery': 'disturbance_recovery_time',    # 反向
            'Robustness': 'disturbance_deviation',      # 反向
        }
        
        # 准备数据
        configs_to_plot = ["Full Model", "w/o Symmetric PRC", "w/o Adaptive Coupling", "Minimal Model"]
        
        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))
        
        angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
        angles += angles[:1]  # 闭合
        
        for config_name in configs_to_plot:
            config_data = self.df[self.df['config'] == config_name].iloc[0]
            
            values = []
            for metric_key, metric_col in metrics.items():
                val = config_data[metric_col]
                
                # 归一化（越大越好）
                if metric_col in ['phase_synchronization', 'phase_convergence_time', 
                                  'disturbance_recovery_time', 'disturbance_deviation']:
                    # 反向指标：用倒数归一化
                    val_norm = 1.0 / (1.0 + val)
                else:
                    # 正向指标：直接归一化
                    val_norm = val
                
                values.append(val_norm)
            
            values += values[:1]  # 闭合
            
            ax.plot(angles, values, 'o-', linewidth=2, label=config_name, 
                   color=self.colors[config_name])
            ax.fill(angles, values, alpha=0.15, color=self.colors[config_name])
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(list(metrics.keys()), size=10)
        ax.set_ylim(0, 1)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], size=8)
        ax.grid(True, linestyle='--', alpha=0.7)
        
        plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=9)
        plt.title('Performance Comparison: Radar Chart', pad=20, fontweight='bold')
        
        output_path = self.output_dir / "fig1_radar_chart.png"
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
        plt.close()
        
        print(f"  ✅ 保存: {output_path}")
    
    def plot_bar_comparison(self):
        """
        图2: 关键指标对比柱状图
        
        清晰展示各配置在关键指标上的差异
        """
        print("\n生成柱状图...")
        
        # 关键指标
        metrics = {
            'Phase Synchronization\n(rad, lower is better)': 'phase_synchronization',
            'Convergence Time\n(s, lower is better)': 'phase_convergence_time',
            'Recovery Time\n(s, lower is better)': 'disturbance_recovery_time',
            'Gait Regularity\n(higher is better)': 'gait_regularity',
        }
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        axes = axes.flatten()
        
        for idx, (title, metric_col) in enumerate(metrics.items()):
            ax = axes[idx]
            
            # 按配置顺序排列
            data_plot = []
            colors_plot = []
            for config in self.config_order:
                if config in self.df['config'].values:
                    val = self.df[self.df['config'] == config].iloc[0][metric_col]
                    data_plot.append(val)
                    colors_plot.append(self.colors[config])
            
            x_pos = np.arange(len(data_plot))
            bars = ax.bar(x_pos, data_plot, color=colors_plot, alpha=0.8, edgecolor='black', linewidth=0.5)
            
            # 标注数值
            for i, (bar, val) in enumerate(zip(bars, data_plot)):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{val:.3f}', ha='center', va='bottom', fontsize=7)
            
            ax.set_xticks(x_pos)
            ax.set_xticklabels([c.replace(' ', '\n') for c in self.config_order[:len(data_plot)]], 
                              rotation=45, ha='right', fontsize=8)
            ax.set_ylabel('Value', fontsize=9)
            ax.set_title(title, fontsize=10, fontweight='bold')
            ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        output_path = self.output_dir / "fig2_bar_comparison.png"
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
        plt.close()
        
        print(f"  ✅ 保存: {output_path}")
    
    def plot_component_contribution(self):
        """
        图3: 组件贡献分析
        
        展示每个组件对性能提升的贡献
        """
        print("\n生成组件贡献图...")
        
        # 计算相对于Minimal Model的提升
        baseline = self.df[self.df['config'] == 'Minimal Model'].iloc[0]
        full_model = self.df[self.df['config'] == 'Full Model'].iloc[0]
        
        components = {
            'Symmetric PRC': 'w/o Symmetric PRC',
            'GRF Weighting': 'w/o GRF Weighting',
            'Adaptive Coupling': 'w/o Adaptive Coupling',
            'Frequency Adapt': 'w/o Frequency Adapt',
            'Shock Suppress': 'w/o Shock Suppress',
        }
        
        # 选择代表性指标
        metric = 'phase_synchronization'
        
        improvements = []
        component_names = []
        
        for comp_name, config_name in components.items():
            if config_name in self.df['config'].values:
                without_val = self.df[self.df['config'] == config_name].iloc[0][metric]
                full_val = full_model[metric]
                
                # 改进率（对于反向指标）
                improvement = (without_val - full_val) / without_val * 100
                
                improvements.append(improvement)
                component_names.append(comp_name)
        
        # 绘图
        fig, ax = plt.subplots(figsize=(10, 6))
        
        colors_comp = ['#e74c3c', '#f39c12', '#3498db', '#9b59b6', '#1abc9c']
        bars = ax.barh(component_names, improvements, color=colors_comp, alpha=0.8, edgecolor='black')
        
        # 标注
        for bar, val in zip(bars, improvements):
            width = bar.get_width()
            ax.text(width + 1, bar.get_y() + bar.get_height()/2,
                   f'{val:.1f}%', ha='left', va='center', fontsize=9, fontweight='bold')
        
        ax.set_xlabel('Improvement in Phase Synchronization (%)', fontsize=11)
        ax.set_title('Component Contribution Analysis\n(Improvement when component is enabled)', 
                    fontsize=12, fontweight='bold', pad=15)
        ax.grid(axis='x', alpha=0.3)
        ax.axvline(0, color='black', linewidth=0.8)
        
        plt.tight_layout()
        
        output_path = self.output_dir / "fig3_component_contribution.png"
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
        plt.close()
        
        print(f"  ✅ 保存: {output_path}")
    
    def plot_performance_summary(self):
        """
        图4: 性能汇总热图
        
        展示所有配置在所有指标上的表现（颜色编码）
        """
        print("\n生成性能热图...")
        
        # 选择所有数值指标
        metric_cols = [col for col in self.df.columns if col != 'config' 
                      and self.df[col].dtype in [np.float64, np.int64]]
        
        # 选择关键指标
        key_metrics = [
            'phase_synchronization',
            'phase_convergence_time',
            'phase_stability',
            'gait_regularity',
            'disturbance_recovery_time',
            'disturbance_deviation',
            'body_oscillation',
        ]
        
        # 准备数据矩阵
        configs = self.config_order
        data_matrix = []
        
        for config in configs:
            if config in self.df['config'].values:
                row = []
                for metric in key_metrics:
                    val = self.df[self.df['config'] == config].iloc[0][metric]
                    row.append(val)
                data_matrix.append(row)
        
        data_matrix = np.array(data_matrix)
        
        # 归一化每列（使用最小-最大归一化）
        data_normalized = np.zeros_like(data_matrix)
        for j in range(data_matrix.shape[1]):
            col = data_matrix[:, j]
            col_min, col_max = col.min(), col.max()
            if col_max > col_min:
                # 反向指标（越小越好）需要反转
                if key_metrics[j] in ['phase_synchronization', 'phase_convergence_time', 
                                     'disturbance_recovery_time', 'disturbance_deviation', 'body_oscillation']:
                    data_normalized[:, j] = 1 - (col - col_min) / (col_max - col_min)
                else:
                    data_normalized[:, j] = (col - col_min) / (col_max - col_min)
        
        # 绘制热图
        fig, ax = plt.subplots(figsize=(10, 6))
        
        im = ax.imshow(data_normalized, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
        
        # 设置刻度
        ax.set_xticks(np.arange(len(key_metrics)))
        ax.set_yticks(np.arange(len(configs)))
        ax.set_xticklabels([m.replace('_', ' ').title() for m in key_metrics], 
                          rotation=45, ha='right', fontsize=9)
        ax.set_yticklabels(configs, fontsize=9)
        
        # 添加数值标注
        for i in range(len(configs)):
            for j in range(len(key_metrics)):
                text = ax.text(j, i, f'{data_normalized[i, j]:.2f}',
                             ha="center", va="center", color="black", fontsize=7)
        
        ax.set_title('Performance Heatmap\n(Normalized scores, green=better)', 
                    fontsize=12, fontweight='bold', pad=15)
        
        # 颜色条
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Normalized Score', rotation=270, labelpad=20)
        
        plt.tight_layout()
        
        output_path = self.output_dir / "fig4_performance_heatmap.png"
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
        plt.close()
        
        print(f"  ✅ 保存: {output_path}")
    
    def plot_robustness_analysis(self):
        """
        图5: 鲁棒性分析
        
        专门对比扰动恢复性能
        """
        print("\n生成鲁棒性分析图...")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # 选择配置
        configs_to_plot = ["Full Model", "w/o Symmetric PRC", "w/o GRF Weighting", "Minimal Model"]
        
        # 图1: 恢复时间
        recovery_times = []
        colors_plot = []
        for config in configs_to_plot:
            if config in self.df['config'].values:
                val = self.df[self.df['config'] == config].iloc[0]['disturbance_recovery_time']
                recovery_times.append(val)
                colors_plot.append(self.colors[config])
        
        x_pos = np.arange(len(configs_to_plot))
        bars1 = ax1.bar(x_pos, recovery_times, color=colors_plot, alpha=0.8, edgecolor='black')
        
        for bar, val in zip(bars1, recovery_times):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{val:.2f}s', ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels([c.replace(' ', '\n') for c in configs_to_plot], fontsize=9)
        ax1.set_ylabel('Recovery Time (s)', fontsize=10)
        ax1.set_title('Disturbance Recovery Time\n(lower is better)', fontsize=11, fontweight='bold')
        ax1.grid(axis='y', alpha=0.3)
        
        # 图2: 最大偏差
        deviations = []
        for config in configs_to_plot:
            if config in self.df['config'].values:
                val = self.df[self.df['config'] == config].iloc[0]['disturbance_deviation']
                deviations.append(val)
        
        bars2 = ax2.bar(x_pos, deviations, color=colors_plot, alpha=0.8, edgecolor='black')
        
        for bar, val in zip(bars2, deviations):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels([c.replace(' ', '\n') for c in configs_to_plot], fontsize=9)
        ax2.set_ylabel('Maximum Deviation (rad)', fontsize=10)
        ax2.set_title('Maximum Phase Deviation\n(lower is better)', fontsize=11, fontweight='bold')
        ax2.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        output_path = self.output_dir / "fig5_robustness_analysis.png"
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
        plt.close()
        
        print(f"  ✅ 保存: {output_path}")
    
    def plot_comprehensive_comparison(self):
        """
        图6: 综合对比图（多子图）
        
        包含：
        - 相位同步
        - 收敛时间
        - 步态规律性
        - 鲁棒性
        """
        print("\n生成综合对比图...")
        
        fig = plt.figure(figsize=(14, 10))
        gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)
        
        configs = ["Full Model", "w/o Symmetric PRC", "w/o Adaptive Coupling", "Minimal Model"]
        
        # 子图1: 相位同步误差
        ax1 = fig.add_subplot(gs[0, 0])
        self._plot_metric_bar(ax1, 'phase_synchronization', configs, 
                             'Phase Sync Error (rad)', inverse=True)
        
        # 子图2: 收敛时间
        ax2 = fig.add_subplot(gs[0, 1])
        self._plot_metric_bar(ax2, 'phase_convergence_time', configs,
                             'Convergence Time (s)', inverse=True)
        
        # 子图3: 步态规律性
        ax3 = fig.add_subplot(gs[0, 2])
        self._plot_metric_bar(ax3, 'gait_regularity', configs,
                             'Gait Regularity', inverse=False)
        
        # 子图4: 相位稳定性
        ax4 = fig.add_subplot(gs[1, 0])
        self._plot_metric_bar(ax4, 'phase_stability', configs,
                             'Phase Stability', inverse=False)
        
        # 子图5: 扰动恢复时间
        ax5 = fig.add_subplot(gs[1, 1])
        self._plot_metric_bar(ax5, 'disturbance_recovery_time', configs,
                             'Recovery Time (s)', inverse=True)
        
        # 子图6: 身体振荡
        ax6 = fig.add_subplot(gs[1, 2])
        self._plot_metric_bar(ax6, 'body_oscillation', configs,
                             'Body Oscillation (rad)', inverse=True)
        
        # 子图7-9: 归一化综合得分对比
        ax7 = fig.add_subplot(gs[2, :])
        self._plot_overall_score(ax7, configs)
        
        fig.suptitle('Comprehensive Ablation Study Results', 
                    fontsize=14, fontweight='bold', y=0.995)
        
        output_path = self.output_dir / "fig6_comprehensive_comparison.png"
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
        plt.close()
        
        print(f"  ✅ 保存: {output_path}")
    
    def _plot_metric_bar(self, ax, metric, configs, ylabel, inverse=False):
        """辅助函数：绘制单个指标柱状图"""
        values = []
        colors_plot = []
        
        for config in configs:
            if config in self.df['config'].values:
                val = self.df[self.df['config'] == config].iloc[0][metric]
                values.append(val)
                colors_plot.append(self.colors[config])
        
        x_pos = np.arange(len(values))
        bars = ax.bar(x_pos, values, color=colors_plot, alpha=0.8, edgecolor='black', linewidth=0.8)
        
        # 标注数值
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{val:.3f}', ha='center', va='bottom', fontsize=7)
        
        ax.set_xticks(x_pos)
        ax.set_xticklabels([c.replace(' ', '\n') for c in configs], fontsize=7)
        ax.set_ylabel(ylabel, fontsize=8)
        ax.grid(axis='y', alpha=0.3)
        
        # 标记最优
        if inverse:
            best_idx = np.argmin(values)
        else:
            best_idx = np.argmax(values)
        bars[best_idx].set_edgecolor('red')
        bars[best_idx].set_linewidth(2)
    
    def _plot_overall_score(self, ax, configs):
        """综合得分对比"""
        # 计算综合得分（归一化多个指标的加权平均）
        metrics_weights = {
            'phase_synchronization': (0.25, True),   # (权重, 是否反向)
            'phase_convergence_time': (0.15, True),
            'gait_regularity': (0.20, False),
            'phase_stability': (0.15, False),
            'disturbance_recovery_time': (0.15, True),
            'body_oscillation': (0.10, True),
        }
        
        overall_scores = []
        
        for config in configs:
            if config in self.df['config'].values:
                config_data = self.df[self.df['config'] == config].iloc[0]
                score = 0.0
                
                for metric, (weight, inverse) in metrics_weights.items():
                    val = config_data[metric]
                    
                    # 归一化
                    all_vals = self.df[metric].values
                    val_min, val_max = all_vals.min(), all_vals.max()
                    
                    if val_max > val_min:
                        if inverse:
                            val_norm = 1 - (val - val_min) / (val_max - val_min)
                        else:
                            val_norm = (val - val_min) / (val_max - val_min)
                    else:
                        val_norm = 1.0
                    
                    score += weight * val_norm
                
                overall_scores.append(score)
        
        x_pos = np.arange(len(configs))
        colors_plot = [self.colors[c] for c in configs]
        bars = ax.bar(x_pos, overall_scores, color=colors_plot, alpha=0.8, 
                     edgecolor='black', linewidth=1.5)
        
        # 标注
        for bar, val in zip(bars, overall_scores):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{val:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        ax.set_xticks(x_pos)
        ax.set_xticklabels(configs, fontsize=10)
        ax.set_ylabel('Overall Normalized Score', fontsize=11, fontweight='bold')
        ax.set_title('Weighted Overall Performance Score', fontsize=11, fontweight='bold')
        ax.set_ylim(0, 1.1)
        ax.grid(axis='y', alpha=0.3)
        
        # 标记最优
        best_idx = np.argmax(overall_scores)
        bars[best_idx].set_edgecolor('red')
        bars[best_idx].set_linewidth(3)
    
    def generate_all_figures(self):
        """生成所有图表"""
        print("\n" + "="*70)
        print(" 生成SCI论文级别图表")
        print("="*70)
        
        self.plot_radar_chart()
        self.plot_bar_comparison()
        self.plot_component_contribution()
        self.plot_performance_summary()
        self.plot_robustness_analysis()
        self.plot_comprehensive_comparison()
        
        print("\n" + "="*70)
        print(f"✅ 所有图表已生成！保存位置: {self.output_dir}")
        print("="*70)
        
        # 生成图表清单
        figure_list = list(self.output_dir.glob("*.png"))
        print(f"\n生成的图表 ({len(figure_list)}):")
        for fig in sorted(figure_list):
            print(f"  - {fig.name}")


# ============ 主执行 ============
if __name__ == '__main__':
    import sys
    
    if len(sys.argv) > 1:
        data_path = sys.argv[1]
    else:
        data_path = "/mnt/user-data/outputs/ablation_results/ablation_results.csv"
    
    print("="*70)
    print(" 消融实验可视化")
    print(" Ablation Study Visualization")
    print("="*70)
    
    visualizer = AblationVisualizer(data_path)
    visualizer.generate_all_figures()
