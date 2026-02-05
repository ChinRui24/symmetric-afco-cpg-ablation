#!/usr/bin/env python3
"""
快速查看消融实验结果
Quick View of Ablation Study Results
"""

import pandas as pd
import json
from pathlib import Path

def print_header(text):
    """打印标题"""
    print("\n" + "="*70)
    print(f" {text}")
    print("="*70 + "\n")

def main():
    # 加载数据
    data_path = Path("ablation_results.csv")
    if not data_path.exists():
        data_path = Path(__file__).parent / "ablation_results.csv"
    df = pd.read_csv(data_path)
    
    print_header("对称AFCO CPG消融实验结果")
    
    # 1. 配置列表
    print("📋 测试配置:")
    for i, config in enumerate(df['config'].values, 1):
        print(f"  {i}. {config}")
    
    # 2. 关键指标对比
    print_header("🎯 关键指标对比")
    
    metrics = [
        ('phase_synchronization', 'rad', '相位同步误差', True),
        ('phase_convergence_time', 's', '收敛时间', True),
        ('disturbance_recovery_time', 's', '扰动恢复时间', True),
        ('gait_regularity', '', '步态规律性', False),
        ('body_oscillation', 'rad', '身体振荡', True),
    ]
    
    for metric, unit, name_cn, is_inverse in metrics:
        print(f"\n{name_cn} ({metric}):")
        
        # 找最优值
        if is_inverse:
            best_val = df[metric].min()
            best_config = df.loc[df[metric].idxmin(), 'config']
        else:
            best_val = df[metric].max()
            best_config = df.loc[df[metric].idxmax(), 'config']
        
        # 打印所有配置
        for _, row in df.iterrows():
            val = row[metric]
            config = row['config']
            marker = "⭐" if config == best_config else "  "
            
            # 计算相对于Full Model的变化
            full_val = df[df['config'] == 'Full Model'].iloc[0][metric]
            if full_val != 0:
                if is_inverse:
                    change = (val - full_val) / full_val * 100
                    if change > 0:
                        change_str = f"(+{change:.1f}%)"
                    else:
                        change_str = f"({change:.1f}%)"
                else:
                    change = (val - full_val) / full_val * 100
                    if change > 0:
                        change_str = f"(+{change:.1f}%)"
                    else:
                        change_str = f"({change:.1f}%)"
            else:
                change_str = ""
            
            unit_str = f" {unit}" if unit else ""
            print(f"  {marker} {config:25s}: {val:.4f}{unit_str} {change_str}")
    
    # 3. 组件贡献分析
    print_header("🔧 组件贡献分析（相对于Full Model的改进率）")
    
    full_model = df[df['config'] == 'Full Model'].iloc[0]
    
    components = {
        'Symmetric PRC': 'w/o Symmetric PRC',
        'GRF Weighting': 'w/o GRF Weighting',
        'Adaptive Coupling': 'w/o Adaptive Coupling',
        'Frequency Adapt': 'w/o Frequency Adapt',
        'Shock Suppress': 'w/o Shock Suppress',
    }
    
    contributions = []
    
    for comp_name, config_name in components.items():
        if config_name in df['config'].values:
            without_val = df[df['config'] == config_name].iloc[0]['phase_synchronization']
            full_val = full_model['phase_synchronization']
            
            # 改进率（对于反向指标，without > full 意味着性能下降）
            improvement = (without_val - full_val) / without_val * 100
            
            contributions.append((comp_name, improvement))
    
    # 排序
    contributions.sort(key=lambda x: x[1], reverse=True)
    
    for i, (comp, imp) in enumerate(contributions, 1):
        stars = "⭐" * min(5, max(1, int(abs(imp) / 10)))
        sign = "+" if imp > 0 else ""
        print(f"  {i}. {comp:20s}: {sign}{imp:6.1f}% {stars}")
    
    # 4. 统计摘要
    print_header("📊 统计摘要")
    
    print("Full Model vs w/o Symmetric PRC:")
    full = df[df['config'] == 'Full Model'].iloc[0]
    wo_prc = df[df['config'] == 'w/o Symmetric PRC'].iloc[0]
    
    print(f"  相位同步误差: {full['phase_synchronization']:.4f} → {wo_prc['phase_synchronization']:.4f} "
          f"(恶化 {(wo_prc['phase_synchronization']/full['phase_synchronization']-1)*100:.1f}%)")
    print(f"  扰动恢复时间: {full['disturbance_recovery_time']:.2f}s → {wo_prc['disturbance_recovery_time']:.2f}s "
          f"(恶化 {(wo_prc['disturbance_recovery_time']/full['disturbance_recovery_time']-1)*100:.1f}%)")
    
    # 5. 文件清单
    print_header("📁 生成的文件")
    
    base_path = Path(__file__).parent
    
    files = {
        "数据文件": [
            "ablation_results.csv",
            "ablation_results.json",
        ],
        "图表文件": [
            "figures/fig1_radar_chart.png",
            "figures/fig2_bar_comparison.png",
            "figures/fig3_component_contribution.png",
            "figures/fig4_performance_heatmap.png",
            "figures/fig5_robustness_analysis.png",
            "figures/fig6_comprehensive_comparison.png",
        ],
        "文档文件": [
            "README.md",
            "ABLATION_REPORT.md",
            "latex_tables.tex",
        ],
        "代码文件": [
            "ablation_study.py",
            "visualization.py",
        ],
    }
    
    for category, file_list in files.items():
        print(f"\n{category}:")
        for f in file_list:
            fpath = base_path / f
            if fpath.exists():
                size = fpath.stat().st_size
                size_str = f"{size/1024:.1f}KB" if size > 1024 else f"{size}B"
                print(f"  ✅ {f:45s} ({size_str})")
            else:
                print(f"  ❌ {f}")
    
    # 6. 下一步建议
    print_header("💡 下一步建议")
    
    print("""
1. 查看详细报告:
   cat ABLATION_REPORT.md

2. 查看图表:
   open figures/  # 或用系统图片查看器

3. 集成到论文:
   - 复制 latex_tables.tex 中的表格
   - 使用 figures/ 中的图表
   - 参考 README.md 中的写作建议

4. 重新运行实验（如需调整参数）:
   python ablation_study.py

5. 重新生成图表（如需修改样式）:
   python visualization.py ablation_results.csv
    """)
    
    print("="*70)
    print(" ✅ 消融实验完成！所有结果已准备就绪。")
    print("="*70)

if __name__ == '__main__':
    main()
