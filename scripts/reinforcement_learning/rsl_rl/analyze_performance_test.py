#!/usr/bin/env python3
# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Performance test analysis script with advanced visualizations"""

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path

# Terrain type mapping
TERRAIN_MAPPING = {
    0: 'A', 1: 'A',  # pyramid_stairs
    2: 'B', 3: 'B',  # pyramid_stairs_inv
    4: 'C', 5: 'C',  # boxes
    6: 'D', 7: 'D',  # random_rough
    8: 'E',          # hf_pyramid_slope
    9: 'F'           # hf_pyramid_slope_inv
}

TERRAIN_NAMES = {
    'A': 'Pyramid Stairs',
    'B': 'Inv Pyramid Stairs',
    'C': 'Boxes',
    'D': 'Random Rough',
    'E': 'Pyramid Slope',
    'F': 'Inv Pyramid Slope'
}


def analyze_performance_csv(csv_path: str, output_dir: str | None = None):
    """Analyze performance test CSV and generate comprehensive visualizations
    
    Args:
        csv_path: Path to CSV file
        output_dir: Output directory, defaults to CSV file directory
    """
    df = pd.read_csv(csv_path)
    
    if output_dir is None:
        output_dir = os.path.dirname(csv_path)
    
    # Add terrain category mapping
    if 'terrain_type' in df.columns:
        df['terrain_category'] = df['terrain_type'].map(TERRAIN_MAPPING)  # type: ignore
    
    has_terrain_info = 'terrain_level' in df.columns and 'terrain_type' in df.columns
    
    # Generate text report
    report_path = os.path.join(output_dir, "performance_report.txt")
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("Performance Test Analysis Report\n")
        f.write("=" * 80 + "\n\n")
        
        # 基本信息
        f.write(f"Test File: {os.path.basename(csv_path)}\n")
        f.write(f"Total Iterations: {df['iteration'].max()}\n")
        f.write(f"Environments per Iteration: {len(df[df['iteration'] == 1])}\n")
        f.write(f"Total Test Environments: {len(df)}\n")
        
        if has_terrain_info:
            f.write(f"Terrain Levels: {df['terrain_level'].min()} - {df['terrain_level'].max()}\n")
            f.write(f"Terrain Types: {df['terrain_type'].min()} - {df['terrain_type'].max()}\n")
        f.write("\n")
        
        # Fall Statistics
        f.write("-" * 80 + "\n")
        f.write("Fall Statistics\n")
        f.write("-" * 80 + "\n")
        total_falls = df['fell_down'].sum()
        fall_rate = total_falls / len(df) * 100
        f.write(f"Total Falls: {total_falls}\n")
        f.write(f"Fall Rate: {fall_rate:.2f}%\n")
        f.write(f"Success Rate: {100 - fall_rate:.2f}%\n\n")
        
        # Fall statistics by iteration
        f.write("Fall Statistics by Iteration:\n")
        for iteration in sorted(df['iteration'].unique()):
            iter_df = df[df['iteration'] == iteration]
            iter_falls = iter_df['fell_down'].sum()
            iter_rate = iter_falls / len(iter_df) * 100
            f.write(f"  Iteration {iteration}: {iter_falls}/{len(iter_df)} ({iter_rate:.2f}%)\n")
        f.write("\n")
        
        # Terrain-based statistics
        if has_terrain_info:
            f.write("-" * 80 + "\n")
            f.write("Terrain-based Performance Analysis\n")
            f.write("-" * 80 + "\n")
            
            # Fall rate by terrain level
            f.write("Fall Rate by Terrain Level:\n")
            for level in sorted(df['terrain_level'].unique()):
                level_df = df[df['terrain_level'] == level]
                level_falls = level_df['fell_down'].sum()
                level_rate = level_falls / len(level_df) * 100
                f.write(f"  Level {level}: {level_falls}/{len(level_df)} ({level_rate:.2f}%)\n")
            f.write("\n")
            
            # Fall rate by terrain type
            f.write("Fall Rate by Terrain Type:\n")
            for terrain_type in sorted(df['terrain_type'].unique()):
                type_df = df[df['terrain_type'] == terrain_type]
                type_falls = type_df['fell_down'].sum()
                type_rate = type_falls / len(type_df) * 100
                f.write(f"  Type {terrain_type}: {type_falls}/{len(type_df)} ({type_rate:.2f}%)\n")
            f.write("\n")
            
            # Performance by terrain level and type
            f.write("Performance by Terrain Level and Type:\n")
            for level in sorted(df['terrain_level'].unique()):
                for terrain_type in sorted(df['terrain_type'].unique()):
                    subset_df = df[(df['terrain_level'] == level) & (df['terrain_type'] == terrain_type)]
                    if len(subset_df) > 0:
                        subset_falls = subset_df['fell_down'].sum()
                        subset_rate = subset_falls / len(subset_df) * 100
                        avg_lin_error = subset_df['lin_vel_tracking_error'].mean()
                        avg_ang_error = subset_df['ang_vel_tracking_error'].mean()
                        f.write(f"  Level {level}, Type {terrain_type}: {subset_falls}/{len(subset_df)} ({subset_rate:.2f}%) | "
                               f"Lin Error: {avg_lin_error:.4f} | Ang Error: {avg_ang_error:.4f}\n")
            f.write("\n")
        
        # Velocity tracking performance
        f.write("-" * 80 + "\n")
        f.write("Velocity Tracking Performance (All Environments)\n")
        f.write("-" * 80 + "\n")
        f.write(f"Average Linear Velocity Tracking Error: {df['lin_vel_tracking_error'].mean():.4f} ± {df['lin_vel_tracking_error'].std():.4f} m/s\n")
        f.write(f"Average Angular Velocity Tracking Error: {df['ang_vel_tracking_error'].mean():.4f} ± {df['ang_vel_tracking_error'].std():.4f} rad/s\n\n")
        
        f.write(f"Linear Velocity Tracking Error Median: {df['lin_vel_tracking_error'].median():.4f} m/s\n")
        f.write(f"Angular Velocity Tracking Error Median: {df['ang_vel_tracking_error'].median():.4f} rad/s\n\n")
        
        f.write(f"Linear Velocity Tracking Error Max: {df['lin_vel_tracking_error'].max():.4f} m/s\n")
        f.write(f"Linear Velocity Tracking Error Min: {df['lin_vel_tracking_error'].min():.4f} m/s\n\n")
        
        # Analyze only successful environments
        success_df = df[df['fell_down'] == 0]
        if len(success_df) > 0:
            f.write("-" * 80 + "\n")
            f.write("Velocity Tracking Performance (Successful Environments Only)\n")
            f.write("-" * 80 + "\n")
            f.write(f"Average Linear Velocity Tracking Error: {success_df['lin_vel_tracking_error'].mean():.4f} ± {success_df['lin_vel_tracking_error'].std():.4f} m/s\n")
            f.write(f"Average Angular Velocity Tracking Error: {success_df['ang_vel_tracking_error'].mean():.4f} ± {success_df['ang_vel_tracking_error'].std():.4f} rad/s\n\n")
            
            f.write(f"Actual Average Linear Velocity Norm: {success_df['avg_actual_lin_vel_xy_norm'].mean():.4f} ± {success_df['avg_actual_lin_vel_xy_norm'].std():.4f} m/s\n")
            f.write(f"Command Average Linear Velocity Norm: {success_df['avg_command_lin_vel_xy_norm'].mean():.4f} ± {success_df['avg_command_lin_vel_xy_norm'].std():.4f} m/s\n\n")
            
            f.write(f"Actual Average Angular Velocity: {success_df['avg_actual_ang_vel_z'].mean():.4f} ± {success_df['avg_actual_ang_vel_z'].std():.4f} rad/s\n")
            f.write(f"Command Average Angular Velocity: {success_df['avg_command_ang_vel_z'].mean():.4f} ± {success_df['avg_command_ang_vel_z'].std():.4f} rad/s\n\n")
        
        
    print(f"Analysis report saved to: {report_path}")
    
    # Generate visualization charts
    create_visualizations(df, output_dir)


def create_visualizations(df: pd.DataFrame, output_dir: str):
    """Create comprehensive visualization charts
    
    Args:
        df: Performance data DataFrame
        output_dir: Output directory
    """
    import matplotlib
    matplotlib.use('Agg')
    
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['figure.dpi'] = 100
    
    has_terrain_info = 'terrain_level' in df.columns and 'terrain_type' in df.columns
    
    # Core visualizations
    create_command_velocity_analysis(df, output_dir)
    
    if has_terrain_info and 'terrain_category' in df.columns:
        create_terrain_performance_analysis(df, output_dir)
        create_cross_analysis(df, output_dir)


def create_command_velocity_analysis(df: pd.DataFrame, output_dir: str):
    """Analyze performance vs command velocities
    
    Generates:
    - command_velocity_analysis.png: Error bars showing tracking error vs command velocities
    - command_velocity_heatmaps.png: 2D heatmaps of errors and fall rate vs lin/ang velocity
    """
    
    # Compute command velocity norms
    df['command_vel_norm'] = np.sqrt(df['avg_command_lin_vel_xy_norm']**2 + df['avg_command_ang_vel_z']**2)
    
    # Bin command velocities for analysis
    df['lin_vel_bin'] = pd.cut(df['avg_command_lin_vel_xy_norm'], bins=10)
    df['ang_vel_bin'] = pd.cut(df['avg_command_ang_vel_z'], bins=10)
    df['total_vel_bin'] = pd.cut(df['command_vel_norm'], bins=10)
    
    # 1. Tracking error vs command velocity (3 subplots)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Linear velocity tracking
    lin_vel_stats = df.groupby('lin_vel_bin').agg({
        'lin_vel_tracking_error': ['mean', 'std', 'count'],
        'fell_down': 'mean'
    })
    
    x_lin = [interval.mid for interval in lin_vel_stats.index]  # type: ignore
    y_lin_error = lin_vel_stats['lin_vel_tracking_error']['mean'].values  # type: ignore
    y_lin_std = lin_vel_stats['lin_vel_tracking_error']['std'].values  # type: ignore
    
    axes[0].errorbar(x_lin, y_lin_error, yerr=y_lin_std, fmt='o-', capsize=5, 
                     linewidth=2, markersize=6, color='steelblue', label='Lin Vel Error')
    axes[0].set_xlabel('Command Linear Velocity (m/s)', fontsize=11, fontweight='bold')
    axes[0].set_ylabel('Tracking Error (m/s)', fontsize=11, fontweight='bold')
    axes[0].set_title('Linear Velocity Tracking vs Command', fontsize=12, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    
    # Angular velocity tracking
    ang_vel_stats = df.groupby('ang_vel_bin').agg({
        'ang_vel_tracking_error': ['mean', 'std', 'count'],
        'fell_down': 'mean'
    })
    
    x_ang = [interval.mid for interval in ang_vel_stats.index]  # type: ignore
    y_ang_error = ang_vel_stats['ang_vel_tracking_error']['mean'].values  # type: ignore
    y_ang_std = ang_vel_stats['ang_vel_tracking_error']['std'].values  # type: ignore
    
    axes[1].errorbar(x_ang, y_ang_error, yerr=y_ang_std, fmt='o-', capsize=5,
                     linewidth=2, markersize=6, color='forestgreen', label='Ang Vel Error')
    axes[1].set_xlabel('Command Angular Velocity (rad/s)', fontsize=11, fontweight='bold')
    axes[1].set_ylabel('Tracking Error (rad/s)', fontsize=11, fontweight='bold')
    axes[1].set_title('Angular Velocity Tracking vs Command', fontsize=12, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    
    # Fall rate vs total command velocity
    total_vel_stats = df.groupby('total_vel_bin').agg({
        'fell_down': ['mean', 'count']
    })
    
    x_total = [interval.mid for interval in total_vel_stats.index]  # type: ignore
    y_fall_rate = total_vel_stats['fell_down']['mean'].values * 100  # type: ignore
    
    axes[2].plot(x_total, y_fall_rate, 'o-', linewidth=2, markersize=6, color='coral')
    axes[2].set_xlabel('Total Command Velocity Norm', fontsize=11, fontweight='bold')
    axes[2].set_ylabel('Fall Rate (%)', fontsize=11, fontweight='bold')
    axes[2].set_title('Fall Rate vs Command Velocity', fontsize=12, fontweight='bold')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'command_velocity_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Chart saved: command_velocity_analysis.png")
    
    # 2. 2D heatmap: Linear vs Angular velocity with tracking error
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Create 2D bins
    lin_bins = np.linspace(df['avg_command_lin_vel_xy_norm'].min(), 
                           df['avg_command_lin_vel_xy_norm'].max(), 10)
    ang_bins = np.linspace(df['avg_command_ang_vel_z'].min(), 
                           df['avg_command_ang_vel_z'].max(), 10)
    
    df['lin_bin_2d'] = pd.cut(df['avg_command_lin_vel_xy_norm'], bins=lin_bins)
    df['ang_bin_2d'] = pd.cut(df['avg_command_ang_vel_z'], bins=ang_bins)
    
    # Linear velocity error heatmap
    pivot_lin = df.pivot_table(values='lin_vel_tracking_error', 
                                index='ang_bin_2d', columns='lin_bin_2d', aggfunc='mean')
    
    im0 = axes[0].imshow(pivot_lin.values, cmap='YlOrRd', aspect='auto', origin='lower')
    axes[0].set_xlabel('Command Linear Velocity (m/s)', fontsize=11, fontweight='bold')
    axes[0].set_ylabel('Command Angular Velocity (rad/s)', fontsize=11, fontweight='bold')
    axes[0].set_title('Linear Velocity Tracking Error', fontsize=12, fontweight='bold')
    axes[0].set_xticks(range(0, len(pivot_lin.columns), 2))
    axes[0].set_yticks(range(0, len(pivot_lin.index), 2))
    axes[0].set_xticklabels([f'{pivot_lin.columns[i].mid:.2f}' for i in range(0, len(pivot_lin.columns), 2)], rotation=45)  # type: ignore
    axes[0].set_yticklabels([f'{pivot_lin.index[i].mid:.2f}' for i in range(0, len(pivot_lin.index), 2)])  # type: ignore
    plt.colorbar(im0, ax=axes[0], label='Error (m/s)')
    
    # Angular velocity error heatmap
    pivot_ang = df.pivot_table(values='ang_vel_tracking_error', 
                                index='ang_bin_2d', columns='lin_bin_2d', aggfunc='mean')
    
    im1 = axes[1].imshow(pivot_ang.values, cmap='YlGnBu', aspect='auto', origin='lower')
    axes[1].set_xlabel('Command Linear Velocity (m/s)', fontsize=11, fontweight='bold')
    axes[1].set_ylabel('Command Angular Velocity (rad/s)', fontsize=11, fontweight='bold')
    axes[1].set_title('Angular Velocity Tracking Error', fontsize=12, fontweight='bold')
    axes[1].set_xticks(range(0, len(pivot_ang.columns), 2))
    axes[1].set_yticks(range(0, len(pivot_ang.index), 2))
    axes[1].set_xticklabels([f'{pivot_ang.columns[i].mid:.2f}' for i in range(0, len(pivot_ang.columns), 2)], rotation=45)  # type: ignore
    axes[1].set_yticklabels([f'{pivot_ang.index[i].mid:.2f}' for i in range(0, len(pivot_ang.index), 2)])  # type: ignore
    plt.colorbar(im1, ax=axes[1], label='Error (rad/s)')
    
    # Fall rate heatmap
    pivot_fall = df.pivot_table(values='fell_down', 
                                 index='ang_bin_2d', columns='lin_bin_2d', aggfunc='mean')
    
    im2 = axes[2].imshow(pivot_fall.values * 100, cmap='Reds', aspect='auto', origin='lower')
    axes[2].set_xlabel('Command Linear Velocity (m/s)', fontsize=11, fontweight='bold')
    axes[2].set_ylabel('Command Angular Velocity (rad/s)', fontsize=11, fontweight='bold')
    axes[2].set_title('Fall Rate', fontsize=12, fontweight='bold')
    axes[2].set_xticks(range(0, len(pivot_fall.columns), 2))
    axes[2].set_yticks(range(0, len(pivot_fall.index), 2))
    axes[2].set_xticklabels([f'{pivot_fall.columns[i].mid:.2f}' for i in range(0, len(pivot_fall.columns), 2)], rotation=45)  # type: ignore
    axes[2].set_yticklabels([f'{pivot_fall.index[i].mid:.2f}' for i in range(0, len(pivot_fall.index), 2)])  # type: ignore
    plt.colorbar(im2, ax=axes[2], label='Fall Rate (%)')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'command_velocity_heatmaps.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Chart saved: command_velocity_heatmaps.png")


def create_terrain_performance_analysis(df: pd.DataFrame, output_dir: str):
    """Analyze performance across different terrain types
    
    Generates:
    - terrain_performance_overview.png: Bar charts comparing metrics across 6 terrain categories (A-F)
    - terrain_error_distributions.png: Violin plots showing error distributions per terrain
    """
    
    # Aggregate by terrain category
    terrain_stats = df.groupby('terrain_category').agg({
        'lin_vel_tracking_error': ['mean', 'std'],
        'ang_vel_tracking_error': ['mean', 'std'],
        'fell_down': ['mean', 'count']
    }).reset_index()
    
    terrain_cats = sorted(df['terrain_category'].unique())
    
    # 1. Terrain performance comparison (bar charts)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Linear velocity error
    lin_means = [df[df['terrain_category']==t]['lin_vel_tracking_error'].mean() for t in terrain_cats]
    lin_stds = [df[df['terrain_category']==t]['lin_vel_tracking_error'].std() for t in terrain_cats]
    
    axes[0, 0].bar(terrain_cats, lin_means, yerr=lin_stds, capsize=5, 
                   color='steelblue', alpha=0.7, edgecolor='black', linewidth=1.5)
    axes[0, 0].set_xlabel('Terrain Type', fontsize=12, fontweight='bold')
    axes[0, 0].set_ylabel('Linear Vel Error (m/s)', fontsize=12, fontweight='bold')
    axes[0, 0].set_title('Linear Velocity Tracking Error by Terrain', fontsize=13, fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3, axis='y')
    
    # Add terrain names as text
    for i, (cat, mean) in enumerate(zip(terrain_cats, lin_means)):
        axes[0, 0].text(i, mean + lin_stds[i] + 0.01, TERRAIN_NAMES[cat], 
                        ha='center', va='bottom', fontsize=8, rotation=0)
    
    # Angular velocity error
    ang_means = [df[df['terrain_category']==t]['ang_vel_tracking_error'].mean() for t in terrain_cats]
    ang_stds = [df[df['terrain_category']==t]['ang_vel_tracking_error'].std() for t in terrain_cats]
    
    axes[0, 1].bar(terrain_cats, ang_means, yerr=ang_stds, capsize=5,
                   color='forestgreen', alpha=0.7, edgecolor='black', linewidth=1.5)
    axes[0, 1].set_xlabel('Terrain Type', fontsize=12, fontweight='bold')
    axes[0, 1].set_ylabel('Angular Vel Error (rad/s)', fontsize=12, fontweight='bold')
    axes[0, 1].set_title('Angular Velocity Tracking Error by Terrain', fontsize=13, fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3, axis='y')
    
    for i, (cat, mean) in enumerate(zip(terrain_cats, ang_means)):
        axes[0, 1].text(i, mean + ang_stds[i] + 0.01, TERRAIN_NAMES[cat],
                        ha='center', va='bottom', fontsize=8, rotation=0)
    
    # Fall rate
    fall_rates = [df[df['terrain_category']==t]['fell_down'].mean() * 100 for t in terrain_cats]
    
    axes[1, 0].bar(terrain_cats, fall_rates, color='coral', alpha=0.7, 
                   edgecolor='black', linewidth=1.5)
    axes[1, 0].set_xlabel('Terrain Type', fontsize=12, fontweight='bold')
    axes[1, 0].set_ylabel('Fall Rate (%)', fontsize=12, fontweight='bold')
    axes[1, 0].set_title('Fall Rate by Terrain', fontsize=13, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    
    for i, (cat, rate) in enumerate(zip(terrain_cats, fall_rates)):
        axes[1, 0].text(i, rate + 1, f'{rate:.1f}%\n{TERRAIN_NAMES[cat]}',
                        ha='center', va='bottom', fontsize=8)
    
    # Success rate and sample count
    success_rates = [100 - rate for rate in fall_rates]
    sample_counts = [len(df[df['terrain_category']==t]) for t in terrain_cats]
    
    ax2 = axes[1, 1].twinx()
    bars = axes[1, 1].bar(terrain_cats, success_rates, color='mediumseagreen', 
                          alpha=0.7, edgecolor='black', linewidth=1.5, label='Success Rate')
    line = ax2.plot(terrain_cats, sample_counts, 'o-', color='darkblue', 
                    linewidth=2, markersize=8, label='Sample Count')
    
    axes[1, 1].set_xlabel('Terrain Type', fontsize=12, fontweight='bold')
    axes[1, 1].set_ylabel('Success Rate (%)', fontsize=12, fontweight='bold')
    axes[1, 1].set_title('Success Rate and Sample Distribution', fontsize=13, fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    ax2.set_ylabel('Sample Count', fontsize=12, fontweight='bold')
    
    # Combined legend
    lines1, labels1 = axes[1, 1].get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    axes[1, 1].legend(lines1 + lines2, labels1 + labels2, loc='lower left')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'terrain_performance_overview.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Chart saved: terrain_performance_overview.png")
    
    # 2. Detailed violin plots for tracking errors
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Filter successful runs for violin plots
    success_df = df[df['fell_down'] == 0]
    
    if len(success_df) > 0:
        # Linear velocity error distribution
        lin_data = [success_df[success_df['terrain_category']==t]['lin_vel_tracking_error'].values  # type: ignore
                    for t in terrain_cats]
        
        parts1 = axes[0].violinplot(lin_data, positions=range(len(terrain_cats)), 
                                     showmeans=True, showmedians=True)
        for pc in parts1['bodies']:
            pc.set_facecolor('steelblue')
            pc.set_alpha(0.6)
        
        axes[0].set_xticks(range(len(terrain_cats)))
        axes[0].set_xticklabels([f'{cat}\n{TERRAIN_NAMES[cat][:10]}' for cat in terrain_cats])
        axes[0].set_xlabel('Terrain Type', fontsize=12, fontweight='bold')
        axes[0].set_ylabel('Linear Vel Error (m/s)', fontsize=12, fontweight='bold')
        axes[0].set_title('Linear Velocity Error Distribution (Successful Episodes)', fontsize=13, fontweight='bold')
        axes[0].grid(True, alpha=0.3, axis='y')
        
        # Angular velocity error distribution
        ang_data = [success_df[success_df['terrain_category']==t]['ang_vel_tracking_error'].values  # type: ignore
                    for t in terrain_cats]
        
        parts2 = axes[1].violinplot(ang_data, positions=range(len(terrain_cats)),
                                     showmeans=True, showmedians=True)
        for pc in parts2['bodies']:
            pc.set_facecolor('forestgreen')
            pc.set_alpha(0.6)
        
        axes[1].set_xticks(range(len(terrain_cats)))
        axes[1].set_xticklabels([f'{cat}\n{TERRAIN_NAMES[cat][:10]}' for cat in terrain_cats])
        axes[1].set_xlabel('Terrain Type', fontsize=12, fontweight='bold')
        axes[1].set_ylabel('Angular Vel Error (rad/s)', fontsize=12, fontweight='bold')
        axes[1].set_title('Angular Velocity Error Distribution (Successful Episodes)', fontsize=13, fontweight='bold')
        axes[1].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'terrain_error_distributions.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Chart saved: terrain_error_distributions.png")


def create_cross_analysis(df: pd.DataFrame, output_dir: str):
    """Advanced cross-analysis: terrain vs command velocity vs performance
    
    Generates:
    - terrain_difficulty_heatmap.png: Normalized difficulty metrics across terrains
    - terrain_velocity_cross_analysis.png: Fall rate by velocity category for each terrain
    - performance_consistency_analysis.png: Scatter plot of consistency vs reliability
    - performance_radar_chart.png: Multi-dimensional performance comparison across terrains
    """
    
    # 1. Terrain difficulty ranking based on multiple metrics
    terrain_cats = sorted(df['terrain_category'].unique())
    
    # Compute normalized metrics for each terrain
    metrics = {}
    for cat in terrain_cats:
        terrain_df = df[df['terrain_category'] == cat]
        success_df = terrain_df[terrain_df['fell_down'] == 0]
        
        metrics[cat] = {
            'fall_rate': terrain_df['fell_down'].mean() * 100,
            'lin_error': terrain_df['lin_vel_tracking_error'].mean(),
            'ang_error': terrain_df['ang_vel_tracking_error'].mean(),
            'success_lin_error': success_df['lin_vel_tracking_error'].mean() if len(success_df) > 0 else np.nan
        }
    
    # Create difficulty heatmap
    fig, ax = plt.subplots(figsize=(12, 8))
    
    metric_names = ['Fall Rate', 'Lin Error', 'Ang Error', 'Success\nLin Error']
    metric_keys = ['fall_rate', 'lin_error', 'ang_error', 'success_lin_error']
    
    heatmap_data = np.zeros((len(terrain_cats), len(metric_names)))
    for i, cat in enumerate(terrain_cats):
        for j, key in enumerate(metric_keys):
            heatmap_data[i, j] = metrics[cat][key]
    
    # Normalize each column for better visualization
    heatmap_normalized = np.zeros_like(heatmap_data)
    for j in range(heatmap_data.shape[1]):
        col = heatmap_data[:, j]
        if not np.all(np.isnan(col)):
            col_min, col_max = np.nanmin(col), np.nanmax(col)
            if col_max > col_min:
                heatmap_normalized[:, j] = (col - col_min) / (col_max - col_min)
    
    im = ax.imshow(heatmap_normalized, cmap='RdYlGn_r', aspect='auto', vmin=0, vmax=1)
    
    ax.set_xticks(range(len(metric_names)))
    ax.set_yticks(range(len(terrain_cats)))
    ax.set_xticklabels(metric_names, fontsize=11, fontweight='bold')
    ax.set_yticklabels([f'{cat} - {TERRAIN_NAMES[cat]}' for cat in terrain_cats], fontsize=11)
    ax.set_title('Terrain Difficulty Analysis (Normalized Metrics)', fontsize=14, fontweight='bold')
    
    # Add text annotations with actual values
    for i in range(len(terrain_cats)):
        for j in range(len(metric_names)):
            value = heatmap_data[i, j]
            if not np.isnan(value):
                text = ax.text(j, i, f'{value:.2f}',
                              ha="center", va="center", color="black", fontsize=9, fontweight='bold')
    
    plt.colorbar(im, ax=ax, label='Normalized Difficulty (0=easy, 1=hard)')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'terrain_difficulty_heatmap.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Chart saved: terrain_difficulty_heatmap.png")
    
    # 2. Command velocity performance across terrains
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # Bin command velocities
    df['vel_magnitude'] = np.sqrt(df['avg_command_lin_vel_xy_norm']**2 + 
                                   df['avg_command_ang_vel_z']**2)
    df['vel_category'] = pd.cut(df['vel_magnitude'], bins=4, labels=['Low', 'Medium', 'High', 'Very High'])
    
    vel_categories = ['Low', 'Medium', 'High', 'Very High']
    colors = ['lightgreen', 'yellow', 'orange', 'red']
    
    for idx, cat in enumerate(terrain_cats):
        row = idx // 3
        col = idx % 3
        ax = axes[row, col]
        
        terrain_df = df[df['terrain_category'] == cat]
        
        # Fall rate by velocity category
        fall_data = []
        sample_counts = []
        for vel_cat in vel_categories:
            vel_df = terrain_df[terrain_df['vel_category'] == vel_cat]
            if len(vel_df) > 0:
                fall_data.append(vel_df['fell_down'].mean() * 100)
                sample_counts.append(len(vel_df))
            else:
                fall_data.append(0)
                sample_counts.append(0)
        
        bars = ax.bar(vel_categories, fall_data, color=colors, alpha=0.7, 
                      edgecolor='black', linewidth=1.5)
        ax.set_xlabel('Command Velocity Category', fontsize=10, fontweight='bold')
        ax.set_ylabel('Fall Rate (%)', fontsize=10, fontweight='bold')
        ax.set_title(f'Terrain {cat}: {TERRAIN_NAMES[cat]}', fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim([0, max(fall_data) * 1.2 if max(fall_data) > 0 else 10])
        
        # Add sample counts on bars
        for i, (bar, count) in enumerate(zip(bars, sample_counts)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                   f'n={count}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'terrain_velocity_cross_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Chart saved: terrain_velocity_cross_analysis.png")
    
    # 3. Performance consistency analysis
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Compute CV (coefficient of variation) for each terrain
    consistency_metrics = {}
    for cat in terrain_cats:
        terrain_df = df[df['terrain_category'] == cat]
        success_df = terrain_df[terrain_df['fell_down'] == 0]
        
        if len(success_df) > 0:
            lin_mean = success_df['lin_vel_tracking_error'].mean()
            lin_std = success_df['lin_vel_tracking_error'].std()
            ang_mean = success_df['ang_vel_tracking_error'].mean()
            ang_std = success_df['ang_vel_tracking_error'].std()
            
            consistency_metrics[cat] = {
                'lin_cv': (lin_std / lin_mean * 100) if lin_mean > 0 else 0,
                'ang_cv': (ang_std / ang_mean * 100) if ang_mean > 0 else 0,
                'fall_rate': terrain_df['fell_down'].mean() * 100
            }
    
    # Scatter plot: consistency (CV) vs performance (fall rate)
    for cat in terrain_cats:
        if cat in consistency_metrics:
            m = consistency_metrics[cat]
            avg_cv = (m['lin_cv'] + m['ang_cv']) / 2
            ax.scatter(avg_cv, m['fall_rate'], s=200, alpha=0.6, label=f"{cat} - {TERRAIN_NAMES[cat]}")
            ax.annotate(cat, (avg_cv, m['fall_rate']), fontsize=12, fontweight='bold',
                       ha='center', va='center')
    
    ax.set_xlabel('Performance Consistency (Avg CV of Errors, %)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Fall Rate (%)', fontsize=12, fontweight='bold')
    ax.set_title('Terrain Performance: Consistency vs Reliability', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=9)
    
    # Add quadrant lines
    if len(consistency_metrics) > 0:
        avg_cv_all = np.mean([m['lin_cv'] + m['ang_cv'] for m in consistency_metrics.values()]) / 2
        avg_fall_all = np.mean([m['fall_rate'] for m in consistency_metrics.values()])
        ax.axvline(avg_cv_all, color='gray', linestyle='--', alpha=0.5, linewidth=2)
        ax.axhline(avg_fall_all, color='gray', linestyle='--', alpha=0.5, linewidth=2)
        
        # Add quadrant labels
        ax.text(ax.get_xlim()[0] + 1, ax.get_ylim()[1] - 2, 'Consistent\nBut Risky', 
                fontsize=10, style='italic', alpha=0.6)
        ax.text(ax.get_xlim()[1] - 5, ax.get_ylim()[1] - 2, 'Inconsistent\n& Risky',
                fontsize=10, style='italic', alpha=0.6)
        ax.text(ax.get_xlim()[0] + 1, ax.get_ylim()[0] + 2, 'Consistent\n& Reliable',
                fontsize=10, style='italic', alpha=0.6)
        ax.text(ax.get_xlim()[1] - 5, ax.get_ylim()[0] + 2, 'Inconsistent\nBut Safer',
                fontsize=10, style='italic', alpha=0.6)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'performance_consistency_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Chart saved: performance_consistency_analysis.png")
    
    # 4. Comprehensive performance radar chart
    if len(terrain_cats) <= 6:
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
        
        # Normalize metrics for radar chart
        radar_metrics = {
            'Success Rate': {},
            'Lin Vel Accuracy': {},
            'Ang Vel Accuracy': {},
            'Consistency': {}
        }
        
        for cat in terrain_cats:
            terrain_df = df[df['terrain_category'] == cat]
            success_df = terrain_df[terrain_df['fell_down'] == 0]
            
            success_rate = (1 - terrain_df['fell_down'].mean()) * 100
            lin_accuracy = 100 if len(success_df) == 0 else max(0, 100 - success_df['lin_vel_tracking_error'].mean() * 100)
            ang_accuracy = 100 if len(success_df) == 0 else max(0, 100 - success_df['ang_vel_tracking_error'].mean() * 50)
            
            lin_cv = success_df['lin_vel_tracking_error'].std() / success_df['lin_vel_tracking_error'].mean() if len(success_df) > 0 else 1
            consistency = max(0, 100 - lin_cv * 100)
            
            radar_metrics['Success Rate'][cat] = success_rate
            radar_metrics['Lin Vel Accuracy'][cat] = lin_accuracy
            radar_metrics['Ang Vel Accuracy'][cat] = ang_accuracy
            radar_metrics['Consistency'][cat] = consistency
        
        # Plot radar chart
        categories = list(radar_metrics.keys())
        N = len(categories)
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
        angles += angles[:1]
        
        colors_radar = plt.cm.Set2(np.linspace(0, 1, len(terrain_cats)))
        
        for idx, cat in enumerate(terrain_cats):
            values = [radar_metrics[metric][cat] for metric in categories]
            values += values[:1]
            
            ax.plot(angles, values, 'o-', linewidth=2, label=f'{cat} - {TERRAIN_NAMES[cat]}', 
                   color=colors_radar[idx])
            ax.fill(angles, values, alpha=0.15, color=colors_radar[idx])
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, fontsize=11, fontweight='bold')
        ax.set_ylim(0, 100)
        ax.set_title('Comprehensive Performance Radar Chart', fontsize=14, fontweight='bold', pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=9)
        ax.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'performance_radar_chart.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Chart saved: performance_radar_chart.png")


def main():
    parser = argparse.ArgumentParser(description="Analyze performance test results")
    parser.add_argument("csv_path", type=str, help="Performance test CSV file path")
    parser.add_argument("--output_dir", type=str, default=None, 
                       help="Output directory (default: same as CSV file directory)")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.csv_path):
        print(f"Error: File not found {args.csv_path}")
        return
    
    analyze_performance_csv(args.csv_path, args.output_dir)
    print("\nAnalysis completed!")


if __name__ == "__main__":
    main()

